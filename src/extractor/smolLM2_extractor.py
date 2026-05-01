import torch
import json
import re
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

class SmolLM2Extractor(BaseExtractor):
    """
    Concrete extractor using the official, domain-specialized SmolLM2-360M-Instruct
    model, fine-tuned on security logs (SecInt-SmolLM2-360M-nginx).
    Now uses uniform JSON output format for fair comparison.
    """
    def __init__(
        self,
        model_name: str = "LeviDeHaan/SecInt-SmolLM2-360M-nginx",
        device: str = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        max_length: int = 512
    ):
        """
        Initialize the extractor with the specified SmolLM2 model.
        :param model_name: Official Hugging Face model identifier.
        :param device: 'cuda', 'cpu', or None (auto-detect).
        :param max_new_tokens: Maximum tokens to generate for the response.
        :param temperature: Sampling temperature (0 = greedy).
        :param max_length: Maximum input length for truncation.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_length = max_length

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading SmolLM2 model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model (no quantization for this small model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        logger.info("SmolLM2 model loaded successfully.")

    def _build_prompt(self, text: str) -> str:
        """Build an instruction prompt that asks for JSON output."""
        truncated = text[:self.max_length]

        prompt = f"""<|system|>
You are a cybersecurity risk analyst. Analyze the following HTTP request log or CVE description.
Classify the asset and threat, and give a confidence score.

Possible assets: public_web_server, internal_database, developer_workstation, test_environment, IoT_device, unknown.
Possible threats: sql_injection, xss, dos, brute_force, privilege_escalation, malicious_file, info_leak, command_and_control, port_scan, unknown.

Answer ONLY with valid JSON in this format:
{{"asset": "...", "threat": "...", "confidence": 0.xx}}

Do not include any extra text.<|im_end|>
<|user|>
Log: {truncated}<|im_end|>
<|assistant|>
"""
        return prompt

    def _parse_response(self, raw_output: str) -> Dict[str, Any]:
        """Extract JSON from the model's output."""
        # Remove any trailing special tokens
        raw_output = raw_output.strip()
        # Find JSON object
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if "asset" in data and "threat" in data and "confidence" in data:
                    # Clamp confidence to [0,1]
                    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
                    logger.debug(f"Parsed JSON: {data}")
                    return data
                else:
                    logger.warning(f"Incomplete JSON: {data}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}\nRaw: {raw_output[:200]}")
        else:
            logger.warning(f"No JSON found in output: {raw_output[:200]}")

        # Fallback with low confidence
        return {"asset": "unknown", "threat": "unknown", "confidence": 0.3}

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, Any]:
        prompt = self._build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=(self.temperature > 0),
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        logger.debug(f"SmolLM2 response: {response}")
        return self._parse_response(response)