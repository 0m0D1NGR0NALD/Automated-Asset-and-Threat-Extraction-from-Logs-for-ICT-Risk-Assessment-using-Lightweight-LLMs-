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
    Extractor using the instruction‑tuned SmolLM2-360M-Instruct model.
    Outputs JSON directly, same as Qwen and TinyLlama.
    """
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        device: str = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading SmolLM2 instruct model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        logger.info("SmolLM2 model loaded successfully.")

    def _build_prompt(self, text: str) -> str:
        """Same prompt as Qwen/TinyLlama for uniform JSON output."""
        truncated = text[:self.max_length]

        prompt = f"""<|im_start|>system
You are a cybersecurity risk analyst. Analyze the following HTTP request log or CVE description.
Classify the asset and threat, and give a confidence score.

Possible assets: public_web_server, internal_database, developer_workstation, test_environment, IoT_device, unknown.
Possible threats: sql_injection, xss, dos, brute_force, privilege_escalation, malicious_file, info_leak, command_and_control, port_scan, unknown.

Answer ONLY with valid JSON in this format:
{{"asset": "...", "threat": "...", "confidence": 0.xx}}

Do not include any extra text.<|im_end|>
<|im_start|>user
Log: {truncated}<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def _parse_response(self, raw_output: str) -> Dict[str, Any]:
        """Extract JSON from the model's output."""
        raw_output = raw_output.strip()
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if "asset" in data and "threat" in data and "confidence" in data:
                    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
                    logger.debug(f"Parsed JSON: {data}")
                    return data
                else:
                    logger.warning(f"Incomplete JSON: {data}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}\nRaw: {raw_output[:200]}")
        else:
            logger.warning(f"No JSON found in output: {raw_output[:200]}")

        # Fallback – low confidence unknown
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

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        logger.debug(f"SmolLM2 response: {response}")
        return self._parse_response(response)