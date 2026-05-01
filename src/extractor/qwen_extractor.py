import json
import re
import torch
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class QwenExtractor(BaseExtractor):
    """
    Extractor using Qwen2.5-3B-Instruct (instruction‑tuned).
    Outputs JSON: {"asset": "...", "threat": "...", "confidence": 0.xx}
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = None,
        use_4bit: bool = True,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Qwen model {model_name} on {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantisation config (if CUDA)
        model_kwargs = {"device_map": "auto" if self.device == "cuda" else None}
        if use_4bit and self.device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                logger.info("Using 4‑bit quantisation (Qwen)")
            except ImportError:
                logger.warning("bitsandbytes not installed – loading in full precision")
        else:
            logger.info("Loading Qwen without quantisation (may use large memory)")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            **model_kwargs,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, text: str) -> str:
        """Build an instruction prompt for Qwen."""
        truncated = text[:1500]  # stay within context window

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
        """Extract JSON from model output."""
        # Remove any <|im_end|> or trailing text
        raw_output = raw_output.split("<|im_end|>")[0].strip()
        # Find JSON block
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if "asset" in data and "threat" in data and "confidence" in data:
                    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
                    return data
                else:
                    logger.warning(f"Incomplete JSON: {data}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}\nRaw: {raw_output[:200]}")
        else:
            logger.warning(f"No JSON found in output: {raw_output[:200]}")

        # Fallback
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

        logger.debug(f"Qwen response: {response}")
        return self._parse_response(response)