import json
import re
import torch
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class TinyLlamaExtractor(BaseExtractor):
    """
    Extractor using TinyLlama-1.1B-Chat.
    Outputs JSON: {"asset": "...", "threat": "...", "confidence": 0.xx}
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading TinyLlama model {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def _build_prompt(self, text: str) -> str:
        """Build a chat prompt for TinyLlama."""
        truncated = text[:1500]

        prompt = f"""<|system|>
You are a cybersecurity risk analyst. Analyze the HTTP request or CVE text.
Classify the asset and threat. Return ONLY valid JSON.

Possible assets: public_web_server, internal_database, developer_workstation, test_environment, IoT_device, unknown.
Possible threats: sql_injection, xss, dos, brute_force, privilege_escalation, malicious_file, info_leak, command_and_control, port_scan, unknown.

Answer format: {{"asset": "...", "threat": "...", "confidence": 0.xx}}
<|user|>
Log: {truncated}
<|assistant|>
"""
        return prompt

    def _parse_response(self, raw_output: str) -> Dict[str, Any]:
        # Remove possible extra spaces or newlines
        raw_output = raw_output.strip()
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if "asset" in data and "threat" in data and "confidence" in data:
                    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
                    return data
                else:
                    logger.warning(f"Missing keys in JSON: {data}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}\nRaw: {raw_output[:200]}")
        else:
            logger.warning(f"No JSON found: {raw_output[:200]}")

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

        logger.debug(f"TinyLlama response: {response}")
        return self._parse_response(response)