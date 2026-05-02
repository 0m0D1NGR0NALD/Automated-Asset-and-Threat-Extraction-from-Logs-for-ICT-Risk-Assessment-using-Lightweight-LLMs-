import torch
import json
import re
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

class TinyLlamaExtractor(BaseExtractor):
    """
    Extractor using TinyLlama-1.1B-Chat.
    Handles both JSON and natural language key‑value output.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
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

        logger.info(f"Loading TinyLlama model {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

    def _build_prompt(self, text: str) -> str:
        """Build a chat prompt that asks for JSON, but the model may still output key‑value."""
        truncated = text[:self.max_length]

        prompt = f"""<|system|>
You are a cybersecurity risk analyst. Analyze the HTTP request or CVE text.
Output ONLY a JSON object with these three fields: asset, threat, confidence.
Possible assets: public_web_server, internal_database, developer_workstation, test_environment, IoT_device, unknown.
Possible threats: sql_injection, xss, dos, brute_force, privilege_escalation, malicious_file, info_leak, command_and_control, port_scan, unknown.

Example output: {{"asset": "public_web_server", "threat": "sql_injection", "confidence": 0.95}}
Do not include any extra text.<|im_end|>
<|user|>
Log: {truncated}<|im_end|>
<|assistant|>
"""
        return prompt

    def _parse_response(self, raw_output: str) -> Dict[str, Any]:
        """Parse either JSON or natural language key‑value pairs."""
        raw_output = raw_output.strip()

        # 1. Try to parse as JSON
        try:
            # Find first JSON object
            start = raw_output.find('{')
            if start != -1:
                # simple bracket matching
                brace_count = 0
                end = start
                for i, ch in enumerate(raw_output[start:], start):
                    if ch == '{':
                        brace_count += 1
                    elif ch == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            break
                if end > start:
                    json_str = raw_output[start:end+1]
                    data = json.loads(json_str)
                    if "asset" in data and "threat" in data and "confidence" in data:
                        data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
                        return data
        except (json.JSONDecodeError, ValueError):
            pass

        # 2. Fallback: extract key-value pairs (case‑insensitive)
        asset = "unknown"
        threat = "unknown"
        confidence = 0.5

        # Patterns
        asset_pattern = re.compile(r'asset:\s*([^\n]+)', re.IGNORECASE)
        threat_pattern = re.compile(r'threat:\s*([^\n]+)', re.IGNORECASE)
        conf_pattern = re.compile(r'confidence:\s*([0-9.]+)', re.IGNORECASE)

        a_match = asset_pattern.search(raw_output)
        if a_match:
            asset = a_match.group(1).strip().lower()
            # map to known assets if needed
            if asset not in ["public_web_server", "internal_database", "developer_workstation", "test_environment", "iot_device", "unknown"]:
                # heuristic mapping
                if "web" in asset or "server" in asset:
                    asset = "public_web_server"
                elif "database" in asset:
                    asset = "internal_database"
                elif "workstation" in asset:
                    asset = "developer_workstation"
                elif "test" in asset:
                    asset = "test_environment"
                else:
                    asset = "unknown"

        t_match = threat_pattern.search(raw_output)
        if t_match:
            threat = t_match.group(1).strip().lower()
            threat = threat.replace(" ", "_")  # e.g., "sql injection" -> "sql_injection"
            # ensure it's in the allowed list
            allowed_threats = ["sql_injection", "xss", "dos", "brute_force", "privilege_escalation",
                               "malicious_file", "info_leak", "command_and_control", "port_scan", "unknown"]
            if threat not in allowed_threats:
                # heuristic mapping
                if "sql" in threat:
                    threat = "sql_injection"
                elif "cross" in threat or "script" in threat:
                    threat = "xss"
                elif "brute" in threat:
                    threat = "brute_force"
                else:
                    threat = "unknown"

        c_match = conf_pattern.search(raw_output)
        if c_match:
            try:
                confidence = float(c_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

        logger.info(f"Parsed from natural language → Asset: {asset}, Threat: {threat}, Conf: {confidence}")
        return {"asset": asset, "threat": threat, "confidence": confidence}

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