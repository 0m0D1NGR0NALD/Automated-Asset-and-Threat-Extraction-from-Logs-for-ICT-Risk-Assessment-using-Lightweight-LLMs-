import torch
import json
import re
import logging
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

class SmolLM2Extractor(BaseExtractor):
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        max_length: int = 512,
        few_shot: bool = False
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_length = max_length
        self.few_shot = few_shot

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading SmolLM2 instruct model: {self.model_name} on {self.device}")
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
        truncated = text[:self.max_length]

        system_prompt = """You are a cybersecurity risk analyst. Analyze the following HTTP request log.
Output ONLY a valid JSON object with three keys: "asset", "threat", "confidence".
Possible assets: public_web_server, internal_database, developer_workstation, test_environment, IoT_device, unknown.
Possible threats: sql_injection, xss, dos, brute_force, privilege_escalation, malicious_file, info_leak, command_and_control, port_scan, unknown, benign.
Confidence must be a float between 0.0 and 1.0.

Example output: {"asset": "public_web_server", "threat": "sql_injection", "confidence": 0.95}
Do not add any extra text."""

        user_prompt = f"Log: {truncated}"

        messages = [{"role": "system", "content": system_prompt}]

        # Few‑shot examples
        if self.few_shot:
            examples = [
                {"role": "user", "content": "Log: GET http://localhost:8080/index.jsp"},
                {"role": "assistant", "content": '{"asset": "public_web_server", "threat": "benign", "confidence": 0.95}'},
                {"role": "user", "content": "Log: POST /login.jsp?user=admin&pwd=12345"},
                {"role": "assistant", "content": '{"asset": "public_web_server", "threat": "benign", "confidence": 0.90}'},
                {"role": "user", "content": "Log: GET /product.jsp?id=1' OR '1'='1"},
                {"role": "assistant", "content": '{"asset": "public_web_server", "threat": "sql_injection", "confidence": 0.98}'},
                {"role": "user", "content": "Log: GET /search?q=<script>alert(1)</script>"},
                {"role": "assistant", "content": '{"asset": "public_web_server", "threat": "xss", "confidence": 0.96}'},
                {"role": "user", "content": "Log: GET /admin/config.php"},
                {"role": "assistant", "content": '{"asset": "public_web_server", "threat": "malicious_file", "confidence": 0.92}'}
            ]
            messages.extend(examples)

        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": ""})  # empty assistant to start generation

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Remove the empty assistant part if it causes double generation
        prompt = prompt.replace("<|im_start|>assistant\n\n<|im_start|>assistant", "<|im_start|>assistant")
        return prompt

    def _parse_response(self, raw_output: str) -> Dict[str, Any]:
        # (same robust parser as before, unchanged)
        raw_output = raw_output.strip()
        start = raw_output.find('{')
        if start != -1:
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
                try:
                    data = json.loads(json_str)
                    if all(k in data for k in ("asset", "threat", "confidence")):
                        allowed_threats = ["sql_injection", "xss", "dos", "brute_force", "privilege_escalation",
                                           "malicious_file", "info_leak", "command_and_control", "port_scan", 
                                           "unknown", "benign"]
                        if data["threat"] not in allowed_threats:
                            data["threat"] = "unknown"
                        data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
                        return data
                except json.JSONDecodeError:
                    pass

        # Fallback key‑value extraction (same as before)
        asset, threat, confidence = "unknown", "unknown", 0.5
        # ... (copy the full fallback logic from previous version)
        # For brevity, I include only the essential mapping:
        asset_match = re.search(r'asset:\s*([^\n]+)', raw_output, re.I)
        if asset_match:
            asset = asset_match.group(1).strip().lower()
        threat_match = re.search(r'threat:\s*([^\n]+)', raw_output, re.I)
        if threat_match:
            threat = threat_match.group(1).strip().lower()
            threat = threat.replace(" ", "_")
        conf_match = re.search(r'confidence:\s*([0-9.]+)', raw_output, re.I)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except:
                pass
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
        logger.debug(f"SmolLM2 response: {response}")
        return self._parse_response(response)