import torch
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

class SmolLM2Extractor(BaseExtractor):
    """
    Concrete extractor using the official, domain-specialized SmolLM2-360M-Instruct
    model, fine-tuned on security logs (SecInt-SmolLM2-360M-nginx).
    """
    def __init__(self, model_name="LeviDeHaan/SecInt-SmolLM2-360M-nginx", max_length=512):
        """
        Initialize the extractor with the specified SmolLM2 model.
        :param model_name: Official Hugging Face model identifier.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length

        logger.info(f"Loading concrete SmolLM2 model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def _build_prompt(self, text):
        """Builds the prompt to classify the log as hack, error, or normal."""
        truncated_text = text[:self.max_length]
        system_prompt = """You are a security log analyzer. Classify the log entry as one of: hack, error, or normal.
        HACK - Any of these patterns indicate an attack:
        - Scanning for sensitive files: .env, .git, .php, config.php, wp-admin, phpmyadmin
        - SQL injection attempts, XSS attempts
        - Invalid login attempts, brute force, "invalid user", "failed password"
        - Exploit attempts, malformed requests
        - 403/404 errors with suspicious paths like .env, .git, admin, wp-, .php
        - Scanner user-agents: sqlmap, nikto, zgrab, nuclei
        - Webshell access attempts
        ERROR - Application errors:
        - 500 errors, crashes, exceptions
        - SSL/TLS errors
        - Database connection failures, [emerg], [alert], [crit], [error] log levels
        NORMAL - Everything else:
        - 200/304 responses to legitimate paths
        - Regular API calls, static files
        - Known good bots: googlebot, facebookbot
        Respond with only one word: hack, error, or normal."""

        user_prompt = f"Classify this log entry as hack, error, or normal.\n\n{truncated_text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _parse_response(self, raw_output: str) -> Dict[str, Any]:
        """Parses the model's output to extract the classification and a confidence score."""
        output = raw_output.strip().lower()
        if "hack" in output:
            classification = "hack"
        elif "error" in output:
            classification = "error"
        else:
            classification = "normal"

        # Map the classification to your project's asset/threat ontology
        asset = "public_web_server" if classification in ["hack", "error"] else "test_environment"
        threat = "sql_injection" if classification == "hack" else "info_leak"
        confidence = 0.9 if classification in ["hack", "error"] else 0.7

        logger.info(f"Input classified as: {classification}, mapped to Asset: {asset}, Threat: {threat}")
        return {
            "asset": asset,
            "threat": threat,
            "confidence": confidence
        }

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, Any]:
        prompt = self._build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.01,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._parse_response(response)