import openai
from .base_extractor import BaseExtractor
from .prompt_templates import GPT_EXTRACTION_PROMPT
import json
import logging

logger = logging.getLogger(__name__)

class GPTExtractor(BaseExtractor):
    def __init__(self, api_key, model="gpt-4o-mini"):
        openai.api_key = api_key
        self.model = model
    
    def extract(self, text: str) -> dict:
        prompt = GPT_EXTRACTION_PROMPT.format(text=text[:2000])
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return {
                "asset": result.get("asset", "unknown"),
                "threat": result.get("threat", "unknown"),
                "confidence": result.get("confidence", 0.5)
            }
        except Exception as e:
            logger.error(f"GPT extraction failed: {e}")
            return {"asset": "unknown", "threat": "unknown", "confidence": 0.0}