import yaml
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.cfg
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value