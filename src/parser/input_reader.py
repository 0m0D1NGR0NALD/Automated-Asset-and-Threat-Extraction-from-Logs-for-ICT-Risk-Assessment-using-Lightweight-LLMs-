import json
import csv
import os
from typing import List, Dict, Any

class InputReader:
    @staticmethod
    def read(file_path: str) -> List[Dict[str, Any]]:
        """
        Returns list of dicts with at least a 'raw' field.
        Also keeps original context.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.json':
            return InputReader._read_json(file_path)
        elif ext == '.csv':
            return InputReader._read_csv(file_path)
        else:
            return InputReader._read_text(file_path)
    
    @staticmethod
    def _read_text(file_path: str) -> List[Dict]:
        entries = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    entries.append({"raw": line.strip()})
        return entries
    
    @staticmethod
    def _read_json(file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [{"raw": json.dumps(item) if isinstance(item, dict) else str(item)} for item in data]
        else:
            return [{"raw": json.dumps(data)}]
    
    @staticmethod
    def _read_csv(file_path: str) -> List[Dict]:
        entries = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # if there's a 'message' or 'raw' column use it, else take first column
                if 'message' in row:
                    raw = row['message']
                elif 'raw' in row:
                    raw = row['raw']
                else:
                    raw = list(row.values())[0]
                entries.append({"raw": raw, **row})
        return entries