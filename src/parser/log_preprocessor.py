import re
from typing import Tuple, Optional

class LogPreprocessor:
    """Cleans raw log lines using regex. Extracts timestamp, IP, and the core message."""
    
    TIMESTAMP_PATTERNS = [
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO
        r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}', # Syslog style
        r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}', # Apache
    ]
    
    IP_PATTERN = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    
    @classmethod
    def clean_line(cls, line: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Returns: (cleaned_message, timestamp, ip_address)
        """
        original = line.strip()
        if not original:
            return "", None, None
        
        # Extract timestamp
        timestamp = None
        for pattern in cls.TIMESTAMP_PATTERNS:
            match = re.search(pattern, original)
            if match:
                timestamp = match.group(0)
                break
        
        # Extract IP
        ip_match = re.search(cls.IP_PATTERN, original)
        ip = ip_match.group(0) if ip_match else None
        
        # Remove timestamp and IP from message for cleaner input to model
        cleaned = original
        if timestamp:
            cleaned = cleaned.replace(timestamp, '')
        if ip:
            cleaned = cleaned.replace(ip, '[IP]')  # anonymize but keep placeholder
        
        # Remove extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned, timestamp, ip