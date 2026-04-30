GPT_EXTRACTION_PROMPT = """
You are a cyber risk analyst. Extract from the following log or CVE text:
- The asset affected (choose one: public_web_server, internal_database, developer_workstation, test_environment, unknown)
- The threat type (choose one: sql_injection, xss, dos, brute_force, privilege_escalation, malicious_file, info_leak)
- Your confidence (0-1)

Text: {text}

Return JSON exactly like: {{"asset": "...", "threat": "...", "confidence": 0.xx}}
"""