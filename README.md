# Automated-Asset-Threat-Extraction-from-Logs-for-ICT-Risk-Assessment-using-Lightweight-LLMs-

**Abstract**
Manual ICT risk assessment is often slow, subjective, and hard to scale because it depends heavily on people manually reviewing systems, interpreting risks differently, and repeating the same work every time the environment changes. Based on that problem, I propose a project where I’ll build a Python tool that does the following:

- Parses semi‑structured log entries (syslog, web logs, or CVE text)
- Extracts assets and threats using a lightweight transformer model (DistilRoBERTa or GPT‑4o‑mini)
- Computes a baseline risk score using a configurable likelihood × impact matrix (aligned with NIST SP 800‑30 / ISO 27005)
- Outputs a structured risk register (CSV) for human review
- 
The tool is designed as a decision support system to accelerate initial risk assessment but does not replace expert judgment.
