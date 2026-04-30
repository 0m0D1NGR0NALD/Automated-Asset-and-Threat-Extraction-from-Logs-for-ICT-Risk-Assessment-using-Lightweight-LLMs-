# Automated Asset & Threat Extraction from Logs for ICT Risk Assessment using Lightweight LLMs

## **Abstract**

Manual ICT risk assessment is often slow, subjective, and hard to scale because it depends heavily on people manually reviewing systems, interpreting risks differently, and repeating the same work every time the environment changes. Based on that problem, I propose a project where I’ll build a Python tool that does the following:

- Parses semi‑structured log entries (syslog, web logs, or CVE text)
- Extracts assets and threats using a lightweight transformer model (DistilRoBERTa or GPT‑4o‑mini)
- Computes a baseline risk score using a configurable likelihood × impact matrix (aligned with NIST SP 800‑30 / ISO 27005)
- Outputs a structured risk register (CSV) for human review
  
The tool is designed as a decision support system to accelerate initial risk assessment but does not replace expert judgment.

## **Foundational Literature**

1. Jeong H, Joe I. "An AI-Based Risk Analysis Framework Using Large Language Models for Web Log Security." Electronics 2025, 14, 3512. https://doi.org/10.3390/electronics14173512

2. N. M. Unal and B. Celiktas, "Automating Cyber Risk Assessment With Public LLMs: An Expert-Validated Framework and Comparative Analysis," in IEEE Access, vol. 14, pp. 47754-47778, 2026, doi: 10.1109/ACCESS.2026.3678044.

3. Chalyi O, Driaunys K, Grigaliūnas Š, Brūzgienė R. "Standard-Oriented Architecture for AI-Powered Information Security Risk Management." Electronics. 2026; 15(6):1282. https://doi.org/10.3390/electronics15061282

4. Karlsen, E., Luo, X., Zincir-Heywood, N. et al. "Benchmarking Large Language Models for Log Analysis, Security, and Interpretation." J Netw Syst Manage 32, 59 (2024). https://doi.org/10.1007/s10922-024-09831-x

5. Shetaia, Amir, and Sean Kauffman. "DeepParse: Hybrid Log Parsing with LLM-Synthesized Regex Masks." arXiv preprint arXiv:2604.20553 (2026).
