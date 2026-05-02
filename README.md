# Automated Asset & Threat Extraction from Logs for ICT Risk Assessment using Lightweight LLMs

## **Abstract**

Manual ICT risk assessment is often slow, subjective, and hard to scale because it depends heavily on people manually reviewing systems, interpreting risks differently, and repeating the same work every time the environment changes. Based on that problem, I propose a project where I’ll build a Python tool that does the following:

- Parses semi‑structured log entries (syslog, web logs, or CVE text)
- Extracts assets and threats using a lightweight transformer model
- Computes a baseline risk score using a configurable likelihood × impact matrix (aligned with NIST SP 800‑30 / ISO 27005)
- Outputs a structured risk register (CSV) for human review
  
The tool is designed as a decision support system to accelerate initial risk assessment but does not replace expert judgment.

## **Features**

- **Hybrid parsing** – regex pre‑cleaning + LLM extraction (reduces noise and tokens)
- **Multiple lightweight LLM backends** – SmolLM2‑360M, Qwen2.5‑3B, TinyLlama‑1.1B
- **Uniform JSON output** – all models produce `{"asset": "...", "threat": "...", "confidence": "..."}`
- **Configurable risk matrix** – asset criticality, likelihood per threat type (YAML)
- **Confidence filtering** – flags low‑confidence extractions for manual review (mitigates hallucinations)
- **Standardised output** – CSV risk register with `requires_review` column
- **Multi-model comparison** – evaluation scripts using the same test set and metrics

## **Installation**

### 1. Clone the repository

```bash
git clone https://github.com/0m0D1NGR0NALD/Automated-Asset-and-Threat-Extraction-from-Logs-for-ICT-Risk-Assessment-using-Lightweight-LLMs-.git
cd Automated-Asset-and-Threat-Extraction-from-Logs-for-ICT-Risk-Assessment-using-Lightweight-LLMs-
```

### 2. Set up a virtual environment
```bash
python -m venv ictra
source ictra/bin/activate # Linux/macOS
ictra\Scripts\activate # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## **Usage**

### Preprocess CSIC 2010 dataset (optional)
```bash
python scripts/generate_ground_truth_from_csic_csv.py
```
Creates a balanced sample (500 normal + 500 anomalous) for running experiments. This is the output:

```data/datasets/ground_truth_sample.csv``` – contains two columns: raw_log (Method + URL + content), classification (Normal or Anomalous)

```data/datasets/csic_database_sample.txt``` – the raw_log (Method + URL + content) for the LLM to process


### Running Individual Models
```bash
# SmolLM2-360M
python -m src.main --model smolLM2 -i data/datasets/csic_database_sample.txt -o results/smolLM2_results.csv

# Qwen2.5-3B (requires GPU with 4‑bit quantisation)
python -m src.main --model qwen -i data/datasets/csic_database_sample.txt -o results/qwen_results.csv

# TinyLlama-1.1B (runs on CPU)
python -m src.main --model tinyllama -i data/datasets/csic_database_sample.txt -o results/tinyllama_results.csv
```

### Batch Execution
Run all three models sequentially and generate a comparison report.

#### Windows PowerShell:
```bash
.\run_all_models.ps1
```

#### Linux / Mac:
```bash
chmod +x ./run_all_models.sh
./run_all_models.sh
```

### Model Comparison & Evaluation
Unified evaluation scripts to compare LLMs on the same data sample.


#### Evaluate a single model
```bash
python experiments/evaluate_model.py --risk_csv results/smolLM2_results.csv --ground_truth data/datasets/ground_truth_sample.csv --model_name SmolLM2
```

#### Compare all models side‑by‑side
```bash
python experiments/compare_models.py --models SmolLM2:results/smolLM2_results.csv Qwen:results/qwen_results.csv TinyLlama:results/tinyllama_results.csv --ground_truth data/datasets/ground_truth_sample.csv --output comparison_report.txt
```

#### Run the Demo
```bash
python demo.py data/demo_data/csic_demo_sample.txt results/demo_results.csv
```

## **Foundational Literature**

1. Jeong H, Joe I. "An AI-Based Risk Analysis Framework Using Large Language Models for Web Log Security." Electronics 2025, 14, 3512. https://doi.org/10.3390/electronics14173512

2. N. M. Unal and B. Celiktas, "Automating Cyber Risk Assessment With Public LLMs: An Expert-Validated Framework and Comparative Analysis," in IEEE Access, vol. 14, pp. 47754-47778, 2026, doi: 10.1109/ACCESS.2026.3678044.

3. Chalyi O, Driaunys K, Grigaliūnas Š, Brūzgienė R. "Standard-Oriented Architecture for AI-Powered Information Security Risk Management." Electronics. 2026; 15(6):1282. https://doi.org/10.3390/electronics15061282

4. Karlsen, E., Luo, X., Zincir-Heywood, N. et al. "Benchmarking Large Language Models for Log Analysis, Security, and Interpretation." J Netw Syst Manage 32, 59 (2024). https://doi.org/10.1007/s10922-024-09831-x

5. Shetaia, Amir, and Sean Kauffman. "DeepParse: Hybrid Log Parsing with LLM-Synthesized Regex Masks." arXiv preprint arXiv:2604.20553 (2026).
