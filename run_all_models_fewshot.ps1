$INPUT_FILE = "data/datasets/csic_database_sample.txt"
$GROUND_TRUTH = "data/datasets/ground_truth_sample.csv"

if (!(Test-Path "results")) { New-Item -ItemType Directory -Path "results" }

Write-Host "Running SmolLM2..."
python -m src.main --model smolLM2 --few-shot -i $INPUT_FILE -o results/smolLM2_fewshot.csv

Write-Host "Running Qwen..."
python -m src.main --model qwen --few-shot -i $INPUT_FILE -o results/qwen_fewshot.csv

Write-Host "Running TinyLlama..."
python -m src.main --model tinyllama --few-shot -i $INPUT_FILE -o results/tinyllama_fewshot.csv

Write-Host "Comparing all models (few-shot)..."
python experiments/compare_models.py `
    --models SmolLM2:results/smolLM2_fewshot.csv Qwen:results/qwen_fewshot.csv TinyLlama:results/tinyllama_fewshot.csv `
    --ground_truth $GROUND_TRUTH `
    --output results/comparison_report_fewshot.txt

Write-Host "Done."