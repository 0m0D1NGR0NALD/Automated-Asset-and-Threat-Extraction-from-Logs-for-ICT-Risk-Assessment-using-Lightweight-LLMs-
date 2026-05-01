INPUT_FILE="data/datasets/csic_database_sample.txt"
GROUND_TRUTH="data/datasets/ground_truth_sample.csv"

echo "Running SmolLM2..."
python -m src.main --model smolLM2 -i $INPUT_FILE -o smolLM2_results.csv

echo "Running Qwen..."
python -m src.main --model qwen -i $INPUT_FILE -o qwen_results.csv

echo "Running TinyLlama..."
python -m src.main --model tinyllama -i $INPUT_FILE -o tinyllama_results.csv

echo "Comparing all models..."
python compare_models_uniform.py \
    --models SmolLM2:smolLM2_results.csv Qwen:qwen_results.csv TinyLlama:tinyllama_results.csv \
    --ground_truth $GROUND_TRUTH \
    --output comparison_report.txt

echo "Done."