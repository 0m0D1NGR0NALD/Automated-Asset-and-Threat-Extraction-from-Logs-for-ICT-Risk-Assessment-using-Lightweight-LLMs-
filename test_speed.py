# Test inference speed of all three models
import time
from src.extractor.smolLM2_extractor import SmolLM2Extractor
from src.extractor.qwen_extractor import QwenExtractor
from src.extractor.tinyllama_extractor import TinyLlamaExtractor

# Test log
test_log = "GET http://localhost:8080/index.jsp HTTP/1.1"

# Model configurations
models = [
    ("SmolLM2-360M", SmolLM2Extractor, {"few_shot": False}),
    ("Qwen2.5-3B", QwenExtractor, {"few_shot": False, "use_4bit": True}),
    ("TinyLlama-1.1B", TinyLlamaExtractor, {"few_shot": False})
]

print("Model Speed Test (CPU)")

for name, model_class, kwargs in models:
    print(f"\nLoading {name}...")
    
    try:
        extractor = model_class(**kwargs)
        
        start = time.time()
        result = extractor.extract(test_log)
        elapsed = time.time() - start
        
        print(f"{name}")
        print(f"   Time: {elapsed:.2f} seconds")
        print(f"   Asset: {result.get('asset', 'N/A')}")
        print(f"   Threat: {result.get('threat', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
        
    except Exception as e:
        print(f"{name} failed: {str(e)[:100]}")

print("Test Complete")
