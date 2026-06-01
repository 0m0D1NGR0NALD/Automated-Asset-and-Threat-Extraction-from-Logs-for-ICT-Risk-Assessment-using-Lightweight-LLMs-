# app.py - Gradio web app for ICT Risk Assessment Tool
import gradio as gr
import pandas as pd
import tempfile
import yaml
import sys
import os
from pathlib import Path
import gc

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.parser.log_preprocessor import LogPreprocessor
from src.risk.risk_scorer import RiskScorer
from src.risk.confidence_filter import ConfidenceFilter

# Import extractors
from src.extractor.smolLM2_extractor import SmolLM2Extractor
from src.extractor.qwen_extractor import QwenExtractor
from src.extractor.tinyllama_extractor import TinyLlamaExtractor

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Model mapping
EXTRACTORS = {
    "SmolLM2-360M": SmolLM2Extractor,
    "Qwen2.5-3B": QwenExtractor,
    "TinyLlama-1.1B": TinyLlamaExtractor
}

def filter_comments(log_content):
    """Remove lines that start with '#' (comments)."""
    if not log_content or not log_content.strip():
        return None, "No input provided."
    
    clean_lines = []
    for line in log_content.strip().split('\n'):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            clean_lines.append(stripped)
    
    if not clean_lines:
        return None, "Error: No valid log lines found after removing comments."
    
    return '\n'.join(clean_lines), None

def process_logs(logs, model_name, few_shot, confidence_threshold):
    """Process logs directly using the selected extractor."""
    if not logs or not logs.strip():
        return pd.DataFrame(), "No input provided."
    
    # Unpack the tuple
    filtered_logs, error = filter_comments(logs)
    if error:
        return pd.DataFrame(), error
    
    # Split into lines and clean
    lines = [l.strip() for l in filtered_logs.strip().split('\n') if l.strip()]
    
    if not lines:
        return pd.DataFrame(), "No valid log lines."
    
    try:
        extractor_class = EXTRACTORS.get(model_name, SmolLM2Extractor)
        extractor = extractor_class(few_shot=few_shot)
    except Exception as e:
        return pd.DataFrame(), f"Error loading model: {str(e)}"
    
    risk_scorer = RiskScorer(config)
    confidence_filter = ConfidenceFilter(threshold=confidence_threshold)
    
    results = []
    
    for idx, line in enumerate(lines):
        try:
            cleaned, timestamp, ip = LogPreprocessor.clean_line(line)
            if not cleaned:
                continue
            
            extraction = extractor.extract(cleaned)
            asset = extraction.get('asset', 'unknown')
            threat = extraction.get('threat', 'unknown')
            confidence = extraction.get('confidence', 0.5)
            
            likelihood, impact, risk = risk_scorer.compute_risk(asset, threat)
            requires_review = confidence_filter.requires_review(confidence)
            
            results.append({
                'Log': line[:80] + ('...' if len(line) > 80 else ''),
                'Asset': asset,
                'Threat': threat,
                'Risk': risk,
                'Confidence': round(confidence, 3),
                'Review': 'Yes' if requires_review else 'No'
            })
        except Exception as e:
            results.append({
                'Log': line[:80],
                'Asset': 'error',
                'Threat': str(e)[:40],
                'Risk': 0,
                'Confidence': 0.0,
                'Review': 'Yes'
            })

    gc.collect()

    if not results:
        return pd.DataFrame(), "No valid logs processed."
    
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    total = len(df)
    high_risk = len(df[df['Risk'] >= 13])
    needs_review = len(df[df['Review'] == 'Yes'])
    avg_conf = df['Confidence'].mean()
    
    summary = f"""
### Summary
| Metric | Value |
|--------|-------|
| Total logs processed | **{total}** |
| High/Critical risk (≥13) | **{high_risk}** |
| Needs manual review | **{needs_review}** |
| Average confidence | **{avg_conf:.3f}** |
"""
    
    return df, summary

def read_file(file):
    """Read uploaded file and return content."""
    if file is None:
        return ""
    with open(file.name, 'r', encoding='utf-8') as f:
        return f.read()

# Create Gradio interface
with gr.Blocks(title="ICT Risk Assessment Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <div style="text-align: center;">
        <h1>ICT Risk Assessment Tool</h1>
        <p style="font-size: 1.1em;">Automated asset and threat extraction from HTTP logs using lightweight LLMs</p>
        <hr>
    </div>
    """)
    
    with gr.Row():
        # Left column - Input
        with gr.Column(scale=5):
            gr.Markdown("### Input")
            
            with gr.Tabs():
                with gr.TabItem("Paste Logs"):
                    log_input = gr.Textbox(
                        label="",
                        placeholder="GET http://localhost:8080/index.jsp HTTP/1.1 \nPOST /login.jsp user=admin&pwd=12345",
                        lines=12
                    )
                
                with gr.TabItem("Upload File"):
                    file_input = gr.File(
                        label="Upload .txt file",
                        file_types=[".txt"],
                        file_count="single"
                    )
                    file_content = gr.Textbox(label="File Preview", lines=8, interactive=False, visible=False)
                    
                    def update_from_file(file):
                        if file:
                            with open(file.name, 'r', encoding='utf-8') as f:
                                content = f.read()
                            return content, gr.update(visible=True, value=content[:500] + ("..." if len(content) > 500 else ""))
                        return "", gr.update(visible=False)
                    
                    file_input.change(update_from_file, inputs=file_input, outputs=[log_input, file_content])
            
            # Settings
            with gr.Row():
                with gr.Column():
                    model_select = gr.Dropdown(
                        choices=["SmolLM2-360M", "Qwen2.5-3B", "TinyLlama-1.1B"],
                        label="Model",
                        value="SmolLM2-360M",
                        info="SmolLM2 recommended for CPU"
                    )
                
                with gr.Column():
                    few_shot = gr.Checkbox(
                        label="Few-shot Prompting",
                        value=True,
                        info="Adds examples to improve accuracy"
                    )
                
                with gr.Column():
                    threshold = gr.Slider(
                        label="Confidence Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.75,
                        step=0.05
                    )
            
            run_btn = gr.Button("Run Assessment", variant="primary", size="lg")
        
        # Right column - Results
        with gr.Column(scale=7):
            gr.Markdown("### Results")
            output_summary = gr.Markdown("_Ready to process logs..._")
            output_table = gr.Dataframe(
                label="Risk Register",
                wrap=True,
                interactive=False
            )
            
            with gr.Row():
                download_btn = gr.File(label="Download CSV", visible=False)
                clear_btn = gr.Button("Clear Results", size="sm", variant="secondary")
    
    def run_and_output(logs, model, fs, thresh):
        df, summary = process_logs(logs, model, fs, thresh)
        if df is not None and not df.empty:
            # Save to temporary CSV file
            csv_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8').name
            df.to_csv(csv_path, index=False)
            return df, summary, gr.update(visible=True, value=csv_path)
        return pd.DataFrame(), summary, gr.update(visible=False)
    
    def clear_results():
        return pd.DataFrame(), "_Ready to process logs..._", gr.update(visible=False)
    
    run_btn.click(
        run_and_output,
        inputs=[log_input, model_select, few_shot, threshold],
        outputs=[output_table, output_summary, download_btn]
    )
    
    clear_btn.click(
        clear_results,
        outputs=[output_table, output_summary, download_btn]
    )
    
    # Help section (collapsible)
    with gr.Accordion("ℹDocumentation", open=False):
        gr.Markdown("""
        ### Input Format
        Each line should contain one complete HTTP request:
                    
            GET http://localhost:8080/index.jsp HTTP/1.1
            POST http://localhost:8080/login.jsp user=admin&pwd=12345
                    
        ### Risk Scoring
        **Risk Score = Impact × Likelihood** (1–25)

        | Level | Score |
        |-------|-------|
        | Low | 1–5 |
        | Medium | 6–12 |
        | High | 13–20 |
        | Critical | 21–25 |

        ### Models
        | Model | Speed | Hardware |
        |-------|-------|----------|
        | SmolLM2-360M | Fast | CPU (~2GB) |
        | Qwen2.5-3B | Slow | GPU (~6GB) |
        | TinyLlama-1.1B | Very Slow | CPU (~4GB) |

        ### Column Explanation
        - **Asset**: System component (web server, database, workstation, test env)
        - **Threat**: Attack type (SQL injection, XSS, brute force, etc.)
        - **Risk**: Numerical risk score (higher = more urgent)
        - **Confidence**: Model certainty (0-1)
        - **Review**: Flagged for manual review if confidence < threshold
        """)

        # Footer
        with gr.Column():
            gr.Markdown("""
            <hr>
            <div style="text-align: center; color: gray; font-size: 0.8em;">
            ICT Risk Assessment Tool | Built with Gradio | Lightweight LLMs
            </div>
            """)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)