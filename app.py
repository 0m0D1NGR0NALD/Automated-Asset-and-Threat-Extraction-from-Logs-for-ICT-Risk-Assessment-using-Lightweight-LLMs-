# app.py - Gradio web app for ICT Risk Assessment Tool

import gradio as gr
import pandas as pd
import tempfile
import subprocess
import os
import sys
from pathlib import Path

# Helper function to run risk assessment tool
def run_risk_assessment(logs, model, few_shot, confidence_threshold):
    """
    Run the ICT risk assessment tool on input logs.
    """
    if not logs or not logs.strip():
        return None, "Error: No log input provided.", None
    
    # Create temporary file for input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(logs)
        temp_input = f.name
    
    # Create temporary file for output
    temp_output = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.main",
        "--model", model,
        "-i", temp_input,
        "-o", temp_output
    ]
    if few_shot:
        cmd.append("--few-shot")
    
    try:
        # Run the tool
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(temp_output):
            # Read results
            df = pd.read_csv(temp_output)
            
            # Apply confidence threshold
            if 'requires_review' in df.columns:
                df['requires_review'] = df['confidence'] < confidence_threshold
            
            # Prepare summary
            summary = f"""
            **Results Summary**
            - Total logs processed: {len(df)}
            - High/Critical risk (≥13): {len(df[df['risk_score'] >= 13])}
            - Needs review: {len(df[df['requires_review'] == True])}
            - Average confidence: {df['confidence'].mean():.3f}
            """
            
            # Convert DataFrame to CSV string for download
            csv_output = df.to_csv(index=False)
            
            # Also return DataFrame for display
            display_df = df[['raw_log_preview', 'extracted_asset', 'extracted_threat', 
                            'risk_score', 'confidence', 'requires_review']].head(50)
            
            return display_df, summary, csv_output
        else:
            return None, f"Error running tool: {result.stderr}", None
            
    except subprocess.TimeoutExpired:
        return None, "Processing timeout (5 minutes). Try with fewer logs.", None
    except Exception as e:
        return None, f"Unexpected error: {e}", None
    finally:
        # Cleanup temp files
        try:
            os.unlink(temp_input)
            if os.path.exists(temp_output):
                os.unlink(temp_output)
        except:
            pass


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="ICT Risk Assessment Tool", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🛡️ ICT Risk Assessment Tool
        Automated asset and threat extraction from HTTP logs using lightweight LLMs.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                # Log input
                log_input = gr.Textbox(
                    label="HTTP Logs",
                    placeholder="GET http://localhost:8080/index.jsp HTTP/1.1\nGET http://localhost:8080/product.jsp?id=1' OR '1'='1\nPOST http://localhost:8080/login.jsp user=admin&pwd=12345",
                    lines=15,
                    info="One HTTP request per line"
                )
                
                # File upload
                file_input = gr.File(
                    label="Or Upload a Text File",
                    file_types=[".txt"],
                    type="binary"
                )
                
                # Model selection
                model_select = gr.Radio(
                    choices=["smolLM2", "qwen", "tinyllama"],
                    label="LLM Model",
                    value="smolLM2",
                    info="SmolLM2 (fast, CPU) | Qwen (best accuracy) | TinyLlama (slow)"
                )
                
                # Few-shot toggle
                few_shot_check = gr.Checkbox(
                    label="Enable Few‑shot Prompting",
                    value=True,
                    info="Adds examples to improve accuracy"
                )
                
                # Confidence threshold
                threshold_slider = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.75,
                    step=0.05,
                    info="Lower confidence predictions are flagged for review"
                )
                
                # Run button
                run_btn = gr.Button("Run Risk Assessment Tool", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                
                # Output table
                output_table = gr.Dataframe(
                    label="Risk Register (first 50 rows)",
                    wrap=True,
                    interactive=False
                )
                
                # Summary
                output_summary = gr.Markdown("_Waiting for input..._")
                
                # Download button (appears after processing)
                download_btn = gr.File(label="Download Full Risk Register (CSV)", visible=False)
        
        # Function to handle file upload
        def update_text_from_file(file):
            if file is not None:
                content = file.decode('utf-8')
                return content
            return ""
        
        file_input.change(update_text_from_file, inputs=file_input, outputs=log_input)
        
        # Function to process and update outputs
        def process_and_output(logs, model, few_shot, threshold):
            df, summary, csv_data = run_risk_assessment(logs, model, few_shot, threshold)
            
            if df is not None:
                # Show download button with CSV data
                return df, summary, gr.update(visible=True, value=csv_data)
            else:
                return None, summary, gr.update(visible=False)
        
        # Connect button to processing
        run_btn.click(
            process_and_output,
            inputs=[log_input, model_select, few_shot_check, threshold_slider],
            outputs=[output_table, output_summary, download_btn]
        )
        
        # Expandable sections for help
        with gr.Accordion("ℹAbout Risk Scoring", open=False):
            gr.Markdown("""
            **Risk Formula:** `Risk Score = Impact × Likelihood`
            
            - **Impact** – Asset criticality:
                - `public_web_server`: 5 (highest)
                - `internal_database`: 4
                - `developer_workstation`: 3
                - `test_environment`: 2
            
            - **Likelihood** – Threat severity:
                - `sql_injection`: 4
                - `xss`: 3
                - `brute_force`: 3
                - `info_leak`: 2
                - `benign`: 0
            
            - **Risk Levels:** Low (1–5), Medium (6–12), High (13–20), Critical (21–25)
            """)
        
        with gr.Accordion("Input Format Example", open=False):
            gr.Markdown("""
            Each line should contain one complete HTTP request:

            GET http://localhost:8080/index.jsp HTTP/1.1
            POST http://localhost:8080/login.jsp user=admin&pwd=12345
            GET http://localhost:8080/product.jsp?id=1' OR '1'='1

                        """)

        with gr.Accordion("Model Information", open=False):
            gr.Markdown("""
            - **SmolLM2‑360M** (~2GB RAM)
            - **Qwen2.5‑3B** (~6GB RAM)
            - **TinyLlama‑1.1B** (~4GB RAM)
            """)

        return demo


# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()