import gradio as gr
import time
import random

# Dummy analysis function (replace with your real Hybrid CNN-Transformer model later)
def analyze_music(audio_file):
    if audio_file is None:
        return None, "❌ Please upload a music file first.", "0%"
    
    time.sleep(1.8)  # Simulate processing time
    
    overall_ai = random.randint(72, 97)
    human_like = round(100 - overall_ai - random.uniform(1, 7), 1)
    confidence = round(random.uniform(87, 99.5), 1)
    
    details = f"""
### 🎯 Analysis Results

**Overall AI Detection Level**  
<span style="font-size: 3.2rem; font-weight: bold; color: #60a5fa;">{overall_ai}%</span>

**Detailed Breakdown**
- **AI-Generated Probability**: **{overall_ai}%**
- **Human / Original Likelihood**: **{human_like}%**
- **Model Confidence**: **{confidence}%**

**Interpretation**: This track shows **strong signs** of being AI-generated.
    """
    
    return overall_ai, details, f"{overall_ai}%"

def reset_ui():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        None,
        None,
        "0%"
    )

# Custom CSS - Modern Dark Theme (closer to typical Figma designs)
custom_css = """
.gradio-container {
    background-color: #0a0f1c;
    color: #e2e8f0;
}

h1 {
    color: #60a5fa !important;
    font-weight: 700;
    letter-spacing: -0.025em;
}

.card {
    background-color: #1e2937;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid #334155;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.audio-upload {
    background-color: #1e2937 !important;
    border: 2px dashed #475569 !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

button.primary {
    background: linear-gradient(135deg, #3b82f6, #1e40af) !important;
    border: none !important;
    font-weight: 600;
    border-radius: 12px !important;
    padding: 16px 32px !important;
    transition: all 0.3s ease;
}

button.primary:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 25px -5px rgba(59, 130, 246, 0.5);
}

.gr-slider input[type=range] {
    accent-color: #60a5fa;
}

.label, .gr-markdown h3 {
    color: #94a3b8;
    font-weight: 500;
}
"""

with gr.Blocks(
    title="Music AI Detection System",
    css=custom_css
) as demo:
    
    gr.Markdown("# 🎵 Music AI Detection System\n**Hybrid CNN-Transformer** — Thesis Prototype")

    with gr.Row(equal_height=True):
        # === UPLOAD SCREEN ===
        with gr.Column(visible=True, elem_classes="card") as upload_col:
            gr.Markdown("### Step 1: Upload Your Music File")
            gr.Markdown("Supported: **MP3 • WAV • OGG • FLAC**")
            
            audio_input = gr.Audio(
                label="Browse & Upload Music File",
                type="filepath",
                sources=["upload"],
                format="mp3",
                elem_classes="audio-upload"
            )
            
            analyze_btn = gr.Button("🔍 Detect File", variant="primary", size="large")

        # === RESULTS SCREEN ===
        with gr.Column(visible=False, elem_classes="card") as results_col:
            with gr.Row():
                # Left: Big Overall Score
                with gr.Column(scale=1):
                    gr.Markdown("### AI Detection Level")
                    overall_label = gr.HTML(
                        value="<div style='text-align:center; font-size:4.5rem; font-weight:bold; color:#60a5fa; margin:20px 0;'>0%</div>",
                        show_label=False
                    )
                    overall_slider = gr.Slider(
                        minimum=0, maximum=100, value=0,
                        label="Detection Strength",
                        interactive=False
                    )

                # Right: Details
                with gr.Column(scale=2):
                    gr.Markdown("### Detailed Analysis")
                    details_md = gr.Markdown()

            analyze_another_btn = gr.Button("📁 Analyze Another File", variant="secondary", size="large")

    # Events
    analyze_btn.click(
        fn=analyze_music,
        inputs=audio_input,
        outputs=[overall_slider, details_md, overall_label]
    ).then(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[upload_col, results_col]
    )

    analyze_another_btn.click(
        fn=reset_ui,
        inputs=None,
        outputs=[upload_col, results_col, audio_input, details_md, overall_label]
    )

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=True,
        debug=True
    )