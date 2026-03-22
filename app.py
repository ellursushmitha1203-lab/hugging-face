import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")
text_gen_model = pipeline("text-generation", model="gpt2")

def sentiment_analysis(text):
    result = sentiment_model(text)[0]
    label = result['label']
    score = round(result['score'] * 100, 1)
    if label == "POSITIVE":
        return f"✅ POSITIVE — Confidence: {score}%"
    else:
        return f"❌ NEGATIVE — Confidence: {score}%"

def translate_text(text):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_text(prompt, max_tokens):
    result = text_gen_model(prompt, max_new_tokens=int(max_tokens))[0]
    return result['generated_text']

css = """
body {background-color: #1a1a2e;}
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    font-family: 'Segoe UI', sans-serif !important;
}
h1 {
    text-align: center !important;
    color: #FFD21E !important;
    font-size: 2rem !important;
    padding: 10px !important;
}
.subtitle {
    text-align: center;
    color: #aaa;
    margin-bottom: 20px;
}
.gr-button-primary {
    background: #FFD21E !important;
    color: #000 !important;
    font-weight: bold !important;
    border: none !important;
}
.gr-button-primary:hover {
    background: #e6bd00 !important;
}
.tech-note {
    background: #2a2a3e;
    border-left: 4px solid #FFD21E;
    padding: 10px 15px;
    border-radius: 5px;
    color: #aaa;
    font-size: 0.85rem;
    margin-top: 10px;
}
.gr-tab-nav {
    background: #2a2a3e !important;
}
"""

with gr.Blocks(css=css, title="AI Workbench") as demo:
    gr.Markdown("# 🤗 AI Workbench")
    gr.Markdown("<p class='subtitle'>Powered by Hugging Face pre-trained models</p>")

    with gr.Tabs():
        with gr.Tab("✏️ Text Generation"):
            gr.Markdown("### Generate creative text from a prompt")
            with gr.Row():
                with gr.Column():
                    g_input = gr.Textbox(
                        label="Story Prompt",
                        placeholder="Once upon a time...",
                        lines=4
                    )
                    g_slider = gr.Slider(50, 300, value=100, step=50, label="Max Tokens")
                    g_btn = gr.Button("▶ Generate Text", variant="primary")
            g_output = gr.Textbox(label="Generated Output", lines=5)
            g_btn.click(generate_text, inputs=[g_input, g_slider], outputs=g_output)
            gr.HTML("<div class='tech-note'>📌 <b>Technical Note:</b> This app is calling a pre-trained model from the Hugging Face Model Hub, removing the need for local GPUs.</div>")

        with gr.Tab("💬 Sentiment Analysis"):
            gr.Markdown("### Detect the sentiment of any text")
            with gr.Row():
                with gr.Column():
                    s_input = gr.Textbox(
                        label="Enter Text",
                        placeholder="I love this product!",
                        lines=3
                    )
                    s_btn = gr.Button("▶ Analyze Sentiment", variant="primary")
            s_output = gr.Textbox(label="Sentiment Result")
            s_btn.click(sentiment_analysis, inputs=s_input, outputs=s_output)
            gr.HTML("<div class='tech-note'>📌 <b>Technical Note:</b> This app is calling a pre-trained model from the Hugging Face Model Hub, removing the need for local GPUs.</div>")

        with gr.Tab("🌐 Translation"):
            gr.Markdown("### Translate English text to French")
            with gr.Row():
                with gr.Column():
                    t_input = gr.Textbox(
                        label="English Text",
                        placeholder="Hello, how are you?",
                        lines=4
                    )
                with gr.Column():
                    t_output = gr.Textbox(
                        label="French Translation",
                        lines=4
                    )
            t_btn = gr.Button("▶ Translate", variant="primary")
            t_btn.click(translate_text, inputs=t_input, outputs=t_output)
            gr.HTML("<div class='tech-note'>📌 <b>Technical Note:</b> This app is calling a pre-trained model from the Hugging Face Model Hub, removing the need for local GPUs.</div>")

demo.launch()
