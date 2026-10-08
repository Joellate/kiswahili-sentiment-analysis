"""Gradio version of the web app (local use, or a Hugging Face Space on a PRO account).

The deployed app is streamlit_app.py; both use the same model code in predict.py.

Run locally:   pip install -r app/requirements-gradio.txt && python app/app.py
"""
import gradio as gr

from predict import EXAMPLES, load_model, predict

tokenizer, model, labels = load_model()


def classify(text: str):
    if not text or not text.strip():
        return {}, ""
    scores, cleaned = predict(text, tokenizer, model, labels)
    note = f"**Model input after cleaning:** `{cleaned}`"
    if max(scores.values()) < 0.5:
        note += "\n\n⚠️ Low confidence: the model is unsure about this text."
    return scores, note


with gr.Blocks(title="Kiswahili Sentiment") as demo:
    gr.Markdown(
        "# Kiswahili Tweet Sentiment\n"
        "Type a tweet or short message in **Kiswahili**. The model predicts whether it is "
        "**positive**, **neutral** or **negative**.\n\n"
        "The model is AfriBERTa-large fine-tuned on the AfriSenti Kiswahili Twitter dataset "
        "(Muhammad et al., 2023). "
        "[Code and report](https://github.com/Joellate/kiswahili-sentiment-analysis)"
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Kiswahili text", lines=4, placeholder="Andika ujumbe hapa...")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            label = gr.Label(label="Predicted sentiment", num_top_classes=3)
            info = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=text)
    gr.Markdown(
        "**Limitations:** trained on only ~1.8k tweets. It often labels clear complaints or praise as neutral, "
        "struggles with sarcasm, slang/Sheng and code-switching, and ignores emojis."
    )
    btn.click(classify, inputs=text, outputs=[label, info])
    text.submit(classify, inputs=text, outputs=[label, info])

if __name__ == "__main__":
    demo.launch()
