"""Gradio web app: Kiswahili tweet sentiment classifier.

Loads the fine-tuned Transformer produced by notebooks/04_transformer_finetune.ipynb
from the Hugging Face Hub (or a local folder) and classifies text as
positive / neutral / negative.

Run locally:   python app/app.py
Deployed on:   Hugging Face Spaces (this folder is the Space's root)
"""
import os
import re
import sys
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Same cleaning as training. The repo's src/ is used when available; the
# Space ships a copy of preprocessing.py next to this file.
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
from preprocessing import clean_text  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "Joellate/kiswahili-sentiment")
LOCAL_MODEL = HERE.parent / "models" / "best_transformer"
model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else MODEL_ID

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
LABELS = [model.config.id2label[i] for i in range(model.config.num_labels)]


def preprocess(text: str) -> str:
    # Training data had emojis, punctuation and @mentions removed, so we do the same here
    text = re.sub(r"@\w+|#", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return clean_text(text, lowercase=False, replace_numbers=False)


@torch.no_grad()
def classify(text: str):
    if not text or not text.strip():
        return {}, ""
    cleaned = preprocess(text)
    inputs = tokenizer(cleaned, truncation=True, max_length=128, return_tensors="pt")
    probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    scores = {LABELS[i]: float(p) for i, p in enumerate(probs)}
    top = max(scores, key=scores.get)
    note = f"**Model input after cleaning:** `{cleaned}`"
    if scores[top] < 0.5:
        note += "\n\n⚠️ Low confidence: the model is unsure about this text."
    return scores, note


EXAMPLES = [
    # tweets from the AfriSenti Kiswahili test split (the third one is shortened)
    "Watu 12 wamefariki na wengine kuokolewa wakiwa salama baada ya Toyota Hiace kutumbukia Ziwa",
    "Watu wengine hawatakupenda hata ufanye nini na watu wengine hawataacha kukupenda hata ufanye nini Nenda mahali penye upendo",
    "Tuna Ofisi za Usajili kwenye kila Wilaya Mkoani Dar es Salaam fika kwenye Ofisi yetu ya usajili",
    # written for the demo
    "Huduma yenu ni mbaya sana, nimesubiri wiki mbili bila majibu",
    "Asante sana kwa msaada wenu, mmenisaidia haraka",
    "Kesho kutakuwa na mkutano saa nne asubuhi",
]

with gr.Blocks(title="Kiswahili Sentiment") as demo:
    gr.Markdown(
        "# Kiswahili Tweet Sentiment\n"
        "Type a tweet or short message in **Kiswahili**. The model predicts whether it is "
        "**positive**, **neutral** or **negative**.\n\n"
        "The model is a Transformer fine-tuned on the AfriSenti Kiswahili Twitter dataset "
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
        "**Limitations:** trained on only ~1.8k tweets. It struggles with sarcasm, with negative *news* "
        "versus negative *opinion*, and with heavy slang or code-switching. Emojis are ignored because "
        "the training data had them removed."
    )
    btn.click(classify, inputs=text, outputs=[label, info])
    text.submit(classify, inputs=text, outputs=[label, info])

if __name__ == "__main__":
    demo.launch()
