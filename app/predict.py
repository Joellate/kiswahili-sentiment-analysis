"""Model loading and prediction, shared by the Streamlit app (deployed) and the Gradio app (local).

The fine-tuned model produced by notebooks/04_transformer_finetune.ipynb is
downloaded from the Hugging Face Hub (or loaded from models/best_transformer
if that folder exists locally).
"""
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Same cleaning as training: reuse src/preprocessing.py from the repo
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
from preprocessing import clean_text  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "RubaxTyra/kiswahili-sentiment")
LOCAL_MODEL = HERE.parent / "models" / "best_transformer"

EXAMPLES = [
    # tweets from the AfriSenti Kiswahili test split (the third one is shortened)
    "Watu 12 wamefariki na wengine kuokolewa wakiwa salama baada ya Toyota Hiace kutumbukia Ziwa",
    "Watu wengine hawatakupenda hata ufanye nini na watu wengine hawataacha kukupenda hata ufanye nini Nenda mahali penye upendo",
    "Tuna Ofisi za Usajili kwenye kila Wilaya Mkoani Dar es Salaam fika kwenye Ofisi yetu ya usajili",
    # written for the demo
    "Asante sana kwa msaada wenu, mmenisaidia haraka",
    "Hongera sana timu yetu kwa ushindi mzuri leo",
    "Huduma yenu ni mbaya sana, nimesubiri wiki mbili bila majibu",
    "Kesho kutakuwa na mkutano saa nne asubuhi",
]


def load_model():
    """Return (tokenizer, model, labels). Labels come from the model's config: positive / neutral / negative."""
    path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return tokenizer, model, labels


def preprocess(text: str) -> str:
    # The training data had emojis, punctuation, @mentions and hashtags removed, so we do the same here
    text = re.sub(r"@\w+|#", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return clean_text(text, lowercase=False, replace_numbers=False)


@torch.no_grad()
def predict(text: str, tokenizer, model, labels):
    """Return ({label: probability}, cleaned_text) for one input text."""
    cleaned = preprocess(text)
    inputs = tokenizer(cleaned, truncation=True, max_length=128, return_tensors="pt")
    probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    return {labels[i]: float(p) for i, p in enumerate(probs)}, cleaned
