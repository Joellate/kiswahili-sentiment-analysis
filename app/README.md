---
title: Kiswahili Sentiment
emoji: 💬
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-4.0
---

# Kiswahili Tweet Sentiment: web app

This is a Gradio interface for the fine-tuned Kiswahili sentiment model. Full project: https://github.com/Joellate/kiswahili-sentiment-analysis

## How it connects to the model

1. `notebooks/04_transformer_finetune.ipynb` fine-tunes the model and uploads it to the Hugging Face Hub as `Joellate/kiswahili-sentiment`.
2. When this app starts, it downloads that model with `AutoModelForSequenceClassification.from_pretrained`. You can override the model with the `MODEL_ID` environment variable. If `models/best_transformer/` exists locally, the app uses that folder instead.
3. Each input is cleaned the same way as the training data (`preprocessing.py`), tokenised, and passed through the model. The softmax probabilities for the three classes are shown.

## Run locally

```bash
pip install -r app/requirements.txt
python app/app.py
```

## Deploy to Hugging Face Spaces

Section 8 of `notebooks/04_transformer_finetune.ipynb` deploys this automatically. It creates the Space `<user>/kiswahili-sentiment-demo` (SDK: Gradio), uploads `app.py`, `requirements.txt`, this `README.md` and `src/preprocessing.py`, and sets `MODEL_ID` to the uploaded model.
