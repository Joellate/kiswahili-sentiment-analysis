# Web app

This is the interface for the fine-tuned Kiswahili sentiment model.

**Live app:** https://kiswahili-sentiment.streamlit.app/ Full project: https://github.com/Joellate/kiswahili-sentiment-analysis

| File | Purpose |
|---|---|
| `predict.py` | Loads the model and makes predictions. Shared by both apps. |
| `streamlit_app.py` | **Deployed app** (Streamlit Community Cloud) |
| `app.py` | Gradio version of the same app, for local use |
| `requirements.txt` | Dependencies of the deployed Streamlit app (CPU-only PyTorch) |
| `requirements-gradio.txt` | Dependencies of the Gradio version |

## How the app connects to the model

1. `notebooks/04_transformer_finetune.ipynb` fine-tunes the model and uploads it to the Hugging Face Hub as `RubaxTyra/kiswahili-sentiment`.
2. On start-up, `predict.load_model()` downloads it with `AutoTokenizer` / `AutoModelForSequenceClassification.from_pretrained`. It uses `models/best_transformer/` instead if that folder exists locally, and the `MODEL_ID` environment variable overrides the model id. Streamlit caches the loaded model (`st.cache_resource`), so it is downloaded only once.
3. Each input is cleaned exactly like the training data (`src/preprocessing.py`, plus removal of @mentions, hashtags, emojis and punctuation), tokenised, and passed through the model. The softmax probabilities for positive / neutral / negative are displayed.

## Run locally

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Or the Gradio version:

```bash
pip install -r app/requirements-gradio.txt
python app/app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Set the repository to `Joellate/kiswahili-sentiment-analysis`, the branch to `main`, and the main file path to `app/streamlit_app.py`.
4. Optional: set the app URL to `kiswahili-sentiment`.
5. Click **Deploy**. The first build takes about 5 minutes.

Hugging Face Spaces was the original target. Gradio Spaces now require a paid PRO account (the API returns HTTP 402), so the app is deployed on Streamlit instead. The model itself stays free on the Hugging Face Hub.
