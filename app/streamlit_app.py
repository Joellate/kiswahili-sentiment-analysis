"""Streamlit web app: Kiswahili tweet sentiment classifier (deployed on Streamlit Community Cloud).

Run locally:  streamlit run app/streamlit_app.py
"""
import streamlit as st

from predict import EXAMPLES, MODEL_ID, load_model, predict

st.set_page_config(page_title="Kiswahili Sentiment", page_icon="💬")

ICONS = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}


@st.cache_resource(show_spinner="Loading the model (first visit takes ~1 minute)...")
def get_model():
    return load_model()


tokenizer, model, labels = get_model()

st.title("💬 Kiswahili Tweet Sentiment")
st.write(
    "Type a tweet or short message in **Kiswahili**. The model predicts whether it is "
    "**positive**, **neutral** or **negative**."
)

choice = st.selectbox("Try an example, or type your own text below", ["(type your own)"] + EXAMPLES)
text = st.text_area(
    "Kiswahili text",
    value="" if choice == "(type your own)" else choice,
    height=120,
    placeholder="Andika ujumbe hapa...",
)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.info("Please type some text first.")
    else:
        scores, cleaned = predict(text, tokenizer, model, labels)
        top = max(scores, key=scores.get)
        st.subheader(f"{ICONS.get(top, '')} {top.capitalize()} ({scores[top]:.0%} confidence)")
        for label in labels:
            st.progress(scores[label], text=f"{ICONS.get(label, '')} {label}: {scores[label]:.1%}")
        if scores[top] < 0.5:
            st.warning("Low confidence: the model is unsure about this text.")
        st.caption(f"Model input after cleaning: `{cleaned}`")

with st.expander("How does it work?"):
    st.markdown(
        f"""
1. Your text is cleaned the same way as the training data: @mentions, hashtags, emojis and punctuation are removed.
2. The cleaned text is split into subword tokens and passed to **AfriBERTa-large**, a Transformer pretrained on
   11 African languages, including Kiswahili. It was fine-tuned on 1,810 labelled Kiswahili tweets from the
   **AfriSenti** dataset (Muhammad et al., 2023).
3. A classification head turns the Transformer's output into three probabilities (softmax), shown above.

Model: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) · test macro-F1 0.568 ·
Code, experiments and report: [GitHub](https://github.com/Joellate/kiswahili-sentiment-analysis)
"""
    )

st.caption(
    "**Limitations:** trained on only ~1.8k tweets. The model often labels clear complaints or praise as neutral, "
    "struggles with sarcasm, slang/Sheng and code-switching, and ignores emojis."
)
