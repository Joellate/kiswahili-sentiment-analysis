# Kiswahili Tweet Sentiment Classification

Capstone project: Text classification for an African language.

Given a tweet written in **Kiswahili**, the system predicts whether its sentiment is **positive**, **neutral** or **negative**.

| | Link |
|---|---|
| GitHub repository | https://github.com/Joellate/kiswahili-sentiment-analysis |
| Live demo | _TODO (Hugging Face Space)_ |
| Demo video | _TODO_ |
| Report (PDF) | _TODO_ |

## Problem

Kiswahili is one of the most widely spoken languages in Africa and a lingua franca of East Africa, yet sentiment tools are mostly built for English. Organisations such as telecoms, banks, NGOs and public services receive Kiswahili feedback on social media. They need to tell complaints apart from praise and neutral information, at a scale too large to read by hand.

## Dataset

**AfriSenti, Kiswahili subset** (Muhammad et al., 2023; SemEval-2023 Task 12). The tweets were annotated by native speakers. We use the official splits.

- Hugging Face: https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment (config `swa`)
- Official release: https://github.com/afrisenti-semeval/afrisent-semeval-2023
- Paper: https://aclanthology.org/2023.emnlp-main.862/
- Licence: CC BY 4.0

| Split | Positive | Neutral | Negative | Total |
|---|---|---|---|---|
| train | 547 | 1,072 | 191 | 1,810 |
| validation | 137 | 268 | 48 | 453 |
| test | 224 | 444 | 80 | 748 |

Key properties, all explored in `notebooks/01_data_exploration.ipynb`:
- **Imbalanced labels:** 59% neutral and 11% negative, so **macro-F1** is the primary metric.
- **Already partly cleaned:** emojis, punctuation, @mentions and hashtags were removed in the release.
- **Annotation noise:** some identical tweets carry conflicting labels.
- **High out-of-vocabulary rate:** 51% of the word types in test are unseen in train, because Kiswahili is agglutinative and tweets use non-standard spelling.

## Method

| Stage | Notebook | Models |
|---|---|---|
| Data exploration | `01_data_exploration` | |
| Baselines | `02_baselines_tfidf` | Majority class; word / char TF-IDF with Logistic Regression, Linear SVM and Naive Bayes |
| Recurrent model | `03_bilstm` | BiLSTM with random vs. fastText embeddings, frozen vs. fine-tuned, max vs. attention pooling |
| Transformers | `04_transformer_finetune` | Fine-tuned mBERT, XLM-R, AfriBERTa and AfroXLMR, plus ablations |

All models are selected on validation macro-F1. The neural models are run with 3 random seeds and reported as mean ± std.

## Repository layout

```
src/
  data.py            # download + cache AfriSenti-swa, label maps
  preprocessing.py   # tweet normalisation (each step switchable for ablations)
  evaluate.py        # shared metrics, seed aggregation, confusion matrices, experiment log
notebooks/
  01_data_exploration.ipynb
  02_baselines_tfidf.ipynb
  03_bilstm.ipynb
  04_transformer_finetune.ipynb   # needs a GPU (Colab T4)
  05_error_analysis.ipynb         # bootstrap significance, confusions, OOV analysis, manual error sample
app/
  app.py             # Gradio web app (loads the fine-tuned model from the Hugging Face Hub)
results/
  experiments.csv    # every experiment's metrics (validation + test)
  predictions_*.csv  # per-tweet test predictions used in error analysis
  error_sample.csv   # 50 errors for manual categorisation
  figures/
```

## Results so far (test set)

Neural models are reported as mean ± std over 3 seeds.

| Experiment | Macro-F1 (val) | Macro-F1 (test) | Accuracy (test) | F1 negative (test) |
|---|---|---|---|---|
| B0 Majority class | 0.248 | 0.248 | 0.594 | 0.000 |
| B1 Word TF-IDF + LogReg (balanced) | 0.438 | 0.474 | 0.588 | 0.296 |
| B1b same, no class weighting | 0.355 | 0.393 | 0.590 | 0.092 |
| B2 Char TF-IDF + LogReg (balanced) | 0.491 | 0.456 | 0.535 | 0.304 |
| B3 Word + char TF-IDF + LogReg | 0.449 | 0.478 | 0.575 | 0.310 |
| L1 BiLSTM, random embeddings | 0.412 | 0.433 ± 0.021 | 0.522 | 0.242 |
| L2 BiLSTM, fastText frozen | 0.417 | 0.444 ± 0.016 | 0.524 | 0.300 |
| L3 BiLSTM, fastText fine-tuned | 0.444 | 0.446 ± 0.008 | 0.545 | 0.277 |
| L4 BiLSTM, fastText + attention | 0.435 | 0.455 ± 0.010 | 0.537 | 0.308 |
| L5 L3 without class weighting | 0.436 | 0.426 ± 0.018 | 0.537 | 0.220 |
| T1–T7 Transformers | _run notebook 04 on Colab_ | | | |

Findings so far:
- **Class weighting** is essential for the rare negative class.
- **The BiLSTM does not beat TF-IDF** on 1.8k tweets: it overfits within about 4 epochs. A paired bootstrap shows the two are statistically tied.
- **fastText embeddings** give a small gain and much more stable training.

The full table is in `results/experiments.csv`.

## How to run

**Google Colab (recommended).** Open a notebook, then run all cells. The first cell clones this repo and installs the requirements.

Notebook 04 needs two extra steps:
1. Switch to a GPU runtime first: *Runtime → Change runtime type → T4 GPU*.
2. When it asks, paste a Hugging Face **write** token (create one at https://huggingface.co/settings/tokens).

At the end it uploads the model, the results and the web app to Hugging Face automatically.

| Notebook | |
|---|---|
| 01 Data exploration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joellate/kiswahili-sentiment-analysis/blob/main/notebooks/01_data_exploration.ipynb) |
| 02 Baselines | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joellate/kiswahili-sentiment-analysis/blob/main/notebooks/02_baselines_tfidf.ipynb) |
| 03 BiLSTM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joellate/kiswahili-sentiment-analysis/blob/main/notebooks/03_bilstm.ipynb) |
| 04 Transformers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joellate/kiswahili-sentiment-analysis/blob/main/notebooks/04_transformer_finetune.ipynb) |

**Locally**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

The data downloads automatically from Hugging Face on the first run, and notebook 03 downloads the fastText vectors (~230 MB).

## References

- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning. *COLING 2022*.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*.
- Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.
- Grave, E., Bojanowski, P., Gupta, P., Joulin, A., & Mikolov, T. (2018). Learning Word Vectors for 157 Languages. *LREC 2018*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9*(8).
- Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. *NAACL 2019*.
- Muhammad, S. H., et al. (2023). AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages. *EMNLP 2023*.
- Muhammad, S. H., et al. (2023). SemEval-2023 Task 12: Sentiment Analysis for African Languages (AfriSenti-SemEval). *SemEval 2023*.
- Ogueji, K., Zhu, Y., & Lin, J. (2021). Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for Low-resourced Languages. *MRL Workshop 2021*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR 12*.
- Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models. *ACL 2021*.
- Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *EMNLP 2020 (Demos)*.
