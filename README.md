# Financial News Sentiment Classifier

Hey there! This is a small little Python project that labels financial news sentences as positive, neutral, or negative, comparing four approaches on the same 970-sentence held-out test set. If you have any thoughts, feedback, errors, or improvements to share, feel free to reach out annasong136@gmail.com :) Enjoy!

| Model | Accuracy |
| TextBlob dictionary | 50.9% |
| TF-IDF + logistic regression | 72.2% |
| TF-IDF trigrams + LinearSVC | 75.6% |
| FinBERT, fine-tuned | 87.8% |

## Data
[Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) on Kaggle. Simply download `all-data.csv` and place it next to the notebook!

## How to run
1. `pip install -r requirements.txt`
2. Open `sentiment.ipynb` and run the cells in order.
3. FinBERT fine-tuning needs a GPU, so I ran it in Google Colab: [https://colab.research.google.com/drive/189mqntOX-6gbeaJSx8tDUvJ6uwZt8-w2]
- `sentiment.ipynb`: baseline models (dictionary, TF-IDF)
- `finbert_finetune.ipynb`: FinBERT fine-tuning (needs a GPU, I used Colab)
