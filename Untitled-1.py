# %%
import pandas as pd

df = pd.read_csv("all-data.csv", encoding="latin-1", names=["sentiment", "text"])
print(df.shape)
print(df["sentiment"].value_counts(normalize=True))
print(df.sample(5, random_state=1))

# %%
from textblob import TextBlob

def lexicon_label(text, thresh=0.05):
    p = TextBlob(text).sentiment.polarity
    return "positive" if p > thresh else "negative" if p < -thresh else "neutral"

df["lex_pred"] = df["text"].apply(lexicon_label)
print((df["lex_pred"] == df["sentiment"]).mean())

# %%
wrong = df[df["lex_pred"] != df["sentiment"]]
print(wrong[["text", "sentiment", "lex_pred"]].sample(8, random_state=1).to_string())

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"],
    test_size=0.2, random_state=42, stratify=df["sentiment"]
)
print(len(X_train), len(X_test))


# %%
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                              ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print((pred == y_test).mean())

# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, pred))

# %%
import numpy as np

vec, clf = model["tfidf"], model["clf"]
names = np.array(vec.get_feature_names_out())
for i, cls in enumerate(clf.classes_):
    print(cls, names[np.argsort(clf.coef_[i])[-10:]])

# %%
import pandas as pd

errors = pd.DataFrame({"text": X_test, "true": y_test, "pred": pred})
errors = errors[errors["true"] != errors["pred"]]
print(errors.sample(10, random_state=1).to_string())

# %%
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

model2 = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
    ("clf", LinearSVC(class_weight="balanced")),
])
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)
print((pred2 == y_test).mean())
print(classification_report(y_test, pred2))

# %%
from transformers import pipeline

finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)

sample = X_test.iloc[:5].tolist()
for s, out in zip(sample, finbert(sample)):
    best = max(out, key=lambda d: d["score"])
    print(best["label"], round(best["score"], 2), "|", s[:80])

# %%
import transformers, torch
print(transformers.__version__, torch.__version__)

# %%
from transformers import pipeline

finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)

sample = X_test.iloc[:5].tolist()
for s, out in zip(sample, finbert(sample)):
    best = max(out, key=lambda d: d["score"])
    print(best["label"], round(best["score"], 2), "|", s[:80])

print(y_test.iloc[:5].tolist())

# %%
from sklearn.metrics import classification_report

pred_fb = [max(o, key=lambda d: d["score"])["label"].lower() for o in outs]

print((pd.Series(pred_fb, index=y_test.index) == y_test).mean())
print(classification_report(y_test, pred_fb))

# %%
import torch
print(torch.cuda.is_available())



