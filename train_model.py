import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ✅ Load dataset
df = pd.read_csv("dataset/imdb_reviews.csv")   # columns: review, sentiment

# ✅ Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["review"] = df["review"].apply(clean_text)

# ✅ Convert sentiment to 0/1
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

X = df["review"]
y = df["sentiment"]

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ TF-IDF
tfidf = TfidfVectorizer(
    max_features=8000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ✅ Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# ✅ Accuracy
pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, pred)
print("✅ Model Accuracy:", acc)

# ✅ Save Model
pickle.dump(model, open("model/sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf_vectorizer.pkl", "wb"))

print("✅ Model Saved Successfully")