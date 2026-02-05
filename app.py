from flask import Flask, render_template, request, jsonify
import os
import pickle
import re
import requests
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ✅ TMDB API KEY (YOURS)
TMDB_API_KEY = "a749807cb78261ccdbe342dc1d5d08ee"

# ✅ Load model + vectorizer
model = pickle.load(open("model/sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords_from_review(review: str, top_n: int = 6):
    words = review.lower().split()
    stop = set(ENGLISH_STOP_WORDS).union({"movie", "film"})
    filtered = [w for w in words if w not in stop and len(w) > 2]
    freq = Counter(filtered)
    return [w for w, c in freq.most_common(top_n)]


# ✅ Extract IMDb ID from any text/url (tt1234567)
def extract_imdb_id(text: str):
    match = re.search(r"(tt\d{5,10})", text)
    return match.group(1) if match else None


# ✅ Get TMDB movie ID using IMDb ID
def get_tmdb_movie_id_from_imdb(imdb_id: str):
    url = f"https://api.themoviedb.org/3/find/{imdb_id}"
    params = {"api_key": TMDB_API_KEY, "external_source": "imdb_id"}
    r = requests.get(url, params=params, timeout=15)

    if r.status_code != 200:
        return None

    data = r.json()
    if data.get("movie_results") and len(data["movie_results"]) > 0:
        return data["movie_results"][0]["id"]
    return None

def train_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    df["review"] = df["review"].apply(clean_text)
    df["sentiment"] = df["sentiment"].map({"positive":1, "negative":0})

    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    model_new = LogisticRegression()
    model_new.fit(X_train_vec, y_train)

    # save updated model
    pickle.dump(model_new, open("model/sentiment_model.pkl","wb"))
    pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl","wb"))

    return "Model updated successfully!"

# ✅ Search TMDB movie by name
def search_tmdb_movie_id(query: str):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query}

    try:
        r = requests.get(url, params=params, timeout=20)
    except requests.exceptions.RequestException:
        print("TMDB search timeout/error")
        return None

    if r.status_code != 200:
        return None

    data = r.json()
    results = data.get("results", [])

    if len(results) == 0:
        return None

    return results[0]["id"]
 # top match


# ✅ Extract page title from URL (works for Wikipedia & many sites)
def try_extract_title_from_url(url: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
    except:
        return None

    m = re.search(r"<title>(.*?)</title>", r.text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    title = m.group(1)
    title = re.sub(r"\s+", " ", title).strip()

    # remove suffix
    title = title.replace(" - Wikipedia", "").replace(" | Wikipedia", "")
    title = title.split("|")[0].strip()

    return title


# ✅ Fetch reviews from TMDB
def get_tmdb_reviews(movie_id: int, limit: int = 25):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": 1}
    r = requests.get(url, params=params, timeout=15)

    if r.status_code != 200:
        return []

    data = r.json()
    reviews = []

    for item in data.get("results", []):
        content = item.get("content", "")
        if content.strip():
            reviews.append(content.strip())

    return reviews[:limit]


@app.route("/")
def home():
    return render_template("index.html")
@app.route("/history")
def history():
    return render_template("history.html")
import os

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"})

    file = request.files["file"]

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file only!"})

    save_path = "dataset/custom_reviews.csv"

    # ✅ If old CSV exists → delete it (replace logic)
    if os.path.exists(save_path):
        os.remove(save_path)

    # ✅ Save new CSV
    file.save(save_path)

    try:
        msg = train_from_csv(save_path)
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"})

    return jsonify({
        "message": "CSV updated successfully! Old dataset replaced and model retrained."
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    user_input = data.get("url", "").strip()

    if not user_input:
        return jsonify({"error": "Please paste movie URL OR type movie name!"})

    tmdb_id = None
    movie_used = None

    # ✅ 1) Try IMDb id
    imdb_id = extract_imdb_id(user_input)
    if imdb_id:
        tmdb_id = get_tmdb_movie_id_from_imdb(imdb_id)
        movie_used = f"IMDb ID: {imdb_id}"

    # ✅ 2) If input looks like URL, try page title
    if not tmdb_id and user_input.startswith("http"):
        title = try_extract_title_from_url(user_input)
        if title:
            tmdb_id = search_tmdb_movie_id(title)
            movie_used = title

    # ✅ 3) Else treat as movie name
    if not tmdb_id:
        tmdb_id = search_tmdb_movie_id(user_input)
        movie_used = user_input

    if not tmdb_id:
        return jsonify({"error": "Movie not found on TMDB. Try another URL or movie name."})

    reviews = get_tmdb_reviews(tmdb_id)

    if len(reviews) == 0:
        return jsonify({"error": "No reviews found on TMDB for this movie. Try another movie."})

   
    pos, neg = 0, 0
    all_keywords = []

    for rev in reviews:
        clean_rev = clean_text(rev)

        vec = tfidf.transform([clean_rev])
        prob = model.predict_proba(vec)[0]  # [neg, pos]

        if prob[1] > prob[0]:
            pos += 1
        else:
            neg += 1

        all_keywords.extend(extract_keywords_from_review(clean_rev, top_n=4))

    total = pos + neg
    positive_percent = round((pos / total) * 100, 2)
    negative_percent = round((neg / total) * 100, 2)

    overall = "Positive ✅" if pos >= neg else "Negative ❌"
    keywords = [w for w, c in Counter(all_keywords).most_common(12)]

    return jsonify({
        "movie_used": movie_used,
        "tmdb_id": tmdb_id,
        "overall_sentiment": overall,
        "positive_percent": positive_percent,
        "negative_percent": negative_percent,
        "total_reviews": total,
        "keywords": keywords
    })


if __name__ == "__main__":
    app.run(debug=True)