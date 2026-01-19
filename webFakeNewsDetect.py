import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from transformers import pipeline
import requests
import os
import streamlit as st
import spacy

# -----------------------------
# Cache the Hugging Face model
# -----------------------------
@st.cache_resource
def load_nli_model():
    return pipeline("text-classification", model="roberta-large-mnli", return_all_scores=True)

nli_model = load_nli_model()
_ = nli_model("Warmup headline", text_pair="Warmup trusted article")  # warm-up

# -----------------------------
# -----------------------------
# Load spaCy for entity extraction (with fallback)
# -----------------------------
import subprocess, sys

def get_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = get_spacy_model()

# -----------------------------
# Cache trusted articles
# -----------------------------
@st.cache_data
def fetch_trusted_news(use_cache=True):
    cache_file = "trusted_news.csv"
    if use_cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        return df['article'].tolist()

    API_KEY = "ea359e7ffa4249798fcf2dd2a5f4ec5e"
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    trusted_articles = []
    for article in data.get('articles', []):
        text = (article.get('title') or "") + " " + (article.get('description') or "")
        trusted_articles.append(text)

    pd.DataFrame(trusted_articles, columns=['article']).to_csv(cache_file, index=False)
    return trusted_articles

# -----------------------------
# Exact similarity checking
# -----------------------------
def check_exact_match(input_news, trusted_articles):
    for trusted in trusted_articles:
        if input_news.strip().lower() == trusted.strip().lower():
            return True
    return False

# -----------------------------
# Entity & event extraction
# -----------------------------
def extract_entities_events(text: str) -> dict:
    doc = nlp(text)
    entities = {"PERSON": set(), "ORG": set(), "GPE": set()}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].add(ent.text)

    lower = text.lower()
    events = {
        "death": any(k in lower for k in ["died", "death", "passed away", "last rite", "funeral"]),
        "appearance": any(k in lower for k in ["appearance", "spotted", "photo", "picture", "video", "interview", "birthday"]),
        "attack": any(k in lower for k in ["attack", "missile", "strike", "war", "bomb", "drone"]),
        "absurd_duration": bool(re.search(r"\b\d{4,}\s+years?\b", lower)),
    }
    return {"entities": {k: list(v) for k, v in entities.items()}, "events": events}

# -----------------------------
# Build entity state from trusted sources
# -----------------------------
def build_entity_state(trusted_articles: list) -> dict:
    state = {}
    for art in trusted_articles:
        info = extract_entities_events(art)
        persons = info["entities"].get("PERSON", [])
        has_appearance = info["events"]["appearance"]

        for p in persons:
            if p not in state:
                state[p] = {"alive_recent": False, "evidence": []}
            if has_appearance:
                state[p]["alive_recent"] = True
            state[p]["evidence"].append(art)
    return state

# -----------------------------
# Logical contradiction checks
# -----------------------------
def detect_logical_contradiction(input_text: str, entity_state: dict) -> tuple:
    info = extract_entities_events(input_text)
    persons = info["entities"].get("PERSON", [])
    events = info["events"]

    if events["absurd_duration"]:
        return True, "🔴 Fake: Impossible duration detected"

    if events["death"] and persons:
        for p in persons:
            st = entity_state.get(p)
            if st and st.get("alive_recent"):
                return True, f"🔴 Fake: Logical contradiction — {p} reported alive recently vs death claim"

    return False, ""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words="english")

def semantic_similarity(a, b):
    vecs = tfidf.fit_transform([a, b])
    return cosine_similarity(vecs[0], vecs[1])[0][0]



# ----------------------------------
# Relevence checking
def is_relevant(input_text, trusted_text):
    # Entity overlap
    input_ents = extract_entities_events(input_text)["entities"]
    trusted_ents = extract_entities_events(trusted_text)["entities"]

    for key in input_ents:
        if set(input_ents[key]) & set(trusted_ents[key]):
            return True

    # Semantic similarity fallback
    sim = semantic_similarity(input_text, trusted_text)
    return sim > 0.25   # this threshold works well for headlines



#-----------------------------------
# -----------------------------
# NLI comparison (confidence)
# -----------------------------
def nli_confidence_verdict(input_text: str, trusted_articles: list, top_n: int = 3, min_sim: float = 0.25):
    max_similarity = 0.0
    contradiction_score = 0.0
    results = []
    
    for trusted in trusted_articles[:20]:
        # Semantic similarity
        sim = semantic_similarity(input_text, trusted)
        max_similarity = max(max_similarity, sim)

        # NLI for contradiction only
        result = nli_model(input_text, text_pair=trusted)
        scores = result[0]

        contra = max(
            (s for s in scores if "contra" in s['label'].lower()),
            key=lambda x: x['score']
        )

        if contra['score'] > 0.8:
            contradiction_score = max(contradiction_score, contra['score'])

        results.append((trusted, sim, contra['score']))
        
    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    top_matches = [r for r in results if r[1] >= min_sim][:top_n]
    
    if contradiction_score > 0:
        verdict = f"🔴 Probably Fake: {int(contradiction_score*100)}% confidence"
        top_matches = [r for r in results if r[2] > 0.5][:top_n]
    elif max_similarity > 0.30:
        verdict = f"🟢 Likely Real: matched trusted reporting ({int(max_similarity*100)}% similarity)"
        top_matches = [r for r in results if r[1] >= min_sim][:top_n]
    else:
        verdict = "🟡 Needs Review (no supporting trusted news)"
        top_matches = []
        
    return verdict, top_matches


# -----------------------------
# Combined system (Hybrid)
# -----------------------------
def full_check(news_text):
    trusted_articles = fetch_trusted_news(use_cache=True)

    # Exact match check
    if check_exact_match(news_text, trusted_articles):
        return {
            #"Classification": "Real News",
            "ContradictionCheck": "🟢 Likely Real : Exact match found in trusted dataset (100% similarity)"
        }

    # Build entity state
    entity_state = build_entity_state(trusted_articles)

    # Logical contradiction check
    is_contra, reason = detect_logical_contradiction(news_text, entity_state)
    if is_contra:
        return {"ContradictionCheck_3": reason, "Matches": []} 

    # NLI fallback
    verdict, matches = nli_confidence_verdict(news_text, trusted_articles)
    return {"ContradictionCheck_3": verdict, "Matches": matches}

# -----------------------------

# -------------------------------
# Streamlit UI
# -----------------------------
#import streamlit as st

# Page setup
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# Centered heading
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>📰 Fake News Detection Demo</h1>", unsafe_allow_html=True)
st.markdown("---")

# Create two columns
col1, col2 = st.columns([1, 1])  # equal width, you can adjust ratio like [1,2]

# Left column: input
with col1:
    user_input = st.text_area("Enter a news headline or article:")
    check_button = st.button("Check")

# Right column: results
with col2:
    if check_button:
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            result = full_check(user_input)

            # Verdict display with color-coded boxes
            verdict_text = result.get("ContradictionCheck_3") or result.get("ContradictionCheck")
            if "Real" in verdict_text:
                st.success(verdict_text)
            elif "Fake" in verdict_text:
                st.error(verdict_text)
            else:
                st.warning(verdict_text)

            # Related matches section
            if result.get("Matches"):
                st.subheader("🔎 Related Matches")
                for trusted, sim, contra_score in result["Matches"]:
                    st.markdown(f"- **{trusted}**  \n   Similarity: `{sim:.2f}` | Contradiction: `{contra_score:.2f}`")
            else:
                st.info("No related matches found.")
# Sidebar cache control
with st.sidebar:
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared! Please rerun the app.")
# Footer
st.markdown("<hr>", unsafe_allow_html=True)
#st.markdown("<p style='text-align:center; color:grey;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
