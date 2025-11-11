import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

load_dotenv()

products = [
    {"name": "Boho Dress",
     "desc": "Flowy maxi dress, earthy tones, tassels — perfect for festival and free-spirited boho vibes.",
     "vibes": ["boho", "festival", "earthy"]},

    {"name": "Urban Bomber Jacket",
     "desc": "Cropped bomber jacket with sleek lines and reflective accents — energetic urban chic for city nights.",
     "vibes": ["urban", "chic", "energetic"]},

    {"name": "Cozy Knit Hoodie",
     "desc": "Oversized knit hoodie in soft pastels for laid-back cozy winter lounging.",
     "vibes": ["cozy", "casual", "warm"]},

    {"name": "Minimalist Slip Dress",
     "desc": "Clean silhouette slip dress; minimal, elegant and versatile for day-to-night styling.",
     "vibes": ["minimal", "elegant", "versatile"]},

    {"name": "Sporty Runner Sneakers",
     "desc": "Lightweight trainers built for motion — athletic, energetic and street-ready.",
     "vibes": ["sporty", "energetic", "street"]},

    {"name": "Retro Denim Jacket",
     "desc": "Distressed denim jacket with retro patches — nostalgic and playful vintage streetwear.",
     "vibes": ["retro", "vintage", "playful"]},

    {"name": "Silk Scarf",
     "desc": "Bright silk scarf with bold patterns, elevates simple outfits with artistic flair.",
     "vibes": ["artistic", "elevated", "bold"]},

    {"name": "Tailored Blazer",
     "desc": "Structured blazer in neutral tones — polished, professional and modern.",
     "vibes": ["polished", "professional", "modern"]},

    {"name": "Gothic Lace Top",
     "desc": "Dark lace top with intricate patterns — mysterious, edgy and dramatic for evening wear.",
     "vibes": ["gothic", "edgy", "dramatic"]},

    {"name": "Preppy Polo Shirt",
     "desc": "Classic polo shirt in crisp white with navy trim — timeless, clean and preppy aesthetic.",
     "vibes": ["preppy", "classic", "clean"]},

    {"name": "Glamorous Sequin Dress",
     "desc": "Sparkling sequin midi dress with flowing silhouette — glamorous, luxurious and party-ready.",
     "vibes": ["glamorous", "luxurious", "party"]},

    {"name": "Edgy Leather Pants",
     "desc": "Fitted leather pants with zipper details — bold, rebellious and street-smart.",
     "vibes": ["edgy", "rebellious", "bold"]},

    {"name": "Romantic Floral Midi Skirt",
     "desc": "Flowing midi skirt with delicate floral print — romantic, feminine and dreamy.",
     "vibes": ["romantic", "feminine", "dreamy"]},

    {"name": "Eco-Friendly Linen Jumpsuit",
     "desc": "Sustainable linen jumpsuit in natural earth tones — eco-conscious, comfortable and versatile.",
     "vibes": ["sustainable", "eco-friendly", "comfortable"]},

    {"name": "Futuristic Metallic Jacket",
     "desc": "Silver metallic puffer jacket with angular design — futuristic, tech-inspired and avant-garde.",
     "vibes": ["futuristic", "tech", "avant-garde"]},
]

df = pd.DataFrame(products)
df["text"] = df.apply(
    lambda r: f"{r['name']}. {r['desc']} Vibes: {', '.join(r['vibes'])}",
    axis=1
)

def try_openai_client():
    """Return OpenAI client if API key exists."""
    if not OPENAI_AVAILABLE:
        return None

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets["OPENAI_API_KEY"]
        except:
            return None

    try:
        return OpenAI(api_key=key)
    except:
        return None

def get_embedding_openai(text):
    """Return OpenAI embedding if available; else None."""
    try:
        client = try_openai_client()
        if client is None:
            return None

        resp = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)

    except Exception:
        return None


tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["text"])

def get_embedding_tfidf(text):
    return tfidf.transform([text]).toarray()[0]


def embed_text(text):
    """
    Try OPENAI → else TF-IDF fallback.
    Always returns a valid vector.
    """
    emb = get_embedding_openai(text)
    if emb is not None:
        return emb
    return get_embedding_tfidf(text)


product_embs = np.vstack([embed_text(t) for t in df["text"]])

def normalize(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm[norm == 0] = 1
    return x / norm

product_embs_norm = normalize(product_embs)


st.title("🧿 Vibe Matcher – Mini Fashion Recommender")
st.write("Type any vibe → get top-K fashion matches!")

query = st.text_input("Enter your vibe:")
threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.70)
top_k = st.slider("Top K Results", 1, 5, 3)

if st.button("Match Vibe"):
    if not query.strip():
        st.warning("Please enter a vibe!")
    else:
        start = time.time()

        q_emb = embed_text(query)
        q_emb_norm = normalize(q_emb.reshape(1, -1))[0]

        sims = cosine_similarity([q_emb_norm], product_embs_norm)[0]
        idx_sorted = sims.argsort()[::-1][:top_k]

        latency = time.time() - start

        st.subheader("Top Matches")

        for rank, idx in enumerate(idx_sorted, start=1):
            st.markdown(
                f"""
                **{rank}. {df.loc[idx, 'name']}**  
                *Similarity:* `{sims[idx]:.4f}`  
                *Vibes:* `{', '.join(df.loc[idx, 'vibes'])}`  
                *Description:* {df.loc[idx, 'desc']}
                """
            )
            st.markdown("---")

        if sims.max() < threshold:
            st.error("No strong match found — fallback triggered.")

        st.info(f"⏱ Latency: `{latency:.4f}s`")

with st.expander("View Product Catalog"):
    st.dataframe(df[["name", "desc", "vibes"]])
