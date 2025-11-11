# Vibe Matcher (Prototype)

A simple Streamlit prototype that recommends catalog items that *match your vibe*.  
It takes a free‑form text description (e.g., _“minimalist, cozy fall outfit with earthy tones”_) and returns the most similar items from a tiny in‑memory product catalog.

The app supports two matching backends:

1. **Local TF‑IDF + Cosine Similarity** (default) — fast, no external services.
2. **OpenAI Embeddings** (optional) — uses `text-embedding-ada-002` to embed the query and item descriptions, then ranks by cosine similarity.

---

## Demo (What it does)

- Type a vibe or a short description (“elevated streetwear”, “bold vintage look”, “clean silhouette, minimalist”).
- The app computes text similarity between your prompt and each item in the catalog.
- You’ll see **Top‑K** matches with a similarity score, the item’s vibes/tags, and a short description.
- It also shows **latency** so you can compare local TF‑IDF vs. OpenAI embedding modes.
- You can expand **“View Product Catalog”** to see the synthetic items bundled with the app.

> **Note:** This is a prototype over a toy catalog to demonstrate the interaction pattern, not a production recommender.

---

## Project Structure

```
.
├── vibe-prototype.py      # Streamlit app (UI + ranking logic + toy catalog)
└── README.md              # This file
```

Key pieces inside `vibe-prototype.py`:

- `get_embedding_openai(text)`: (optional) Uses OpenAI to create a vector for a string.
- `rank_with_tfidf(query, df)`: Builds TF‑IDF vectors for item descriptions + query; ranks items by cosine similarity.
- `rank_with_openai(query, df)`: If OpenAI is configured, embeds each item + query and ranks by cosine similarity.
- Streamlit UI: text input, **Top‑K** slider, **Similarity Threshold**, a toggle for **Use OpenAI**, and results table/cards.

---

## Requirements

- Python **3.11** (recommended). Streamlit and some libs can be finicky on 3.12.
- Packages:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `python-dotenv`
  - `openai` *(optional — only if you want the embedding mode)*

---

## Quickstart

### 1) Create & activate a virtual environment (recommended)

**Conda**:

```bash
conda create -n vibe python=3.11 -y
conda activate vibe
```

**venv**:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip wheel setuptools
pip install streamlit pandas numpy scikit-learn python-dotenv
# Optional: only if you plan to use OpenAI embeddings
pip install openai
```

### 3) (Optional) Configure OpenAI

Create a `.env` file next to `vibe-prototype.py` and add:

```
OPENAI_API_KEY=sk-...
```

The app uses `dotenv` to load it automatically. If the key or the `openai` package is missing, the app will silently fall back to the TF‑IDF mode.

### 4) Run the app

```bash
streamlit run vibe-prototype.py
```

Then open the local URL that Streamlit prints (usually http://localhost:8501).

---

## Usage Tips

- Keep your **query** short and descriptive (2–20 words tends to work well).
- Adjust **Top‑K** to see more or fewer results.
- Increase the **Similarity Threshold** if you want to filter weaker matches.
- Use **OpenAI mode** when your catalog grows beyond a few dozen items or when you want better semantic matching.
- The built‑in catalog lives in a pandas DataFrame with columns:
  - `name` (str): product title
  - `desc` (str): product description
  - `vibes` (list[str]): tags like `"minimalist"`, `"street"`, `"vintage"`

---

## How It Works (Under the Hood)

### TF‑IDF Path
1. Build a corpus from each item’s `desc` (and optionally its `vibes`).
2. Fit a `TfidfVectorizer`, vectorize the catalog and the user query.
3. Compute `cosine_similarity(query_vector, item_vectors)` to rank items.

### OpenAI Embeddings Path (Optional)
1. Use `text-embedding-ada-002` to embed each `desc` (and/or the vibes).
2. Embed the user query.
3. Compute cosine similarity between the query embedding and each item embedding.
4. Rank and return the top results.

Both paths report latency so you can compare performance.

---

## Customization

- **Add/Edit Items:** Modify the in‑code list that seeds the DataFrame.
- **Change Ranking Signals:** Concatenate vibes into the description, or add weights to title/desc/vibes.
- **Swap Models:** Replace TF‑IDF with another local embedding model (e.g., `sentence-transformers`) to avoid API usage.
- **Persist a Real Catalog:** Replace the toy DataFrame with a CSV/DB fetch, then cache embeddings for speed.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'plotly'`**  
  This app doesn’t require Plotly; remove any unrelated imports. Make sure you installed only the packages above.
  
- **`streamlit` import errors on Python 3.12**  
  Use Python 3.11 (`conda create -n vibe python=3.11`) to avoid known compatibility issues.

- **OpenAI not found or no key**  
  The app will fall back to TF‑IDF. Install `openai` and set `OPENAI_API_KEY` to enable embedding mode.

- **Nothing matches my query**  
  Lower the **Similarity Threshold** or try a different query (e.g., “retro denim jacket”, “minimalist neutral dress”).

---

## Roadmap Ideas

- Real product catalog + images.
- Cached embeddings for fast startup.
- Reranking with a small LLM or cross‑encoder.
- Multi‑vector scoring: title vs description vs tags.
- User feedback loop (👍/👎) to learn preferences.

---

## License

MIT License — feel free to use and adapt for your prototypes.