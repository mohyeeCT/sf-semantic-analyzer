import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import re
import json
import requests

st.set_page_config(
    page_title="SF Semantic Analyzer",
    page_icon="🔍",
    layout="wide"
)

# ── Model registry ─────────────────────────────────────────────────────────────

MODELS = {
    "all-MiniLM-L6-v2 — Fast (default)": {
        "id":       "all-MiniLM-L6-v2",
        "dims":     384,
        "type":     "local",
        "note":     "Best for quick audits. Good general accuracy.",
    },
    "all-mpnet-base-v2 — Balanced": {
        "id":       "all-mpnet-base-v2",
        "dims":     768,
        "type":     "local",
        "note":     "2x the vector dimensions. Better at distinguishing near-similar pages. Recommended for most migrations.",
    },
    "mxbai-embed-large-v1 — Best local quality": {
        "id":       "mixedbread-ai/mxbai-embed-large-v1",
        "dims":     1024,
        "type":     "local",
        "note":     "Highest local accuracy. Slower on large crawls. Best for sites with thin metadata.",
    },
    "text-embedding-3-small — OpenAI (API key required)": {
        "id":       "text-embedding-3-small",
        "dims":     1536,
        "type":     "openai",
        "note":     "Best overall accuracy. Requires OpenAI API key. Cost is negligible (~$0.00002 per 1k URLs).",
    },
}

# ── SF column groups ───────────────────────────────────────────────────────────

SF_EMBED_COLS = [
    "Title 1", "H1-1", "H2-1", "H2-2", "H3-1",
    "Meta Description 1", "Meta Keywords 1",
]
SF_SCORE_COLS = ["Word Count", "Inlinks", "Unique Inlinks", "Crawl Depth 1", "Readability"]
SF_BODY_COLS  = ["Custom Extraction 1", "Custom Extraction 2", "Custom Extraction 3", "Body Text", "Body"]

INLINKS_DST_COL    = "Destination"
INLINKS_ANCHOR_COL = "Anchor Text"
INLINKS_SRC_TITLE  = "Source Title"
INLINKS_TYPE_COL   = "Type"

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_slug_tokens(url: str) -> str:
    try:
        path = urlparse(url).path
        path = re.sub(r"\.\w+$", "", path)
        return re.sub(r"[-_/]", " ", path).strip()
    except Exception:
        return ""

def clean_val(val) -> str:
    s = str(val).strip()
    return "" if s.lower() in ("nan", "n/a", "none", "") else s

def build_composite(row: pd.Series) -> str:
    parts = []
    for col in SF_EMBED_COLS:
        v = clean_val(row.get(col, ""))
        if v:
            parts.append(v)
    for col in SF_BODY_COLS:
        v = clean_val(row.get(col, ""))
        if v:
            parts.append(v[:400])
            break
    slug = extract_slug_tokens(clean_val(row.get("Address", "")))
    if slug:
        parts.append(slug)
    anchors = clean_val(row.get("_anchor_string", ""))
    if anchors:
        parts.append(anchors)
    src_titles = clean_val(row.get("_src_title_string", ""))
    if src_titles:
        parts.append(src_titles)
    return " | ".join(parts)

def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Status Code" in df.columns:
        df = df[df["Status Code"] == 200]
    if "Indexability" in df.columns:
        df = df[df["Indexability"].str.strip().str.lower() == "indexable"]
    return df.reset_index(drop=True)

def normalise(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)

def compute_page_value(df: pd.DataFrame, uniqueness: np.ndarray) -> pd.Series:
    idx = df.index
    u   = normalise(pd.Series(uniqueness, index=idx))
    has_wc    = "Word Count"    in df.columns
    has_il    = "Inlinks"       in df.columns
    has_depth = "Crawl Depth 1" in df.columns
    w_unique  = 0.50 + (0.20 * (not has_wc)) + (0.15 * (not has_il)) + (0.15 * (not has_depth))
    total = w_unique * u
    if has_wc:
        wc = pd.to_numeric(df["Word Count"], errors="coerce").fillna(0)
        total += 0.20 * normalise(wc)
    if has_il:
        il = pd.to_numeric(df["Inlinks"], errors="coerce").fillna(0)
        total += 0.15 * normalise(il)
    if has_depth:
        depth = pd.to_numeric(df["Crawl Depth 1"], errors="coerce").fillna(99)
        total += 0.15 * normalise(1 / (depth + 1))
    return (total * 100).round(1)

def migration_rec(max_sim: float) -> str:
    if max_sim >= 0.85: return "Consolidate / 301"
    if max_sim >= 0.70: return "Review overlap"
    return "Migrate"

# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_local_model(model_id: str) -> SentenceTransformer:
    return SentenceTransformer(model_id)

@st.cache_data(show_spinner="Generating embeddings...")
def get_local_embeddings(_model, texts: list) -> np.ndarray:
    return _model.encode(texts, batch_size=32, show_progress_bar=False)

def get_openai_embeddings(texts: list, api_key: str, model_id: str = "text-embedding-3-small") -> np.ndarray:
    """Call OpenAI embeddings API in batches of 500."""
    all_embeddings = []
    batch_size = 500
    progress = st.progress(0, text="Generating embeddings via OpenAI...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model_id, "input": batch},
        )
        data = response.json()
        if "error" in data:
            st.error(f"OpenAI API error: {data['error']['message']}")
            st.stop()
        batch_embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        all_embeddings.extend(batch_embeddings)
        progress.progress(min((i + batch_size) / len(texts), 1.0), text=f"Embedding {min(i+batch_size, len(texts))} / {len(texts)} pages...")
    progress.empty()
    return np.array(all_embeddings)

# ── Inlinks merge ──────────────────────────────────────────────────────────────

def merge_inlinks(df: pd.DataFrame, inlinks: pd.DataFrame) -> pd.DataFrame:
    if INLINKS_DST_COL not in inlinks.columns:
        st.warning("Inlinks file: no Destination column found. Skipping.")
        return df
    il = inlinks.copy()
    il[INLINKS_DST_COL] = il[INLINKS_DST_COL].astype(str).str.strip()
    if INLINKS_TYPE_COL in il.columns:
        il = il[il[INLINKS_TYPE_COL].str.lower().str.contains("html|hyperlink", na=False)]
    if INLINKS_ANCHOR_COL in il.columns:
        il[INLINKS_ANCHOR_COL] = il[INLINKS_ANCHOR_COL].astype(str).str.strip()
        il_a = il[~il[INLINKS_ANCHOR_COL].str.lower().isin(
            ["nan", "", "n/a", "click here", "read more", "here", "this", "link"]
        )]
        anchor_agg = (
            il_a.groupby(INLINKS_DST_COL)[INLINKS_ANCHOR_COL]
            .apply(lambda x: " | ".join(x.value_counts().head(8).index.tolist()))
            .reset_index()
            .rename(columns={INLINKS_DST_COL: "Address", INLINKS_ANCHOR_COL: "_anchor_string"})
        )
        df = df.merge(anchor_agg, on="Address", how="left")
    if INLINKS_SRC_TITLE in il.columns:
        il[INLINKS_SRC_TITLE] = il[INLINKS_SRC_TITLE].astype(str).str.strip()
        title_agg = (
            il[~il[INLINKS_SRC_TITLE].str.lower().isin(["nan", "", "n/a"])]
            .drop_duplicates(subset=[INLINKS_DST_COL, INLINKS_SRC_TITLE])
            .groupby(INLINKS_DST_COL)[INLINKS_SRC_TITLE]
            .apply(lambda x: " | ".join(x.head(5).tolist()))
            .reset_index()
            .rename(columns={INLINKS_DST_COL: "Address", INLINKS_SRC_TITLE: "_src_title_string"})
        )
        df = df.merge(title_agg, on="Address", how="left")
    inlink_count = (
        il.groupby(INLINKS_DST_COL).size()
        .reset_index(name="_inlink_count")
        .rename(columns={INLINKS_DST_COL: "Address"})
    )
    df = df.merge(inlink_count, on="Address", how="left")
    df["_inlink_count"] = df["_inlink_count"].fillna(0).astype(int)
    return df

# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("SF Semantic Analyzer")
st.caption("Screaming Frog · Semantic embeddings · Page value · Migration audit")

# ── Model selector ─────────────────────────────────────────────────────────────

st.subheader("Embedding Model")

model_label = st.selectbox(
    "Select model",
    options=list(MODELS.keys()),
    index=0,
    help="Higher quality models produce more accurate similarity scores but are slower."
)

model_cfg = MODELS[model_label]
st.caption(f"Dimensions: {model_cfg['dims']} — {model_cfg['note']}")

openai_key = None
if model_cfg["type"] == "openai":
    openai_key = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-...",
        help="Your key is used only for this session and never stored."
    )
    if not openai_key:
        st.warning("Enter your OpenAI API key to use this model.")

st.divider()

# ── File uploads ───────────────────────────────────────────────────────────────

st.subheader("Uniqueness Audit")
st.markdown("Single site analysis. Scores each page for semantic uniqueness and migration value using SF data only.")

with st.expander("SF Internal HTML — columns used"):
    st.markdown("""
**Required:** `Address`, `Status Code`, `Indexability`

**Embedded:** `Title 1`, `H1-1`, `H2-1`, `H2-2`, `H3-1`, `Meta Description 1`, `Meta Keywords 1`

**Body text (SF Custom Extraction):** `Custom Extraction 1/2/3`

**Scoring signals (not embedded):** `Word Count`, `Inlinks`, `Unique Inlinks`, `Crawl Depth 1`, `Readability`
    """)

with st.expander("SF All Inlinks export (optional but recommended)"):
    st.markdown("""
**How to export:** Bulk Exports > All Inlinks > CSV

**Columns used:** `Destination`, `Anchor Text`, `Source Title`, `Type`

Anchor text from inlinking pages is the strongest semantic signal available in SF without external data.
The app aggregates top 8 anchor texts and top 5 source page titles per destination URL.
    """)

sf_file = st.file_uploader("Screaming Frog Internal HTML CSV (required)", type="csv", key="sf1")
il_file = st.file_uploader("SF All Inlinks CSV (optional)", type="csv", key="il1")

if sf_file:
    raw = pd.read_csv(sf_file)
    df  = filter_df(raw)

    c1, c2 = st.columns(2)
    c1.metric("Total in export", len(raw))
    c2.metric("After filtering (200 + Indexable)", len(df))

    if len(df) == 0:
        st.error("No pages remain after filtering. Check Status Code and Indexability columns.")
        st.stop()

    sources = ["SF metadata"]
    if il_file:
        inlinks_df = pd.read_csv(il_file, low_memory=False)
        df = merge_inlinks(df, inlinks_df)
        sources.append("SF inlinks (anchor text + source titles)")

    st.info(f"Active sources: {', '.join(sources)}  |  Model: {model_cfg['id']}")

    found_embed = [c for c in SF_EMBED_COLS if c in df.columns]
    found_score = [c for c in SF_SCORE_COLS if c in df.columns]
    found_body  = next((c for c in SF_BODY_COLS if c in df.columns), None)
    has_anchors = "_anchor_string" in df.columns

    col_info = f"Embedding: {', '.join(found_embed)}"
    if found_body:  col_info += f", {found_body}"
    if has_anchors: col_info += ", anchor texts, source titles"
    if found_score: col_info += f" | Scoring: {', '.join(found_score)}"
    st.caption(f"Columns detected — {col_info}")

    df["Composite Text"] = df.apply(build_composite, axis=1)

    with st.expander("Preview composite text (first 10 rows)"):
        st.dataframe(df[["Address", "Composite Text"]].head(10), use_container_width=True)

    # Block run if OpenAI key is missing
    run_blocked = model_cfg["type"] == "openai" and not openai_key
    if run_blocked:
        st.button("Run Uniqueness Audit", type="primary", key="run1", disabled=True)
        st.caption("Add your OpenAI API key above to run.")
    elif st.button("Run Uniqueness Audit", type="primary", key="run1"):

        # ── Generate embeddings ────────────────────────────────────────────────
        texts = df["Composite Text"].tolist()

        if model_cfg["type"] == "openai":
            embeddings = get_openai_embeddings(texts, openai_key, model_cfg["id"])
        else:
            local_model = load_local_model(model_cfg["id"])
            embeddings  = get_local_embeddings(local_model, texts)

        # ── Similarity matrix ──────────────────────────────────────────────────
        with st.spinner("Calculating similarity matrix..."):
            sim = cosine_similarity(embeddings)
            np.fill_diagonal(sim, 0)
            max_sim        = sim.max(axis=1)
            best_match_idx = sim.argmax(axis=1)
            uniqueness     = 1 - max_sim
            page_value     = compute_page_value(df, uniqueness)

        result_cols = {
            "URL":              df["Address"].values,
            "Title":            df["Title 1"].values if "Title 1" in df.columns else "",
            "H1":               df["H1-1"].values    if "H1-1"    in df.columns else "",
            "Composite Text":   df["Composite Text"].values,
            "Model Used":       model_cfg["id"],
            "Uniqueness Score": uniqueness.round(4),
            "Max Similarity":   max_sim.round(4),
            "Most Similar URL": df["Address"].iloc[best_match_idx].values,
            "Page Value Score": page_value.values,
            "Recommendation":   [migration_rec(s) for s in max_sim],
        }

        for col, label in [
            ("Word Count",        "SF Word Count"),
            ("Inlinks",           "SF Inlinks"),
            ("Unique Inlinks",    "SF Unique Inlinks"),
            ("Crawl Depth 1",     "SF Crawl Depth"),
            ("Readability",       "SF Readability"),
            ("_inlink_count",     "Inlinks (inlinks export)"),
            ("_anchor_string",    "Anchor Texts"),
            ("_src_title_string", "Source Page Titles"),
        ]:
            if col in df.columns:
                result_cols[label] = df[col].values

        results = pd.DataFrame(result_cols)
        st.session_state["audit_results"] = results

        st.success(f"Done. {len(results)} pages analysed with {model_cfg['id']}.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Migrate",           len(results[results["Recommendation"] == "Migrate"]))
        m2.metric("Review overlap",    len(results[results["Recommendation"] == "Review overlap"]))
        m3.metric("Consolidate / 301", len(results[results["Recommendation"] == "Consolidate / 301"]))
        m4.metric("Avg Page Value",    f"{results['Page Value Score'].mean():.1f}")

        st.dataframe(results.sort_values("Page Value Score", ascending=False), use_container_width=True)

        st.download_button(
            "Download Results CSV",
            data=results.to_csv(index=False),
            file_name="uniqueness_audit.csv",
            mime="text/csv",
        )
