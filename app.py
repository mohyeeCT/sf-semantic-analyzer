import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import re
import io

st.set_page_config(
    page_title="SF Semantic Analyzer",
    page_icon="🔍",
    layout="wide"
)

# ── Model ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_slug_tokens(url: str) -> str:
    """Convert URL path into readable tokens for embedding."""
    try:
        path = urlparse(url).path
        path = re.sub(r"\.\w+$", "", path)           # strip file extensions
        tokens = re.sub(r"[-_/]", " ", path).strip() # hyphens/slashes → spaces
        return tokens
    except Exception:
        return ""


def build_composite(row: pd.Series) -> str:
    """Combine metadata fields + URL slug into a single embedding string."""
    fields = ["Title 1", "H1-1", "Meta Description 1"]
    parts = []
    for f in fields:
        val = str(row.get(f, "")).strip()
        if val and val.lower() not in ("nan", "n/a", ""):
            parts.append(val)
    slug = extract_slug_tokens(str(row.get("Address", "")))
    if slug:
        parts.append(slug)
    return " | ".join(parts)


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 200-status, indexable pages."""
    if "Status Code" in df.columns:
        df = df[df["Status Code"] == 200]
    if "Indexability" in df.columns:
        df = df[df["Indexability"].str.strip().str.lower() == "indexable"]
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Generating embeddings...")
def get_embeddings(_model, texts: list) -> np.ndarray:
    return _model.encode(texts, batch_size=32, show_progress_bar=False)


def migration_recommendation(max_sim: float) -> str:
    if max_sim >= 0.85:
        return "Consolidate / 301"
    elif max_sim >= 0.70:
        return "Review overlap"
    return "Migrate"


def confidence_label(score: float) -> str:
    if score >= 0.85:
        return "High"
    elif score >= 0.65:
        return "Medium"
    return "Low"

# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("SF Semantic Analyzer")
st.caption(
    "Screaming Frog export · Semantic embeddings · Page value without GSC data"
)

tab1, tab2 = st.tabs(["Uniqueness Audit", "Redirect Mapping"])

# ── TAB 1: UNIQUENESS AUDIT ───────────────────────────────────────────────────

with tab1:
    st.subheader("Uniqueness Audit")
    st.markdown(
        "Upload a single Screaming Frog internal HTML export. "
        "The app flags near-duplicate pages and scores each page's semantic uniqueness."
    )

    with st.expander("What columns are required?"):
        st.markdown(
            """
            **Required:** `Address`, `Status Code`, `Indexability`  
            **Used for embedding:** `Title 1`, `H1-1`, `Meta Description 1`  
            **Optional enrichment (not required):** `Word Count`, `H2-1`

            Use the default Screaming Frog internal HTML export. No column renaming needed.
            """
        )

    uploaded = st.file_uploader(
        "Upload SF Internal HTML CSV", type="csv", key="audit_upload"
    )

    if uploaded:
        raw = pd.read_csv(uploaded)
        df = filter_df(raw)

        col_a, col_b = st.columns(2)
        col_a.metric("Total pages in export", len(raw))
        col_b.metric("Pages after filtering (200 + Indexable)", len(df))

        if len(df) == 0:
            st.error(
                "No pages remain after filtering. "
                "Check that your export includes `Status Code` and `Indexability` columns."
            )
            st.stop()

        df["Composite Text"] = df.apply(build_composite, axis=1)

        with st.expander("Preview composite text (first 10 rows)"):
            st.dataframe(
                df[["Address", "Composite Text"]].head(10),
                use_container_width=True,
            )

        if st.button("Run Uniqueness Audit", type="primary"):
            embeddings = get_embeddings(model, df["Composite Text"].tolist())

            with st.spinner("Calculating similarity matrix..."):
                sim = cosine_similarity(embeddings)
                np.fill_diagonal(sim, 0)

                max_sim = sim.max(axis=1)
                best_match_idx = sim.argmax(axis=1)

                results = pd.DataFrame(
                    {
                        "URL": df["Address"],
                        "Title": df.get("Title 1", pd.Series(dtype=str)),
                        "H1": df.get("H1-1", pd.Series(dtype=str)),
                        "Composite Text": df["Composite Text"],
                        "Uniqueness Score": (1 - max_sim).round(4),
                        "Max Similarity": max_sim.round(4),
                        "Most Similar URL": df["Address"].iloc[best_match_idx].values,
                        "Recommendation": [
                            migration_recommendation(s) for s in max_sim
                        ],
                    }
                )

            st.success(f"Done. {len(results)} pages analysed.")

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Migrate",
                len(results[results["Recommendation"] == "Migrate"]),
            )
            m2.metric(
                "Review overlap",
                len(results[results["Recommendation"] == "Review overlap"]),
            )
            m3.metric(
                "Consolidate / 301",
                len(results[results["Recommendation"] == "Consolidate / 301"]),
            )

            st.dataframe(
                results.sort_values("Max Similarity", ascending=False),
                use_container_width=True,
            )

            st.download_button(
                "Download Results CSV",
                data=results.to_csv(index=False),
                file_name="uniqueness_audit.csv",
                mime="text/csv",
            )

# ── TAB 2: REDIRECT MAPPING ───────────────────────────────────────────────────

with tab2:
    st.subheader("Redirect Mapping")
    st.markdown(
        "Upload the old site and new site exports. "
        "The app maps each old URL to its best semantic match on the new site."
    )

    with st.expander("What columns are required?"):
        st.markdown(
            """
            Same columns as Uniqueness Audit for both exports.  
            The app compares page composites **across** the two sites, not within a single site.
            """
        )

    left, right = st.columns(2)
    old_file = left.file_uploader("Old Site CSV", type="csv", key="old_upload")
    new_file = right.file_uploader("New Site CSV", type="csv", key="new_upload")

    top_n = st.slider(
        "Top N matches per old URL", min_value=1, max_value=5, value=3
    )

    sim_threshold = st.slider(
        "Minimum similarity score to include", min_value=0.0, max_value=1.0,
        value=0.0, step=0.05,
        help="Set to 0 to include all matches regardless of score."
    )

    if old_file and new_file:
        df_old = filter_df(pd.read_csv(old_file))
        df_new = filter_df(pd.read_csv(new_file))

        o1, o2 = st.columns(2)
        o1.metric("Old site pages (filtered)", len(df_old))
        o2.metric("New site pages (filtered)", len(df_new))

        if len(df_old) == 0 or len(df_new) == 0:
            st.error("One or both exports returned no pages after filtering.")
            st.stop()

        df_old["Composite Text"] = df_old.apply(build_composite, axis=1)
        df_new["Composite Text"] = df_new.apply(build_composite, axis=1)

        if st.button("Run Redirect Mapping", type="primary"):
            emb_old = get_embeddings(model, df_old["Composite Text"].tolist())
            emb_new = get_embeddings(model, df_new["Composite Text"].tolist())

            with st.spinner("Calculating cross-site similarity..."):
                sim = cosine_similarity(emb_old, emb_new)

                rows = []
                for i in range(len(df_old)):
                    scores = sim[i]
                    top_idx = scores.argsort()[::-1][:top_n]
                    for rank, j in enumerate(top_idx, 1):
                        score = float(scores[j])
                        if score < sim_threshold:
                            continue
                        rows.append(
                            {
                                "Old URL": df_old["Address"].iloc[i],
                                "Old Title": df_old.get(
                                    "Title 1", pd.Series(dtype=str)
                                ).iloc[i] if "Title 1" in df_old.columns else "",
                                "Old Composite": df_old["Composite Text"].iloc[i],
                                "Match Rank": rank,
                                "New URL": df_new["Address"].iloc[j],
                                "New Title": df_new.get(
                                    "Title 1", pd.Series(dtype=str)
                                ).iloc[j] if "Title 1" in df_new.columns else "",
                                "New Composite": df_new["Composite Text"].iloc[j],
                                "Similarity Score": round(score, 4),
                                "Confidence": confidence_label(score),
                            }
                        )

                results = pd.DataFrame(rows)

            st.success(
                f"Done. {len(df_old)} old URLs mapped against {len(df_new)} new pages."
            )

            top1 = results[results["Match Rank"] == 1]
            c1, c2, c3 = st.columns(3)
            c1.metric("High confidence", len(top1[top1["Confidence"] == "High"]))
            c2.metric("Medium confidence", len(top1[top1["Confidence"] == "Medium"]))
            c3.metric("Low confidence", len(top1[top1["Confidence"] == "Low"]))

            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download Redirect Map CSV",
                data=results.to_csv(index=False),
                file_name="redirect_mapping.csv",
                mime="text/csv",
            )
