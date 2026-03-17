import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import re

st.set_page_config(
    page_title="SF Semantic Analyzer",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

SF_EMBED_COLS = ["Title 1", "H1-1", "H2-1", "H2-2", "Meta Description 1"]
SEMRUSH_URL_COL      = "URL"
SEMRUSH_KW_COL       = "Top keyword"
SEMRUSH_VOL_COL      = "Search Volume"
SEMRUSH_POS_COL      = "Position"
SEMRUSH_TRAFFIC_COL  = "Traffic"
SEMRUSH_KW_COUNT_COL = "Number of Keywords"
GA4_URL_COL          = "Landing page"
GA4_SESSIONS_COL     = "Sessions"
GA4_ENGAGEMENT_COL   = "Engagement rate"
GA4_EVENTS_COL       = "Event count"

def extract_slug_tokens(url: str) -> str:
    try:
        path = urlparse(url).path
        path = re.sub(r"\.\w+$", "", path)
        tokens = re.sub(r"[-_/]", " ", path).strip()
        return tokens
    except Exception:
        return ""

def clean_val(val) -> str:
    s = str(val).strip()
    return "" if s.lower() in ("nan", "n/a", "none", "") else s

def build_composite(row: pd.Series, extra_cols: list = None) -> str:
    parts = []
    for col in SF_EMBED_COLS:
        v = clean_val(row.get(col, ""))
        if v:
            parts.append(v)
    for col in ["Custom Extraction 1", "Body Text", "Body"]:
        v = clean_val(row.get(col, ""))
        if v:
            parts.append(v[:300])
            break
    slug = extract_slug_tokens(clean_val(row.get("Address", "")))
    if slug:
        parts.append(slug)
    if extra_cols:
        for col in extra_cols:
            v = clean_val(row.get(col, ""))
            if v:
                parts.append(v)
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
    scores = pd.DataFrame(index=df.index)
    scores["uniqueness"] = normalise(pd.Series(uniqueness, index=df.index))

    if "Word Count" in df.columns:
        wc = pd.to_numeric(df["Word Count"], errors="coerce").fillna(0)
        scores["richness"] = normalise(wc)
    else:
        scores["richness"] = 0.0

    has_ga4 = "_ga4_sessions" in df.columns and pd.to_numeric(df["_ga4_sessions"], errors="coerce").fillna(0).sum() > 0
    has_semrush = "_semrush_traffic" in df.columns and pd.to_numeric(df["_semrush_traffic"], errors="coerce").fillna(0).sum() > 0
    if not has_semrush:
        has_semrush = "_semrush_kw_count" in df.columns and pd.to_numeric(df["_semrush_kw_count"], errors="coerce").fillna(0).sum() > 0

    w_unique  = 0.40
    w_rich    = 0.20
    w_ga4     = 0.20 if has_ga4 else 0.0
    w_semrush = 0.20 if has_semrush else 0.0
    unused    = (0.20 * (not has_ga4)) + (0.20 * (not has_semrush))
    w_unique  += unused

    total = w_unique * scores["uniqueness"] + w_rich * scores["richness"]

    if has_ga4:
        s = pd.to_numeric(df["_ga4_sessions"], errors="coerce").fillna(0)
        total += w_ga4 * normalise(s)

    if has_semrush:
        col = "_semrush_traffic" if "_semrush_traffic" in df.columns else "_semrush_kw_count"
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        total += w_semrush * normalise(s)

    return (total * 100).round(1)

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

@st.cache_data(show_spinner="Generating embeddings...")
def get_embeddings(_model, texts: list) -> np.ndarray:
    return _model.encode(texts, batch_size=32, show_progress_bar=False)

def merge_semrush(df: pd.DataFrame, semrush: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    if SEMRUSH_KW_COL in semrush.columns:
        col_map[SEMRUSH_KW_COL] = "_semrush_top_kw"
    if SEMRUSH_TRAFFIC_COL in semrush.columns:
        col_map[SEMRUSH_TRAFFIC_COL] = "_semrush_traffic"
    if SEMRUSH_KW_COUNT_COL in semrush.columns:
        col_map[SEMRUSH_KW_COUNT_COL] = "_semrush_kw_count"
    if SEMRUSH_VOL_COL in semrush.columns:
        col_map[SEMRUSH_VOL_COL] = "_semrush_vol"
    if not col_map:
        st.warning("Semrush file uploaded but no recognised columns found. Skipping.")
        return df
    semrush_slim = semrush[[SEMRUSH_URL_COL] + list(col_map.keys())].copy()
    semrush_slim = semrush_slim.rename(columns={SEMRUSH_URL_COL: "Address", **col_map})
    semrush_slim["Address"] = semrush_slim["Address"].str.strip()
    return df.merge(semrush_slim, on="Address", how="left")

def merge_ga4(df: pd.DataFrame, ga4: pd.DataFrame) -> pd.DataFrame:
    url_col = None
    for c in [GA4_URL_COL, "Page path", "Page path and screen class", "URL"]:
        if c in ga4.columns:
            url_col = c
            break
    if url_col is None:
        st.warning("GA4 file uploaded but no URL/page path column found. Skipping.")
        return df
    col_map = {}
    if GA4_SESSIONS_COL in ga4.columns:
        col_map[GA4_SESSIONS_COL] = "_ga4_sessions"
    if GA4_ENGAGEMENT_COL in ga4.columns:
        col_map[GA4_ENGAGEMENT_COL] = "_ga4_engagement"
    if GA4_EVENTS_COL in ga4.columns:
        col_map[GA4_EVENTS_COL] = "_ga4_events"
    ga4_slim = ga4[[url_col] + list(col_map.keys())].copy()
    ga4_slim = ga4_slim.rename(columns={url_col: "_ga4_url", **col_map})
    ga4_slim["_ga4_url"] = ga4_slim["_ga4_url"].str.strip()
    sample = ga4_slim["_ga4_url"].dropna().iloc[0] if len(ga4_slim) else ""
    if not sample.startswith("http"):
        df["_join_key"] = df["Address"].apply(lambda u: urlparse(u).path)
        ga4_slim = ga4_slim.rename(columns={"_ga4_url": "_join_key"})
        df = df.merge(ga4_slim, on="_join_key", how="left").drop(columns=["_join_key"])
    else:
        ga4_slim = ga4_slim.rename(columns={"_ga4_url": "Address"})
        df = df.merge(ga4_slim, on="Address", how="left")
    return df

# ── UI ──────────────────────────────────────────────────────────────────────

st.title("SF Semantic Analyzer")
st.caption("Screaming Frog · Semrush · GA4 · Semantic embeddings · Page value assessment")

tab1, tab2 = st.tabs(["Uniqueness Audit", "Redirect Mapping"])

# ── TAB 1 ───────────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Uniqueness Audit")
    st.markdown("Upload a Screaming Frog internal HTML export. Optionally enrich with Semrush and GA4 for a full page value score.")

    with st.expander("SF columns used"):
        st.markdown("""
**Required:** `Address`, `Status Code`, `Indexability`

**Embedded (semantic):** `Title 1`, `H1-1`, `H2-1`, `H2-2`, `Meta Description 1`

**Scoring signals (not embedded):** `Word Count`, `Inlinks`, `Crawl Depth 1`

**Custom body text (if extracted in SF):** `Custom Extraction 1` or `Body Text`
        """)

    with st.expander("Semrush — expected columns"):
        st.markdown("""
Export from **Semrush > Organic Research > Pages**.

Columns used: `URL`, `Top keyword`, `Traffic`, `Number of Keywords`

The top keyword is appended to the embedding composite. Traffic and keyword count feed the value score.
        """)

    with st.expander("GA4 — expected columns"):
        st.markdown("""
Export from **GA4 > Engagement > Landing pages** or **Pages and screens**.

Columns used: `Landing page` (or `Page path`), `Sessions`, `Engagement rate`

GA4 can export full URLs or paths only — the app handles both.
        """)

    sf_file      = st.file_uploader("Screaming Frog CSV (required)", type="csv", key="sf_audit")
    semrush_file = st.file_uploader("Semrush Organic Pages CSV (optional)", type="csv", key="sr_audit")
    ga4_file     = st.file_uploader("GA4 Landing Pages CSV (optional)", type="csv", key="ga4_audit")

    if sf_file:
        raw = pd.read_csv(sf_file)
        df  = filter_df(raw)

        c1, c2 = st.columns(2)
        c1.metric("Total in export", len(raw))
        c2.metric("After filtering (200 + Indexable)", len(df))

        if len(df) == 0:
            st.error("No pages remain after filtering. Check Status Code and Indexability columns.")
            st.stop()

        sources_active = ["SF metadata"]

        if semrush_file:
            sr = pd.read_csv(semrush_file)
            df = merge_semrush(df, sr)
            sources_active.append("Semrush")

        if ga4_file:
            g4 = pd.read_csv(ga4_file)
            df = merge_ga4(df, g4)
            sources_active.append("GA4")

        st.info(f"Active data sources: {', '.join(sources_active)}")

        semrush_extra = ["_semrush_top_kw"] if "_semrush_top_kw" in df.columns else []
        df["Composite Text"] = df.apply(lambda row: build_composite(row, extra_cols=semrush_extra), axis=1)

        with st.expander("Preview composite text (first 10 rows)"):
            st.dataframe(df[["Address", "Composite Text"]].head(10), use_container_width=True)

        if st.button("Run Uniqueness Audit", type="primary", key="run_audit"):
            embeddings = get_embeddings(model, df["Composite Text"].tolist())

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
                "Uniqueness Score": uniqueness.round(4),
                "Max Similarity":   max_sim.round(4),
                "Most Similar URL": df["Address"].iloc[best_match_idx].values,
                "Page Value Score": page_value.values,
                "Recommendation":   [migration_recommendation(s) for s in max_sim],
            }

            for col, label in [
                ("Word Count",        "SF Word Count"),
                ("Inlinks",           "SF Inlinks"),
                ("Crawl Depth 1",     "SF Crawl Depth"),
                ("_semrush_top_kw",   "Semrush Top Keyword"),
                ("_semrush_traffic",  "Semrush Traffic"),
                ("_semrush_kw_count", "Semrush Keyword Count"),
                ("_ga4_sessions",     "GA4 Sessions"),
                ("_ga4_engagement",   "GA4 Engagement Rate"),
            ]:
                if col in df.columns:
                    result_cols[label] = df[col].values

            results = pd.DataFrame(result_cols)

            st.success(f"Done. {len(results)} pages analysed.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Migrate",           len(results[results["Recommendation"] == "Migrate"]))
            m2.metric("Review overlap",    len(results[results["Recommendation"] == "Review overlap"]))
            m3.metric("Consolidate / 301", len(results[results["Recommendation"] == "Consolidate / 301"]))
            m4.metric("Avg Page Value",    f"{results['Page Value Score'].mean():.1f}")

            st.dataframe(results.sort_values("Page Value Score", ascending=False), use_container_width=True)

            st.download_button(
                "Download Results CSV",
                data=results.to_csv(index=False),
                file_name="uniqueness_audit_enriched.csv",
                mime="text/csv",
            )

# ── TAB 2 ───────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Redirect Mapping")
    st.markdown("Upload old and new site SF exports. Optionally enrich old site with Semrush and GA4 to prioritise which old URLs matter most.")

    with st.expander("Column requirements"):
        st.markdown("""
Same SF columns as Uniqueness Audit for both exports.

Semrush and GA4 apply to the **old site only** — they signal which old URLs carried real traffic and need careful mapping.
        """)

    left, right = st.columns(2)
    old_sf  = left.file_uploader("Old Site SF CSV",                  type="csv", key="old_sf")
    new_sf  = right.file_uploader("New Site SF CSV",                 type="csv", key="new_sf")
    old_sr  = left.file_uploader("Old Site Semrush CSV (optional)",  type="csv", key="old_sr")
    old_ga4 = right.file_uploader("Old Site GA4 CSV (optional)",     type="csv", key="old_ga4")

    top_n         = st.slider("Top N matches per URL", 1, 5, 3)
    sim_threshold = st.slider("Minimum similarity score", 0.0, 1.0, 0.0, 0.05)

    if old_sf and new_sf:
        df_old = filter_df(pd.read_csv(old_sf))
        df_new = filter_df(pd.read_csv(new_sf))

        o1, o2 = st.columns(2)
        o1.metric("Old site pages (filtered)", len(df_old))
        o2.metric("New site pages (filtered)", len(df_new))

        if len(df_old) == 0 or len(df_new) == 0:
            st.error("One or both exports returned no pages after filtering.")
            st.stop()

        sources_active = ["SF metadata"]

        if old_sr:
            df_old = merge_semrush(df_old, pd.read_csv(old_sr))
            sources_active.append("Semrush (old site)")

        if old_ga4:
            df_old = merge_ga4(df_old, pd.read_csv(old_ga4))
            sources_active.append("GA4 (old site)")

        st.info(f"Active data sources: {', '.join(sources_active)}")

        semrush_extra = ["_semrush_top_kw"] if "_semrush_top_kw" in df_old.columns else []
        df_old["Composite Text"] = df_old.apply(lambda row: build_composite(row, extra_cols=semrush_extra), axis=1)
        df_new["Composite Text"] = df_new.apply(build_composite, axis=1)

        if st.button("Run Redirect Mapping", type="primary", key="run_redirect"):
            emb_old = get_embeddings(model, df_old["Composite Text"].tolist())
            emb_new = get_embeddings(model, df_new["Composite Text"].tolist())

            with st.spinner("Calculating cross-site similarity..."):
                sim = cosine_similarity(emb_old, emb_new)
                rows = []
                for i in range(len(df_old)):
                    scores  = sim[i]
                    top_idx = scores.argsort()[::-1][:top_n]
                    for rank, j in enumerate(top_idx, 1):
                        score = float(scores[j])
                        if score < sim_threshold:
                            continue
                        row = {
                            "Old URL":          df_old["Address"].iloc[i],
                            "Old Title":        df_old["Title 1"].iloc[i] if "Title 1" in df_old.columns else "",
                            "Old Composite":    df_old["Composite Text"].iloc[i],
                            "Match Rank":       rank,
                            "New URL":          df_new["Address"].iloc[j],
                            "New Title":        df_new["Title 1"].iloc[j] if "Title 1" in df_new.columns else "",
                            "New Composite":    df_new["Composite Text"].iloc[j],
                            "Similarity Score": round(score, 4),
                            "Confidence":       confidence_label(score),
                        }
                        for col, label in [
                            ("_semrush_top_kw",  "Old Semrush Top KW"),
                            ("_semrush_traffic", "Old Semrush Traffic"),
                            ("_ga4_sessions",    "Old GA4 Sessions"),
                            ("_ga4_engagement",  "Old GA4 Engagement"),
                        ]:
                            if col in df_old.columns:
                                row[label] = df_old[col].iloc[i]
                        rows.append(row)

                results = pd.DataFrame(rows)

            st.success(f"Done. {len(df_old)} old URLs mapped against {len(df_new)} new pages.")

            top1 = results[results["Match Rank"] == 1]
            c1, c2, c3 = st.columns(3)
            c1.metric("High confidence",   len(top1[top1["Confidence"] == "High"]))
            c2.metric("Medium confidence", len(top1[top1["Confidence"] == "Medium"]))
            c3.metric("Low confidence",    len(top1[top1["Confidence"] == "Low"]))

            sort_col = "Old GA4 Sessions" if "Old GA4 Sessions" in results.columns else "Similarity Score"
            st.dataframe(results.sort_values(sort_col, ascending=False), use_container_width=True)

            st.download_button(
                "Download Redirect Map CSV",
                data=results.to_csv(index=False),
                file_name="redirect_mapping_enriched.csv",
                mime="text/csv",
            )
