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

# ── Model ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ── Column constants ───────────────────────────────────────────────────────────

SF_EMBED_COLS = ["Title 1", "H1-1", "H2-1", "H2-2", "Meta Description 1"]

# Semrush Pages export (Organic Research > Pages)
SR_PAGES_URL     = "URL"
SR_PAGES_KW      = "Top keyword"
SR_PAGES_TRAFFIC = "Traffic"
SR_PAGES_KW_COUNT= "Number of Keywords"
SR_PAGES_VOL     = "Search Volume"

# Semrush Positions export (Organic Research > Positions)
SR_POS_URL     = "URL"
SR_POS_KW      = "Keyword"
SR_POS_VOL     = "Search Volume"
SR_POS_POS     = "Position"
SR_POS_TRAFFIC = "Traffic"

# GA4 export
GA4_COLS_URL = ["Landing page", "Page path", "Page path and screen class", "URL"]
GA4_SESSIONS = "Sessions"
GA4_ENGAGE   = "Engagement rate"
GA4_EVENTS   = "Event count"

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
    idx = df.index
    scores = pd.DataFrame(index=idx)
    scores["uniqueness"] = normalise(pd.Series(uniqueness, index=idx))

    wc = pd.to_numeric(df.get("Word Count", pd.Series(0, index=idx)), errors="coerce").fillna(0)
    scores["richness"] = normalise(wc)

    has_ga4 = "_ga4_sessions" in df.columns and pd.to_numeric(df["_ga4_sessions"], errors="coerce").fillna(0).sum() > 0
    has_sr  = "_sr_traffic"   in df.columns and pd.to_numeric(df["_sr_traffic"],   errors="coerce").fillna(0).sum() > 0
    if not has_sr:
        has_sr = "_sr_kw_count" in df.columns and pd.to_numeric(df["_sr_kw_count"], errors="coerce").fillna(0).sum() > 0

    w_unique  = 0.40 + (0.20 * (not has_ga4)) + (0.20 * (not has_sr))
    w_rich    = 0.20
    w_ga4     = 0.20 if has_ga4 else 0.0
    w_sr      = 0.20 if has_sr  else 0.0

    total = w_unique * scores["uniqueness"] + w_rich * scores["richness"]
    if has_ga4:
        s = pd.to_numeric(df["_ga4_sessions"], errors="coerce").fillna(0)
        total += w_ga4 * normalise(s)
    if has_sr:
        col = "_sr_traffic" if "_sr_traffic" in df.columns else "_sr_kw_count"
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        total += w_sr * normalise(s)

    return (total * 100).round(1)

def migration_rec(max_sim: float) -> str:
    if max_sim >= 0.85: return "Consolidate / 301"
    if max_sim >= 0.70: return "Review overlap"
    return "Migrate"

def confidence_label(score: float) -> str:
    if score >= 0.85: return "High"
    if score >= 0.65: return "Medium"
    return "Low"

@st.cache_data(show_spinner="Generating embeddings...")
def get_embeddings(_model, texts: list) -> np.ndarray:
    return _model.encode(texts, batch_size=32, show_progress_bar=False)

# ── Semrush merges ─────────────────────────────────────────────────────────────

def merge_sr_pages(df: pd.DataFrame, sr: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    if SR_PAGES_KW       in sr.columns: col_map[SR_PAGES_KW]       = "_sr_top_kw"
    if SR_PAGES_TRAFFIC  in sr.columns: col_map[SR_PAGES_TRAFFIC]  = "_sr_traffic"
    if SR_PAGES_KW_COUNT in sr.columns: col_map[SR_PAGES_KW_COUNT] = "_sr_kw_count"
    if SR_PAGES_VOL      in sr.columns: col_map[SR_PAGES_VOL]      = "_sr_top_vol"
    if not col_map:
        st.warning("Semrush Pages file: no recognised columns. Skipping.")
        return df
    slim = sr[[SR_PAGES_URL] + list(col_map.keys())].copy()
    slim = slim.rename(columns={SR_PAGES_URL: "Address", **col_map})
    slim["Address"] = slim["Address"].str.strip()
    return df.merge(slim, on="Address", how="left")

def merge_sr_positions(df: pd.DataFrame, sr: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate positions export: group by URL, take top 5 keywords by volume,
    concatenate into _sr_kw_string and append to composite.
    """
    required = [SR_POS_URL, SR_POS_KW]
    if not all(c in sr.columns for c in required):
        st.warning("Semrush Positions file: missing URL or Keyword column. Skipping.")
        return df

    vol_col = SR_POS_VOL if SR_POS_VOL in sr.columns else None

    sr = sr[[SR_POS_URL, SR_POS_KW] + ([vol_col] if vol_col else [])].copy()
    sr[SR_POS_URL] = sr[SR_POS_URL].str.strip()

    if vol_col:
        sr[vol_col] = pd.to_numeric(sr[vol_col], errors="coerce").fillna(0)
        sr = sr.sort_values(vol_col, ascending=False)

    agg = (
        sr.groupby(SR_POS_URL)[SR_POS_KW]
        .apply(lambda x: " | ".join(x.head(5).tolist()))
        .reset_index()
        .rename(columns={SR_POS_URL: "Address", SR_POS_KW: "_sr_kw_string"})
    )

    if vol_col:
        vol_agg = (
            sr.groupby(SR_POS_URL)[vol_col]
            .sum()
            .reset_index()
            .rename(columns={SR_POS_URL: "Address", vol_col: "_sr_total_vol"})
        )
        agg = agg.merge(vol_agg, on="Address", how="left")

    kw_count = (
        sr.groupby(SR_POS_URL)[SR_POS_KW]
        .count()
        .reset_index()
        .rename(columns={SR_POS_URL: "Address", SR_POS_KW: "_sr_pos_kw_count"})
    )
    agg = agg.merge(kw_count, on="Address", how="left")

    return df.merge(agg, on="Address", how="left")

def merge_ga4(df: pd.DataFrame, ga4: pd.DataFrame) -> pd.DataFrame:
    url_col = next((c for c in GA4_COLS_URL if c in ga4.columns), None)
    if url_col is None:
        st.warning("GA4 file: no URL/page path column found. Skipping.")
        return df
    col_map = {}
    if GA4_SESSIONS in ga4.columns: col_map[GA4_SESSIONS] = "_ga4_sessions"
    if GA4_ENGAGE   in ga4.columns: col_map[GA4_ENGAGE]   = "_ga4_engagement"
    if GA4_EVENTS   in ga4.columns: col_map[GA4_EVENTS]   = "_ga4_events"
    slim = ga4[[url_col] + list(col_map.keys())].copy()
    slim = slim.rename(columns={url_col: "_ga4_url", **col_map})
    slim["_ga4_url"] = slim["_ga4_url"].str.strip()
    sample = slim["_ga4_url"].dropna().iloc[0] if len(slim) else ""
    if not sample.startswith("http"):
        df["_join_key"] = df["Address"].apply(lambda u: urlparse(u).path)
        slim = slim.rename(columns={"_ga4_url": "_join_key"})
        df = df.merge(slim, on="_join_key", how="left").drop(columns=["_join_key"])
    else:
        slim = slim.rename(columns={"_ga4_url": "Address"})
        df = df.merge(slim, on="Address", how="left")
    return df

# ── AI report ──────────────────────────────────────────────────────────────────

def build_report_prompt(results: pd.DataFrame, mode: str, site_name: str) -> str:
    total = len(results)
    rec_counts = results["Recommendation"].value_counts().to_dict() if "Recommendation" in results.columns else {}

    migrate_df = results[results["Recommendation"] == "Migrate"].sort_values("Page Value Score", ascending=False) if "Recommendation" in results.columns else pd.DataFrame()
    consolidate_df = results[results["Recommendation"] == "Consolidate / 301"].sort_values("Max Similarity", ascending=False) if "Recommendation" in results.columns else pd.DataFrame()
    review_df = results[results["Recommendation"] == "Review overlap"] if "Recommendation" in results.columns else pd.DataFrame()

    top_migrate = migrate_df.head(15)[["URL", "Page Value Score", "Max Similarity", "Most Similar URL"]].to_string(index=False) if len(migrate_df) else "None"
    top_consolidate = consolidate_df.head(15)[["URL", "Max Similarity", "Most Similar URL"]].to_string(index=False) if len(consolidate_df) else "None"
    top_review = review_df.head(10)[["URL", "Max Similarity", "Most Similar URL"]].to_string(index=False) if len(review_df) else "None"

    has_ga4     = "GA4 Sessions" in results.columns
    has_semrush = "Semrush Traffic" in results.columns or "Semrush Keyword Count" in results.columns

    value_stats = results["Page Value Score"].describe().round(2).to_string() if "Page Value Score" in results.columns else "Not available"

    high_value_low_unique = ""
    if has_ga4 and "GA4 Sessions" in results.columns and "Recommendation" in results.columns:
        risky = results[
            (results["Recommendation"].isin(["Consolidate / 301", "Review overlap"])) &
            (pd.to_numeric(results["GA4 Sessions"], errors="coerce").fillna(0) > 0)
        ].sort_values("GA4 Sessions", ascending=False).head(10)
        if len(risky):
            high_value_low_unique = risky[["URL", "GA4 Sessions", "Recommendation", "Most Similar URL"]].to_string(index=False)

    prompt = f"""You are a senior SEO consultant with 20 years of experience in site migrations.

You have run a semantic embedding analysis on {site_name or "a client website"} using Screaming Frog crawl data{"," + " Semrush organic data," if has_semrush else ""}{" and GA4 traffic data" if has_ga4 else ""}.

Analysis mode: {mode}
Total pages analysed: {total}
Recommendation breakdown: {json.dumps(rec_counts)}
Data sources active: SF metadata{"+ Semrush" if has_semrush else ""}{"+ GA4" if has_ga4 else ""}

Page Value Score statistics (0-100):
{value_stats}

TOP MIGRATE CANDIDATES (semantically unique, highest value):
{top_migrate}

TOP CONSOLIDATION TARGETS (near-duplicate pages, similarity >= 0.85):
{top_consolidate}

REVIEW OVERLAP PAGES (similarity 0.70-0.84):
{top_review}

{"HIGH TRAFFIC PAGES FLAGGED FOR CONSOLIDATION (risk alert):" + chr(10) + high_value_low_unique if high_value_low_unique else ""}

Write a professional migration analysis report with the following sections:

1. SITE OVERVIEW
   Summarise what the semantic analysis reveals about the site's content structure. Be specific about the numbers.

2. KEY FINDINGS
   3-5 bullet points covering the most important patterns found. Call out structural problems plainly.

3. HIGH-RISK PAGES
   Pages that carry real traffic or value but are flagged for consolidation. These need human review before any 301 is applied. Be direct about the risk.

4. CONSOLIDATION PRIORITIES
   Top duplicate clusters to action first. Group by pattern where possible (e.g. date-based news releases, paginated category pages).

5. PAGES TO PROTECT
   The highest-value unique pages. Explain why they matter and what happens if they are dropped or broken in migration.

6. PRIORITISED ACTION LIST
   Numbered list, ordered by impact. Specific and actionable. No vague recommendations.

Tone: direct, plain English, no marketing language. Write as if briefing a technical project manager who will execute the migration.
"""
    return prompt

def call_claude(prompt: str) -> str:
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"Content-Type": "application/json"},
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    data = response.json()
    if "content" in data:
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
    return f"API error: {data.get('error', {}).get('message', str(data))}"

# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("SF Semantic Analyzer")
st.caption("Screaming Frog · Semrush · GA4 · Semantic embeddings · Page value · AI migration report")

tab1, tab2, tab3 = st.tabs(["Uniqueness Audit", "Redirect Mapping", "AI Migration Report"])

# ── TAB 1: UNIQUENESS AUDIT ───────────────────────────────────────────────────

with tab1:
    st.subheader("Uniqueness Audit")
    st.markdown("Upload a Screaming Frog internal HTML export. Enrich with Semrush and GA4 for a full page value score.")

    with st.expander("SF columns used"):
        st.markdown("""
**Required:** `Address`, `Status Code`, `Indexability`

**Embedded:** `Title 1`, `H1-1`, `H2-1`, `H2-2`, `Meta Description 1`

**Scoring signals:** `Word Count`, `Inlinks`, `Crawl Depth 1`

**Optional body text (SF custom extraction):** `Custom Extraction 1` or `Body Text`
        """)

    with st.expander("Semrush — Pages export"):
        st.markdown("""
**Semrush > Organic Research > Pages > Export**

Columns used: `URL`, `Top keyword`, `Traffic`, `Number of Keywords`

The top keyword is appended to the embedding composite.
        """)

    with st.expander("Semrush — Positions export"):
        st.markdown("""
**Semrush > Organic Research > Positions > Export**

Columns used: `URL`, `Keyword`, `Search Volume`, `Position`

The app groups by URL, takes the top 5 keywords by search volume, and appends them as a keyword string to the composite. This gives the model the full semantic footprint of the page, not just the top keyword.
        """)

    with st.expander("GA4 — Landing pages export"):
        st.markdown("""
**GA4 > Engagement > Landing pages > Export (top right icon)**

Columns used: `Landing page` or `Page path`, `Sessions`, `Engagement rate`

Works with full URLs or path-only exports.
        """)

    sf_file   = st.file_uploader("Screaming Frog CSV (required)",          type="csv", key="sf1")
    sr_p_file = st.file_uploader("Semrush Pages CSV (optional)",           type="csv", key="srp1")
    sr_k_file = st.file_uploader("Semrush Positions CSV (optional)",       type="csv", key="srk1")
    ga4_file  = st.file_uploader("GA4 Landing Pages CSV (optional)",       type="csv", key="ga1")

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

        if sr_p_file:
            df = merge_sr_pages(df, pd.read_csv(sr_p_file))
            sources.append("Semrush Pages")

        if sr_k_file:
            df = merge_sr_positions(df, pd.read_csv(sr_k_file))
            sources.append("Semrush Positions")

        if ga4_file:
            df = merge_ga4(df, pd.read_csv(ga4_file))
            sources.append("GA4")

        st.info(f"Active sources: {', '.join(sources)}")

        embed_extra = []
        if "_sr_top_kw"    in df.columns: embed_extra.append("_sr_top_kw")
        if "_sr_kw_string" in df.columns: embed_extra.append("_sr_kw_string")

        df["Composite Text"] = df.apply(lambda r: build_composite(r, embed_extra), axis=1)

        with st.expander("Preview composite text (first 10 rows)"):
            st.dataframe(df[["Address", "Composite Text"]].head(10), use_container_width=True)

        if st.button("Run Uniqueness Audit", type="primary", key="run1"):
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
                "Recommendation":   [migration_rec(s) for s in max_sim],
            }

            for col, label in [
                ("Word Count",         "SF Word Count"),
                ("Inlinks",            "SF Inlinks"),
                ("Crawl Depth 1",      "SF Crawl Depth"),
                ("_sr_top_kw",         "Semrush Top Keyword"),
                ("_sr_kw_string",      "Semrush Top 5 Keywords"),
                ("_sr_traffic",        "Semrush Traffic"),
                ("_sr_kw_count",       "Semrush Keyword Count"),
                ("_sr_pos_kw_count",   "Semrush Positions KW Count"),
                ("_sr_total_vol",      "Semrush Total Search Volume"),
                ("_ga4_sessions",      "GA4 Sessions"),
                ("_ga4_engagement",    "GA4 Engagement Rate"),
            ]:
                if col in df.columns:
                    result_cols[label] = df[col].values

            results = pd.DataFrame(result_cols)
            st.session_state["audit_results"] = results

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

# ── TAB 2: REDIRECT MAPPING ───────────────────────────────────────────────────

with tab2:
    st.subheader("Redirect Mapping")
    st.markdown("Upload old and new site SF exports. Enrich old site with Semrush and GA4 to prioritise which URLs matter most.")

    with st.expander("Column requirements"):
        st.markdown("""
Same SF columns as Uniqueness Audit for both exports.

Semrush and GA4 apply to the **old site only**. They signal which old URLs carried real traffic and need careful mapping.
        """)

    left, right = st.columns(2)
    old_sf   = left.file_uploader("Old Site SF CSV",                  type="csv", key="old_sf")
    new_sf   = right.file_uploader("New Site SF CSV",                 type="csv", key="new_sf")
    old_srp  = left.file_uploader("Old Site Semrush Pages CSV",       type="csv", key="old_srp")
    old_srk  = right.file_uploader("Old Site Semrush Positions CSV",  type="csv", key="old_srk")
    old_ga4  = left.file_uploader("Old Site GA4 CSV",                 type="csv", key="old_ga4")

    top_n         = st.slider("Top N matches per URL", 1, 5, 3, key="topn2")
    sim_threshold = st.slider("Minimum similarity score", 0.0, 1.0, 0.0, 0.05, key="thresh2")

    if old_sf and new_sf:
        df_old = filter_df(pd.read_csv(old_sf))
        df_new = filter_df(pd.read_csv(new_sf))

        o1, o2 = st.columns(2)
        o1.metric("Old site pages (filtered)", len(df_old))
        o2.metric("New site pages (filtered)", len(df_new))

        if len(df_old) == 0 or len(df_new) == 0:
            st.error("One or both exports returned no pages after filtering.")
            st.stop()

        sources = ["SF metadata"]
        if old_srp:
            df_old = merge_sr_pages(df_old, pd.read_csv(old_srp))
            sources.append("Semrush Pages (old)")
        if old_srk:
            df_old = merge_sr_positions(df_old, pd.read_csv(old_srk))
            sources.append("Semrush Positions (old)")
        if old_ga4:
            df_old = merge_ga4(df_old, pd.read_csv(old_ga4))
            sources.append("GA4 (old)")

        st.info(f"Active sources: {', '.join(sources)}")

        embed_extra = []
        if "_sr_top_kw"    in df_old.columns: embed_extra.append("_sr_top_kw")
        if "_sr_kw_string" in df_old.columns: embed_extra.append("_sr_kw_string")

        df_old["Composite Text"] = df_old.apply(lambda r: build_composite(r, embed_extra), axis=1)
        df_new["Composite Text"] = df_new.apply(build_composite, axis=1)

        if st.button("Run Redirect Mapping", type="primary", key="run2"):
            emb_old = get_embeddings(model, df_old["Composite Text"].tolist())
            emb_new = get_embeddings(model, df_new["Composite Text"].tolist())

            with st.spinner("Calculating cross-site similarity..."):
                sim  = cosine_similarity(emb_old, emb_new)
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
                            ("_sr_top_kw",    "Old Semrush Top KW"),
                            ("_sr_kw_string", "Old Semrush Top 5 KWs"),
                            ("_sr_traffic",   "Old Semrush Traffic"),
                            ("_ga4_sessions", "Old GA4 Sessions"),
                            ("_ga4_engagement","Old GA4 Engagement"),
                        ]:
                            if col in df_old.columns:
                                row[label] = df_old[col].iloc[i]
                        rows.append(row)

                results = pd.DataFrame(rows)
                st.session_state["redirect_results"] = results

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

# ── TAB 3: AI MIGRATION REPORT ────────────────────────────────────────────────

with tab3:
    st.subheader("AI Migration Report")
    st.markdown(
        "Generates a written migration analysis from your audit or redirect mapping results. "
        "Run Tab 1 or Tab 2 first, then come here to generate the report."
    )

    site_name = st.text_input("Site name or domain (for the report header)", placeholder="e.g. gosemo.com")

    mode = st.radio(
        "Which results to report on?",
        ["Uniqueness Audit (Tab 1)", "Redirect Mapping (Tab 2)"],
        horizontal=True
    )

    has_audit    = "audit_results"    in st.session_state
    has_redirect = "redirect_results" in st.session_state

    if mode == "Uniqueness Audit (Tab 1)" and not has_audit:
        st.warning("No audit results found. Run the Uniqueness Audit in Tab 1 first.")
    elif mode == "Redirect Mapping (Tab 2)" and not has_redirect:
        st.warning("No redirect mapping results found. Run the Redirect Mapping in Tab 2 first.")
    else:
        results = (
            st.session_state["audit_results"]
            if mode == "Uniqueness Audit (Tab 1)"
            else st.session_state["redirect_results"]
        )
        mode_label = "Uniqueness Audit" if "Audit" in mode else "Redirect Mapping"

        st.info(f"Ready to report on {len(results)} rows from {mode_label}.")

        if st.button("Generate AI Report", type="primary", key="run3"):
            with st.spinner("Generating report..."):
                prompt = build_report_prompt(results, mode_label, site_name)
                report = call_claude(prompt)

            st.markdown("---")
            st.markdown(report)
            st.markdown("---")

            st.download_button(
                "Download Report as TXT",
                data=report,
                file_name=f"migration_report_{(site_name or 'site').replace('.','_')}.txt",
                mime="text/plain",
            )
