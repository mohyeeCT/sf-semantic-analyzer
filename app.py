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

# ── SF column groups ───────────────────────────────────────────────────────────

# All used for embedding composite — ordered by signal strength
SF_EMBED_COLS = [
    "Title 1",
    "H1-1",
    "H2-1", "H2-2",
    "H3-1",
    "Meta Description 1",
    "Meta Keywords 1",
]

# Scoring signals — numeric, not embedded
SF_SCORE_COLS = [
    "Word Count",
    "Inlinks",
    "Unique Inlinks",
    "Crawl Depth 1",
    "Readability",
]

# Custom body text extraction columns SF can export
SF_BODY_COLS = [
    "Custom Extraction 1",
    "Custom Extraction 2",
    "Custom Extraction 3",
    "Body Text",
    "Body",
]

# All Inlinks export columns
INLINKS_SRC_COL    = "Source"
INLINKS_DST_COL    = "Destination"
INLINKS_ANCHOR_COL = "Anchor Text"
INLINKS_SRC_TITLE  = "Source Title"
INLINKS_TYPE_COL   = "Type"
INLINKS_FOLLOW_COL = "Follow"

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
    """
    Build the richest possible embedding string from SF data alone.
    Order: title → headings → meta → body snippet → slug → inlink anchors
    """
    parts = []

    # Core metadata fields
    for col in SF_EMBED_COLS:
        v = clean_val(row.get(col, ""))
        if v:
            parts.append(v)

    # Body text from SF custom extraction (truncated)
    for col in SF_BODY_COLS:
        v = clean_val(row.get(col, ""))
        if v:
            parts.append(v[:400])
            break

    # URL slug tokens
    slug = extract_slug_tokens(clean_val(row.get("Address", "")))
    if slug:
        parts.append(slug)

    # Inlink anchor text aggregation (added by merge_inlinks)
    anchors = clean_val(row.get("_anchor_string", ""))
    if anchors:
        parts.append(anchors)

    # Source page titles of inlinking pages (added by merge_inlinks)
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
    """
    Page value from SF signals only.
      50% semantic uniqueness
      20% content richness (word count)
      15% structural importance (inlinks)
      15% crawl depth (inverted — shallower = more important)
    Weights redistribute if columns are absent.
    """
    idx = df.index

    u = normalise(pd.Series(uniqueness, index=idx))

    has_wc    = "Word Count"   in df.columns
    has_il    = "Inlinks"      in df.columns
    has_depth = "Crawl Depth 1" in df.columns

    w_unique = 0.50 + (0.20 * (not has_wc)) + (0.15 * (not has_il)) + (0.15 * (not has_depth))

    total = w_unique * u

    if has_wc:
        wc = pd.to_numeric(df["Word Count"], errors="coerce").fillna(0)
        total += 0.20 * normalise(wc)

    if has_il:
        il = pd.to_numeric(df["Inlinks"], errors="coerce").fillna(0)
        total += 0.15 * normalise(il)

    if has_depth:
        depth = pd.to_numeric(df["Crawl Depth 1"], errors="coerce").fillna(99)
        # Invert: depth 1 is most important
        total += 0.15 * normalise(1 / (depth + 1))

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

# ── SF inlinks merge ───────────────────────────────────────────────────────────

def merge_inlinks(df: pd.DataFrame, inlinks: pd.DataFrame) -> pd.DataFrame:
    """
    From the SF All Inlinks export:
    - Group by Destination URL
    - Collect unique anchor texts (top 8 by frequency)
    - Collect unique source page titles (top 5)
    - Append both as strings to the main dataframe
    """
    required = [INLINKS_DST_COL]
    if not all(c in inlinks.columns for c in required):
        st.warning("Inlinks file: no Destination column found. Skipping.")
        return df

    il = inlinks.copy()
    il[INLINKS_DST_COL] = il[INLINKS_DST_COL].astype(str).str.strip()

    # Filter to internal HTML links only if Type column is present
    if INLINKS_TYPE_COL in il.columns:
        il = il[il[INLINKS_TYPE_COL].str.lower().str.contains("html|hyperlink", na=False)]

    # Aggregate anchor texts
    if INLINKS_ANCHOR_COL in il.columns:
        il[INLINKS_ANCHOR_COL] = il[INLINKS_ANCHOR_COL].astype(str).str.strip()
        il_anchors = il[il[INLINKS_ANCHOR_COL].str.lower().notin(["nan", "", "n/a", "click here", "read more", "here", "this", "link"])
                       if hasattr(il[INLINKS_ANCHOR_COL].str.lower(), "notin")
                       else ~il[INLINKS_ANCHOR_COL].str.lower().isin(["nan", "", "n/a", "click here", "read more", "here", "this", "link"])]

        anchor_agg = (
            il_anchors.groupby(INLINKS_DST_COL)[INLINKS_ANCHOR_COL]
            .apply(lambda x: " | ".join(
                x.value_counts().head(8).index.tolist()
            ))
            .reset_index()
            .rename(columns={INLINKS_DST_COL: "Address", INLINKS_ANCHOR_COL: "_anchor_string"})
        )
        df = df.merge(anchor_agg, on="Address", how="left")

    # Aggregate source page titles
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

    # Inlink count per destination
    inlink_count = (
        il.groupby(INLINKS_DST_COL)
        .size()
        .reset_index(name="_inlink_count")
        .rename(columns={INLINKS_DST_COL: "Address"})
    )
    df = df.merge(inlink_count, on="Address", how="left")
    df["_inlink_count"] = df["_inlink_count"].fillna(0).astype(int)

    return df

# ── AI report ──────────────────────────────────────────────────────────────────

def build_report_prompt(results: pd.DataFrame, mode: str, site_name: str) -> str:
    total      = len(results)
    rec_counts = results["Recommendation"].value_counts().to_dict() if "Recommendation" in results.columns else {}

    migrate_df     = results[results["Recommendation"] == "Migrate"].sort_values("Page Value Score", ascending=False) if "Recommendation" in results.columns else pd.DataFrame()
    consolidate_df = results[results["Recommendation"] == "Consolidate / 301"].sort_values("Max Similarity", ascending=False) if "Recommendation" in results.columns else pd.DataFrame()
    review_df      = results[results["Recommendation"] == "Review overlap"] if "Recommendation" in results.columns else pd.DataFrame()

    def safe_table(df, cols):
        available = [c for c in cols if c in df.columns]
        return df[available].head(15).to_string(index=False) if len(df) else "None"

    top_migrate     = safe_table(migrate_df,     ["URL", "Page Value Score", "Uniqueness Score", "SF Word Count", "SF Inlinks"])
    top_consolidate = safe_table(consolidate_df, ["URL", "Max Similarity", "Most Similar URL", "SF Inlinks"])
    top_review      = safe_table(review_df,      ["URL", "Max Similarity", "Most Similar URL"])

    value_stats = results["Page Value Score"].describe().round(2).to_string() if "Page Value Score" in results.columns else "Not available"

    has_inlinks  = "SF Inlinks"  in results.columns
    has_wc       = "SF Word Count" in results.columns
    has_anchors  = "Anchor Texts" in results.columns

    sources_note = "Screaming Frog crawl data only (metadata, headings, URL structure"
    if has_inlinks:  sources_note += ", internal link graph"
    if has_anchors:  sources_note += ", inlink anchor texts"
    sources_note += ")"

    prompt = f"""You are a senior SEO consultant with 20 years of experience in site migrations.

You have run a semantic embedding analysis on {site_name or "a client website"} using {sources_note}.

Analysis mode: {mode}
Total pages analysed: {total}
Recommendation breakdown: {json.dumps(rec_counts)}

Page Value Score statistics (0-100, derived from semantic uniqueness, word count, inlinks, crawl depth):
{value_stats}

TOP MIGRATE CANDIDATES (semantically unique pages — highest value, do not drop):
{top_migrate}

TOP CONSOLIDATION TARGETS (near-duplicate pages, similarity >= 0.85):
{top_consolidate}

REVIEW OVERLAP PAGES (similarity 0.70–0.84, need human review):
{top_review}

Write a professional migration analysis report with the following sections:

1. SITE OVERVIEW
   What the semantic analysis reveals about the content structure. Be specific about the numbers and patterns.

2. KEY FINDINGS
   3-5 bullet points. Call out structural problems plainly. No vague language.

3. HIGH-RISK CONSOLIDATION TARGETS
   Pages flagged for 301 that show signals of importance (high inlinks, shallow crawl depth, high word count). These need human review before any redirect is applied.

4. CONSOLIDATION PRIORITIES
   Top duplicate clusters to action first. Group by content pattern where possible.

5. PAGES TO PROTECT
   Highest-value unique pages. What makes them distinct. What happens if they are dropped or broken.

6. PRIORITISED ACTION LIST
   Numbered, ordered by impact. Specific and executable. No generic recommendations.

Tone: direct, plain English, no marketing language. Write for a technical project manager who will execute the migration.
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
st.caption("Screaming Frog only · Maximum SF signal · Semantic embeddings · Page value · AI migration report")

tab1, tab2, tab3 = st.tabs(["Uniqueness Audit", "Redirect Mapping", "AI Migration Report"])

# ── SHARED EXPANDERS ──────────────────────────────────────────────────────────

def expander_sf_columns():
    with st.expander("SF Internal HTML — columns used"):
        st.markdown("""
**Required:** `Address`, `Status Code`, `Indexability`

**Embedded into composite text:**
`Title 1`, `H1-1`, `H2-1`, `H2-2`, `H3-1`, `Meta Description 1`, `Meta Keywords 1`

**Body text (if you ran Custom Extraction in SF):**
`Custom Extraction 1`, `Custom Extraction 2`, `Custom Extraction 3`

**Scoring signals (not embedded):**
`Word Count`, `Inlinks`, `Unique Inlinks`, `Crawl Depth 1`, `Readability`

Use the default internal HTML bulk export. No column renaming needed.
        """)

def expander_inlinks():
    with st.expander("SF All Inlinks export (optional but recommended)"):
        st.markdown("""
**How to export:** Bulk Exports > All Inlinks > CSV

**Columns used:** `Destination`, `Anchor Text`, `Source Title`, `Type`

**What it adds:** The anchor text of inlinking pages is one of the strongest semantic signals available in SF without external data. It tells the model what the rest of the site *calls* each page. For pages with thin metadata (Title = H1, empty meta description), this transforms the embedding quality.

The app aggregates top 8 anchor texts and top 5 source page titles per destination URL and appends them to the composite.
        """)

# ── TAB 1: UNIQUENESS AUDIT ───────────────────────────────────────────────────

with tab1:
    st.subheader("Uniqueness Audit")
    st.markdown("Single site analysis. Scores each page for semantic uniqueness and migration value using SF data only.")

    expander_sf_columns()
    expander_inlinks()

    sf_file = st.file_uploader(
        "Screaming Frog Internal HTML CSV (required)",
        type="csv", key="sf1"
    )
    il_file = st.file_uploader(
        "SF All Inlinks CSV (optional — strongly recommended)",
        type="csv", key="il1"
    )

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

        st.info(f"Active sources: {', '.join(sources)}")

        # Show which SF columns were found
        found_embed = [c for c in SF_EMBED_COLS if c in df.columns]
        found_score = [c for c in SF_SCORE_COLS if c in df.columns]
        found_body  = next((c for c in SF_BODY_COLS if c in df.columns), None)
        has_anchors = "_anchor_string" in df.columns

        col_info = f"Embedding: {', '.join(found_embed)}"
        if found_body:   col_info += f", {found_body}"
        if has_anchors:  col_info += ", anchor texts, source titles"
        if found_score:  col_info += f" | Scoring: {', '.join(found_score)}"

        st.caption(f"Columns detected — {col_info}")

        df["Composite Text"] = df.apply(build_composite, axis=1)

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
                ("Word Count",        "SF Word Count"),
                ("Inlinks",           "SF Inlinks"),
                ("Unique Inlinks",    "SF Unique Inlinks"),
                ("Crawl Depth 1",     "SF Crawl Depth"),
                ("Readability",       "SF Readability"),
                ("_inlink_count",     "Inlinks (from inlinks export)"),
                ("_anchor_string",    "Anchor Texts"),
                ("_src_title_string", "Source Page Titles"),
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

            st.dataframe(
                results.sort_values("Page Value Score", ascending=False),
                use_container_width=True
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
    st.markdown("Cross-site analysis. Maps each old URL to its best semantic match on the new site.")

    expander_sf_columns()
    expander_inlinks()

    with st.expander("Inlinks for redirect mapping"):
        st.markdown("""
Upload the **old site** inlinks export. This enriches the old URL embeddings with anchor context.

The new site inlinks export is optional — useful if the new site is already partially built and crawled.
        """)

    left, right = st.columns(2)
    old_sf = left.file_uploader("Old Site SF CSV (required)",  type="csv", key="old_sf")
    new_sf = right.file_uploader("New Site SF CSV (required)", type="csv", key="new_sf")
    old_il = left.file_uploader("Old Site Inlinks CSV (optional)", type="csv", key="old_il")
    new_il = right.file_uploader("New Site Inlinks CSV (optional)", type="csv", key="new_il")

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

        if old_il:
            df_old = merge_inlinks(df_old, pd.read_csv(old_il, low_memory=False))
            sources.append("Old site inlinks")
        if new_il:
            df_new = merge_inlinks(df_new, pd.read_csv(new_il, low_memory=False))
            sources.append("New site inlinks")

        st.info(f"Active sources: {', '.join(sources)}")

        df_old["Composite Text"] = df_old.apply(build_composite, axis=1)
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
                            ("Word Count",        "Old SF Word Count"),
                            ("Inlinks",           "Old SF Inlinks"),
                            ("Crawl Depth 1",     "Old SF Crawl Depth"),
                            ("_inlink_count",     "Old Inlinks (inlinks export)"),
                            ("_anchor_string",    "Old Anchor Texts"),
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

            sort_col = "Old SF Inlinks" if "Old SF Inlinks" in results.columns else "Similarity Score"
            st.dataframe(
                results.sort_values(sort_col, ascending=False),
                use_container_width=True
            )

            st.download_button(
                "Download Redirect Map CSV",
                data=results.to_csv(index=False),
                file_name="redirect_mapping.csv",
                mime="text/csv",
            )

# ── TAB 3: AI MIGRATION REPORT ────────────────────────────────────────────────

with tab3:
    st.subheader("AI Migration Report")
    st.markdown(
        "Generates a written migration analysis from your audit or redirect mapping results. "
        "Run Tab 1 or Tab 2 first, then generate here."
    )

    site_name = st.text_input(
        "Site name or domain",
        placeholder="e.g. gosemo.com"
    )

    mode = st.radio(
        "Report on:",
        ["Uniqueness Audit (Tab 1)", "Redirect Mapping (Tab 2)"],
        horizontal=True
    )

    has_audit    = "audit_results"    in st.session_state
    has_redirect = "redirect_results" in st.session_state

    if mode == "Uniqueness Audit (Tab 1)" and not has_audit:
        st.warning("No audit results found. Run the Uniqueness Audit in Tab 1 first.")
    elif mode == "Redirect Mapping (Tab 2)" and not has_redirect:
        st.warning("No redirect mapping results found. Run Redirect Mapping in Tab 2 first.")
    else:
        results    = st.session_state["audit_results"] if "Audit" in mode else st.session_state["redirect_results"]
        mode_label = "Uniqueness Audit" if "Audit" in mode else "Redirect Mapping"

        st.info(f"Ready: {len(results)} rows from {mode_label}.")

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
                file_name=f"migration_report_{(site_name or 'site').replace('.', '_')}.txt",
                mime="text/plain",
            )
