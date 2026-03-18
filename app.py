import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import re
import requests
import time
import concurrent.futures
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="SF Semantic Analyzer",
    page_icon="SF",
    layout="wide"
)

# ── Model registry ─────────────────────────────────────────────────────────────

MODELS = {
    "all-MiniLM-L6-v2 - Fast (default)": {
        "id":   "all-MiniLM-L6-v2",
        "dims": 384,
        "type": "local",
        "note": "Best for quick audits. Good general accuracy.",
    },
    "all-mpnet-base-v2 - Balanced": {
        "id":   "all-mpnet-base-v2",
        "dims": 768,
        "type": "local",
        "note": "2x the vector dimensions. Better at distinguishing near-similar pages. Recommended for most migrations.",
    },
    "mxbai-embed-large-v1 - Best local quality": {
        "id":   "mixedbread-ai/mxbai-embed-large-v1",
        "dims": 1024,
        "type": "local",
        "note": "Highest local accuracy. Slower on large crawls. Best for sites with thin metadata.",
    },
    "text-embedding-3-small - OpenAI (API key required)": {
        "id":   "text-embedding-3-small",
        "dims": 1536,
        "type": "openai",
        "note": "Best overall accuracy. Requires OpenAI API key. Cost is negligible (~$0.00002 per 1k URLs).",
    },
}

# ── SF column groups ───────────────────────────────────────────────────────────

SF_EMBED_COLS = ["Title 1","H1-1","H2-1","H2-2","H3-1","Meta Description 1","Meta Keywords 1"]
SF_SCORE_COLS = ["Word Count","Inlinks","Unique Inlinks","Crawl Depth 1","Readability"]
SF_BODY_COLS  = ["Custom Extraction 1","Custom Extraction 2","Custom Extraction 3","Body Text","Body"]

INLINKS_DST_COL    = "Destination"
INLINKS_ANCHOR_COL = "Anchor Text"
INLINKS_SRC_TITLE  = "Source Title"
INLINKS_TYPE_COL   = "Type"

GENERIC_ANCHORS = {"nan","","n/a","click here","read more","here","this","link","learn more","more"}

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
    return "" if s.lower() in ("nan","n/a","none","") else s

def build_composite(row: pd.Series) -> str:
    parts = []
    for col in SF_EMBED_COLS:
        v = clean_val(row.get(col, ""))
        if v: parts.append(v)
    for col in SF_BODY_COLS:
        v = clean_val(row.get(col, ""))
        if v:
            parts.append(v[:400])
            break
    # Scraped body text (added by scrape step)
    scraped = clean_val(row.get("_scraped_body", ""))
    if scraped:
        parts.append(scraped[:800])
    slug = extract_slug_tokens(clean_val(row.get("Address", "")))
    if slug: parts.append(slug)
    anchors = clean_val(row.get("_anchor_string", ""))
    if anchors: parts.append(anchors)
    src_titles = clean_val(row.get("_src_title_string", ""))
    if src_titles: parts.append(src_titles)
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
    w_unique  = 0.50 + (0.20*(not has_wc)) + (0.15*(not has_il)) + (0.15*(not has_depth))
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
    all_embeddings = []
    batch_size = 500
    progress = st.progress(0, text="Generating embeddings via OpenAI...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model_id, "input": batch},
        )
        data = response.json()
        if "error" in data:
            st.error(f"OpenAI API error: {data['error']['message']}")
            st.stop()
        batch_emb = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        all_embeddings.extend(batch_emb)
        progress.progress(
            min((i + batch_size) / len(texts), 1.0),
            text=f"Embedding {min(i+batch_size, len(texts))} / {len(texts)} pages..."
        )
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
        il_a = il[~il[INLINKS_ANCHOR_COL].str.lower().isin(GENERIC_ANCHORS)]
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
            il[~il[INLINKS_SRC_TITLE].str.lower().isin(["nan","","n/a"])]
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

# ── Body scraping ──────────────────────────────────────────────────────────────

SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SEO-Analyzer/1.0; +https://github.com/mohyeeCT/sf-semantic-analyzer)"
}

BOILERPLATE_SELECTORS = [
    "nav","header","footer","[role='navigation']","[role='banner']",
    "[role='contentinfo']",".nav",".header",".footer",".menu",
    ".sidebar",".cookie","#cookie","script","style","noscript",
]

def extract_body_native(url: str, timeout: int = 10) -> str:
    """Fetch URL and extract clean body text using BeautifulSoup."""
    try:
        r = requests.get(url, headers=SCRAPE_HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for sel in BOILERPLATE_SELECTORS:
            for tag in soup.select(sel):
                tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|main|body", re.I))
        target = main if main else soup.body
        if not target:
            return ""
        text = " ".join(target.get_text(" ", strip=True).split())
        return text[:1200]
    except Exception:
        return ""

def extract_body_firecrawl(url: str, api_key: str, timeout: int = 30) -> str:
    """Scrape URL via Firecrawl API and return clean markdown text."""
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            timeout=timeout,
        )
        data = r.json()
        if not data.get("success"):
            return ""
        md = data.get("data", {}).get("markdown", "")
        # Strip markdown formatting for embedding
        md = re.sub(r"#{1,6}\s*", "", md)
        md = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", md)
        md = re.sub(r"[*_`~]", "", md)
        text = " ".join(md.split())
        return text[:1200]
    except Exception:
        return ""

def scrape_urls(urls: list, method: str, firecrawl_key: str = None,
                max_workers: int = 5, delay: float = 0.5) -> dict:
    """
    Scrape a list of URLs and return {url: body_text}.
    method: 'native' or 'firecrawl'
    """
    results = {}
    progress = st.progress(0, text="Starting scrape...")
    total = len(urls)

    def fetch(url):
        if method == "firecrawl":
            text = extract_body_firecrawl(url, firecrawl_key)
        else:
            text = extract_body_native(url)
        time.sleep(delay)
        return url, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, url): url for url in urls}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            url, text = future.result()
            results[url] = text
            done += 1
            progress.progress(
                done / total,
                text=f"Scraped {done} / {total} pages..."
            )

    progress.empty()
    return results

# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("SF Semantic Analyzer")

# ── Model selector ─────────────────────────────────────────────────────────────

st.subheader("Embedding Model")
model_label = st.selectbox(
    "Select model",
    options=list(MODELS.keys()),
    index=0,
)
model_cfg = MODELS[model_label]
st.caption(f"Dimensions: {model_cfg['dims']} | {model_cfg['note']}")

openai_key = None
if model_cfg["type"] == "openai":
    openai_key = st.text_input(
        "OpenAI API key", type="password", placeholder="sk-...",
        help="Used only for this session. Never stored."
    )
    if not openai_key:
        st.warning("Enter your OpenAI API key to use this model.")

st.divider()

# ── Body scraping ──────────────────────────────────────────────────────────────

st.subheader("Body Text Scraping")
st.markdown(
    "Fetch page body text to enrich embeddings. "
    "This is what Screaming Frog does before sending content to Gemini. "
    "Skip this if your SF export already includes Custom Extraction columns."
)

scrape_mode = st.radio(
    "Scraping method",
    ["Off - use SF export only", "Native HTTP - free, no JS rendering", "Firecrawl - JS rendering, cleaner extraction (API key required)"],
    index=0,
    horizontal=True,
)

firecrawl_key  = None
scrape_workers = 5
scrape_delay   = 0.5

if scrape_mode != "Off - use SF export only":
    col1, col2 = st.columns(2)
    scrape_workers = col1.slider("Concurrent requests", 1, 10, 5,
        help="Higher = faster but more load on the target server. Keep at 3-5 for client sites.")
    scrape_delay = col2.slider("Delay between requests (seconds)", 0.2, 3.0, 0.5, step=0.1,
        help="Adds a pause between each request. Increase for sensitive or rate-limited sites.")

    if "Firecrawl" in scrape_mode:
        firecrawl_key = st.text_input(
            "Firecrawl API key", type="password", placeholder="fc-...",
            help="Used only for this session. Never stored. Get a key at firecrawl.dev."
        )
        if not firecrawl_key:
            st.warning("Enter your Firecrawl API key to use this method.")
        else:
            st.info("Firecrawl renders JavaScript and strips boilerplate. Best match for SF + Gemini quality.")
    else:
        st.info("Native HTTP fetches raw HTML and strips nav/footer/sidebar. Fast and free. Does not render JavaScript.")

st.divider()

# ── File uploads ───────────────────────────────────────────────────────────────

st.subheader("Uniqueness Audit")

with st.expander("SF Internal HTML - columns used"):
    st.markdown("""
**Required:** `Address`, `Status Code`, `Indexability`

**Embedded:** `Title 1`, `H1-1`, `H2-1`, `H2-2`, `H3-1`, `Meta Description 1`, `Meta Keywords 1`

**Body text (SF Custom Extraction):** `Custom Extraction 1/2/3` - if present, body scraping above is redundant.

**Scoring signals (not embedded):** `Word Count`, `Inlinks`, `Unique Inlinks`, `Crawl Depth 1`, `Readability`
    """)

with st.expander("SF All Inlinks export (required)"):
    st.markdown("""
**How to export:** Bulk Exports > All Inlinks > CSV

**Columns used:** `Destination`, `Anchor Text`, `Source Title`, `Type`

Required because anchor text from inlinking pages is the strongest semantic signal SF can provide. Without it, pages that share the same title and H1 template look identical to the model even if they cover completely different content. The app aggregates top 8 anchor texts and top 5 source page titles per URL and appends them to the composite.
    """)

sf_file = st.file_uploader("Screaming Frog Internal HTML CSV (required)", type="csv", key="sf1")
il_file = st.file_uploader("SF All Inlinks CSV (required)", type="csv", key="il1")

if sf_file:
    raw = pd.read_csv(sf_file)
    df  = filter_df(raw)

    c1, c2 = st.columns(2)
    c1.metric("Total in export", len(raw))
    c2.metric("After filtering (200 + Indexable)", len(df))

    if len(df) == 0:
        st.error("No pages remain after filtering. Check Status Code and Indexability columns.")
        st.stop()

    if not il_file:
        st.error(
            "All Inlinks CSV is required. Export it from Screaming Frog via "
            "Bulk Exports > All Inlinks > CSV and upload it above."
        )
        st.stop()

    sources = ["SF metadata"]
    inlinks_df = pd.read_csv(il_file, low_memory=False)
    df = merge_inlinks(df, inlinks_df)
    sources.append("SF inlinks")

    # ── Body scraping step ─────────────────────────────────────────────────────
    body_col_exists = any(c in df.columns for c in SF_BODY_COLS)
    run_scrape = scrape_mode != "Off - use SF export only"

    if run_scrape and body_col_exists:
        st.warning(
            "SF export already contains body text columns (Custom Extraction). "
            "Scraping is redundant. Disable it or remove the extraction columns to avoid duplicate signal."
        )
        run_scrape = False

    if run_scrape and "Firecrawl" in scrape_mode and not firecrawl_key:
        st.warning("Firecrawl API key required. Add it above or switch to Native HTTP.")
        run_scrape = False

    if run_scrape and model_cfg["type"] == "openai" and not openai_key:
        run_scrape = False  # Will be blocked at run button anyway

    scraped_body = {}
    if run_scrape:
        urls_to_scrape = df["Address"].tolist()
        method = "firecrawl" if "Firecrawl" in scrape_mode else "native"
        st.info(
            f"Will scrape {len(urls_to_scrape)} URLs using {method} "
            f"({scrape_workers} concurrent, {scrape_delay}s delay). "
            f"Estimated time: {len(urls_to_scrape) * scrape_delay / scrape_workers / 60:.1f} min."
        )
        if st.button("Scrape Pages", key="scrape_btn"):
            scraped_body = scrape_urls(
                urls_to_scrape, method=method,
                firecrawl_key=firecrawl_key,
                max_workers=scrape_workers,
                delay=scrape_delay,
            )
            st.session_state["scraped_body"] = scraped_body
            fetched = sum(1 for v in scraped_body.values() if v)
            st.success(f"Scraped {fetched} / {len(urls_to_scrape)} pages successfully.")
            sources.append(f"Body text ({method})")

    # Restore scraped body from session if already done
    if "scraped_body" in st.session_state and not scraped_body:
        scraped_body = st.session_state["scraped_body"]

    if scraped_body:
        df["_scraped_body"] = df["Address"].map(scraped_body).fillna("")

    st.info(f"Active sources: {', '.join(sources)}  |  Model: {model_cfg['id']}")

    found_embed = [c for c in SF_EMBED_COLS if c in df.columns]
    found_score = [c for c in SF_SCORE_COLS if c in df.columns]
    found_body  = next((c for c in SF_BODY_COLS if c in df.columns), None)
    has_anchors = "_anchor_string" in df.columns
    has_scraped = "_scraped_body" in df.columns and df["_scraped_body"].str.len().sum() > 0

    col_info = f"Embedding: {', '.join(found_embed)}"
    if found_body:   col_info += f", {found_body}"
    if has_scraped:  col_info += ", scraped body text"
    if has_anchors:  col_info += ", anchor texts, source titles"
    if found_score:  col_info += f" | Scoring: {', '.join(found_score)}"
    st.caption(f"Columns detected: {col_info}")

    df["Composite Text"] = df.apply(build_composite, axis=1)

    with st.expander("Preview composite text (first 10 rows)"):
        st.dataframe(df[["Address", "Composite Text"]].head(10), use_container_width=True)

    # ── Run button ─────────────────────────────────────────────────────────────
    run_blocked = model_cfg["type"] == "openai" and not openai_key
    if run_blocked:
        st.button("Run Uniqueness Audit", type="primary", key="run1", disabled=True)
        st.caption("Add your OpenAI API key above to run.")
    elif st.button("Run Uniqueness Audit", type="primary", key="run1"):

        texts = df["Composite Text"].tolist()

        if model_cfg["type"] == "openai":
            embeddings = get_openai_embeddings(texts, openai_key, model_cfg["id"])
        else:
            local_model = load_local_model(model_cfg["id"])
            embeddings  = get_local_embeddings(local_model, texts)

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
            "Body Source":      ("firecrawl" if "Firecrawl" in scrape_mode else "native") if has_scraped else "SF export only",
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
