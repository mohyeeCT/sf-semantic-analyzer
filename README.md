# SF Semantic Analyzer

Screaming Frog export analysis using sentence embeddings. Assesses page value and maps redirects without GSC data.

## What it does

**Uniqueness Audit (single site)**
- Uploads a Screaming Frog internal HTML export
- Filters to 200-status, indexable pages
- Builds a composite text string per page: `Title | H1 | Meta Description | URL slug tokens`
- Generates embeddings via `all-MiniLM-L6-v2`
- Calculates cosine similarity across all pages
- Outputs uniqueness score, most similar URL, and migration recommendation per page

**Redirect Mapping (old site vs new site)**
- Uploads two SF exports (old and new site)
- Generates embeddings for both
- Cross-matches each old URL to its top N semantic matches on the new site
- Outputs similarity score and confidence label per match

## Migration recommendations

| Max Similarity | Label |
|---|---|
| >= 0.85 | Consolidate / 301 |
| 0.70 – 0.84 | Review overlap |
| < 0.70 | Migrate |

## Required columns (Screaming Frog default export)

- `Address`
- `Status Code`
- `Indexability`
- `Title 1`
- `H1-1`
- `Meta Description 1`

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app > select this repo > `app.py` as entry point
4. Deploy

Note: First deploy downloads the `all-MiniLM-L6-v2` model (~25MB). Subsequent runs use the cached version.

## Planned additions

- Semrush / DataForSEO keyword data integration into the composite text
- Cluster visualisation (2D UMAP plot)
- Content gap detection across old vs new site
