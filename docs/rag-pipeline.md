# RAG Pipeline: Ingestion and Retrieval

This document covers the full lifecycle of the RAG knowledge base: how documents are ingested from Wikipedia, arXiv, and local files; how the ChromaDB vector store is structured; the automated daily refresh schedule; and how to query the knowledge base directly or add your own documents.

Source: `/home/merry/rag-ingest/ingest.py`

---

## 1. Architecture

```
Wikipedia API ──┐
arXiv API      ──┤──► ingest.py ──► ChromaDB PersistentClient ──► /home/merry/chromadb/
~/Documents/   ──┘                           │
                                             │
                           chromadb.service (HTTP server, port 8001)
                                             │
                                    agent-hub /rag/query endpoint
```

**ChromaDB storage**: `chromadb.PersistentClient(path="/home/merry/chromadb/")` writes to disk at `/home/merry/chromadb/`. This is an embedded database — no separate ChromaDB server process is needed for ingest. The ingest script opens the database file directly.

**ChromaDB HTTP server**: `chromadb.service` runs a separate HTTP server on port 8001 that Agent Hub connects to via `chromadb.AsyncHttpClient`. This server reads from the same `/home/merry/chromadb/` directory.

**Collection names**: The ingest script creates and writes to a collection named **`knowledge_base`**. Agent Hub's RAG agent reads from a collection named **`agent_hub_knowledge`**. These are two different collections. To wire them together, update `CHROMA_COLLECTION_NAME` in `/home/merry/agent-hub/main.py` to `"knowledge_base"`, or update `COLLECTION_NAME` in ingest.py to `"agent_hub_knowledge"`.

Both the ingest collection and Agent Hub's collection are created with `{"hnsw:space": "cosine"}` metadata, meaning all nearest-neighbour searches use cosine similarity.

---

## 2. Embedding Model

| Property | Value |
|---|---|
| Model name | `all-MiniLM-L6-v2` |
| Source | sentence-transformers library |
| Vector dimensions | 384 |
| Runtime | CPU, within the ingest virtualenv |
| Loaded via | `SentenceTransformer("all-MiniLM-L6-v2")` |

The model is loaded once per ingest run (`model = SentenceTransformer(EMBED_MODEL)`) before any source is processed. Embeddings are computed in batches per chunk group using `model.encode(documents, show_progress_bar=False)`.

all-MiniLM-L6-v2 is a distilled model — only 22M parameters, ~80 MB on disk — which makes it practical for continuous CPU-side embedding on a Pi 5. The 384-dimensional output is a good trade-off between retrieval quality and storage size. At full DOTPROD throughput on Cortex-A76, expect **50–100 documents per minute** depending on average document length after chunking.

The model is downloaded automatically from HuggingFace on first run to the sentence-transformers cache (typically `~/.cache/huggingface/`). Subsequent runs use the cached weights.

---

## 3. Ingestion Sources

### 3a. Wikipedia

**Configured constants**:
- `WIKI_ARTICLE_COUNT = 50` — number of articles to fetch per run
- Language: English (`wikipedia.set_lang("en")`)

**Article selection**: Articles are selected using `wikipedia.random(WIKI_ARTICLE_COUNT)`, which returns a list of random Wikipedia article titles via the Wikipedia API's `list=random` action. The selection changes every run — there is no concept of "featured articles" or topical filtering. Titles are resolved to full page objects via `wikipedia.page(title, auto_suggest=False)`.

**Text extraction**: For each article, ingest uses:
```python
raw = page.summary + "\n\n" + page.content[:2000]
```
This concatenates the article's summary section with the first 2000 characters of the full article body. The full `page.content` can be very long; the 2000-character cap prevents any single article from dominating the collection with dozens of chunks.

**Metadata stored per chunk**:
```json
{
  "source": "wikipedia",
  "title": "<article title>",
  "url": "<article URL>",
  "ingested_at": "<ISO 8601 timestamp>"
}
```

**Rate limiting**: `time.sleep(0.2)` between articles — 200 ms pause to avoid hammering the Wikipedia API.

**Disambiguation handling**: If a random title is a disambiguation page, ingest automatically retries with `e.options[0]` (the first disambiguation option). If that also fails, the title is skipped with a warning log.

### 3b. arXiv

**Configured constants**:
- `ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"]`
- `ARXIV_PAPER_COUNT = 100` — total papers across all categories
- Papers per category: `100 // 3 = 33` (integer division)
- Sort: newest first (`SortCriterion.SubmittedDate`, `SortOrder.Descending`)

**Client configuration**:
```python
arxiv.Client(page_size=100, delay_seconds=1.0, num_retries=3)
```
1-second delay between API pages, 3 retries on failure.

**Text extracted**: Abstract only (no PDF download):
```python
text = f"Title: {paper.title}\n\nAuthors: {', '.join(a.name for a in paper.authors)}\n\nAbstract: {paper.summary}"
```
This produces a compact, information-dense representation. Full PDF text is not fetched — abstracts are sufficient for semantic retrieval of "is this paper relevant to my question?" queries.

**Metadata stored per chunk**:
```json
{
  "source": "arxiv",
  "title": "<paper title>",
  "arxiv_id": "<short ID, e.g. 2405.12345>",
  "category": "cs.AI",
  "published": "<ISO 8601 date>",
  "url": "<arXiv entry URL>",
  "ingested_at": "<ISO 8601 timestamp>"
}
```

### 3c. Local Documents

**Source directory**: `/home/merry/Documents/`

**File types**: `.txt` and `.md` files, discovered recursively using `Path.glob("**/*.txt")` and `Path.glob("**/*.md")`. Subdirectories are traversed.

**Text extraction**: Files are read as UTF-8 text with `errors="replace"` (invalid bytes are replaced with the Unicode replacement character rather than raising an exception). Empty files are skipped.

**Metadata stored per chunk**:
```json
{
  "source": "local",
  "filename": "<basename>",
  "filepath": "<absolute path>",
  "extension": ".txt or .md",
  "ingested_at": "<ISO 8601 timestamp>"
}
```

---

## 4. Chunking Strategy

Chunking is **character-level**, not token-level. The `chunk_text()` function:

```python
CHUNK_SIZE = 512      # characters
CHUNK_OVERLAP = 50    # characters

def chunk_text(text, chunk_size=512, overlap=50):
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap   # advance by (512 - 50) = 462 chars
    return chunks
```

Each chunk is 512 characters with 50 characters of overlap between consecutive chunks. A typical English sentence is 70–120 characters, so each chunk holds roughly 4–7 sentences. The overlap ensures that sentences split at a chunk boundary appear in both adjacent chunks, reducing the chance of losing context at retrieval boundaries.

**Chunk IDs** are deterministic. The `make_id()` function:
```python
def make_id(source: str, index: int, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{source}_{index}_{digest}"
```

The ID encodes source tag, document index, and a 16-character SHA-256 prefix of the chunk content. Because ChromaDB's `upsert` is used (not `add`), re-running ingest with the same content produces the same IDs and updates the existing records rather than duplicating them. Changed content produces a different hash, creating a new record — the old record with the same index position but different content will remain unless explicitly deleted.

---

## 5. Current Collection Stats

From the completed ingest run on 2026-05-19:

| Source | Chunks |
|---|---|
| Wikipedia | 58 |
| arXiv | 392 |
| Local documents | 38 |
| **Total** | **488** |

The arXiv source dominates because abstracts from 100 papers produce more total text than 50 truncated Wikipedia articles, and local documents are relatively sparse. The ratio will shift as more local files are added to `~/Documents/`.

---

## 6. Systemd Timer

The ingest pipeline runs automatically via two systemd units:

**`rag-ingest.timer`** — fires daily at 02:00 local time:
```ini
[Unit]
Description=Daily RAG ingest timer

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

`Persistent=true` means if the Pi was off at 02:00, the timer fires immediately at next boot (within a catchup window).

**`rag-ingest.service`** — `Type=oneshot`, runs ingest.py and exits:
```ini
[Unit]
Description=RAG knowledge base ingest
After=network.target chromadb.service

[Service]
Type=oneshot
User=merry
WorkingDirectory=/home/merry/rag-ingest
ExecStart=/home/merry/rag-ingest/venv/bin/python3 ingest.py --source all
StandardOutput=append:/home/merry/rag-logs/%Y-%m-%d.log
StandardError=append:/home/merry/rag-logs/%Y-%m-%d.log
```

**Log location**: `/home/merry/rag-logs/YYYY-MM-DD.log` — one log file per day. The log file is also written to from within ingest.py's own `setup_logging()` function, which creates `os.path.join(LOG_DIR, datetime.now().strftime("%Y-%m-%d") + ".log")`. Both the service unit and the Python logger write to the same file path.

**Check timer status:**
```bash
systemctl list-timers rag-ingest.timer
journalctl -u rag-ingest.service --since "1 day ago"
```

---

## 7. Adding Custom Documents

Drop `.txt` or `.md` files into `~/Documents/` — they will be picked up automatically on the next scheduled run (02:00) or manual trigger.

Subdirectories are supported. For example:
```
~/Documents/
    project-notes.md
    research/
        paper-summary.txt
        experiment-log.md
```

All three files are discovered via the recursive `**/*.txt` and `**/*.md` glob patterns.

**Manual trigger** (runs immediately, does not wait for 02:00):
```bash
sudo systemctl start rag-ingest.service
```

Watch the log in real time:
```bash
tail -f /home/merry/rag-logs/$(date +%Y-%m-%d).log
```

**Ingest a single source only:**
```bash
cd /home/merry/rag-ingest
source venv/bin/activate
python3 ingest.py --source local      # only ~/Documents/
python3 ingest.py --source wikipedia  # only Wikipedia
python3 ingest.py --source arxiv      # only arXiv
python3 ingest.py --source all        # everything (default)
```

Because `upsert` is used, re-running on already-ingested content is safe and idempotent — no duplicates are created.

---

## 8. Manual ChromaDB Query

### Python (direct PersistentClient)

```python
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="/home/merry/chromadb/")
collection = client.get_collection("knowledge_base")

model = SentenceTransformer("all-MiniLM-L6-v2")
query = "attention mechanisms in transformer models"
embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=embedding,
    n_results=5,
    include=["documents", "metadatas", "distances"],
)

for i, doc in enumerate(results["documents"][0]):
    meta = results["metadatas"][0][i]
    dist = results["distances"][0][i]
    print(f"--- Result {i+1} (distance={dist:.4f}) ---")
    print(f"Source: {meta.get('source')} | Title: {meta.get('title', meta.get('filename', '?'))}")
    print(doc[:300])
    print()
```

### Python (HTTP client — same interface as Agent Hub)

```python
import chromadb

async def query_http():
    client = await chromadb.AsyncHttpClient(host="localhost", port=8001)
    collection = await client.get_collection("knowledge_base")
    results = await collection.query(
        query_texts=["attention mechanisms in transformer models"],
        n_results=5,
        include=["documents", "metadatas"],
    )
    return results
```

### HTTP (raw API)

Check that ChromaDB is running:
```bash
curl http://localhost:8001/api/v2/heartbeat
# {"nanosecond heartbeat": 1716123456789012345}
```

List collections:
```bash
curl http://localhost:8001/api/v2/collections
```

Get collection document count:
```bash
curl "http://localhost:8001/api/v2/collections/knowledge_base/count"
```

Query (you must supply pre-computed embeddings via the HTTP API — ChromaDB's HTTP endpoint does not embed `query_texts` server-side unless you configure an embedding function on the server):
```bash
# Get the collection ID first
COLL_ID=$(curl -s "http://localhost:8001/api/v2/collections/knowledge_base" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# Query with pre-embedded vector (384 dimensions for all-MiniLM-L6-v2)
curl -s -X POST "http://localhost:8001/api/v2/collections/${COLL_ID}/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query_embeddings": [[0.01, 0.02, ...]],
    "n_results": 3,
    "include": ["documents", "metadatas"]
  }' | python3 -m json.tool
```

For most use cases, prefer the Python client which handles embedding automatically.

---

## 9. Scaling Recommendations

### Additional arXiv Categories

The current three categories (`cs.AI`, `cs.LG`, `cs.CL`) cover AI/ML theory and NLP. Extend the `ARXIV_CATEGORIES` list in ingest.py to cover more ground:

| Category | Scope |
|---|---|
| `cs.CV` | Computer vision, image processing |
| `cs.RO` | Robotics, control systems |
| `cs.NE` | Neural and evolutionary computing |
| `stat.ML` | Statistical machine learning |
| `cs.IR` | Information retrieval |
| `cs.HC` | Human-computer interaction |

Add them to the list and increase `ARXIV_PAPER_COUNT` proportionally. The papers-per-category is computed as `ARXIV_PAPER_COUNT // len(ARXIV_CATEGORIES)`, so adding 3 categories (total 6) and keeping `ARXIV_PAPER_COUNT=100` gives ~16 papers per category; increase to 200 to maintain ~33 per category.

### Growing the Collection

ChromaDB's HNSW index scales well to millions of vectors on a Pi 5 — the bottleneck is RAM for the in-memory index during search, not disk. At 384 dimensions and 4 bytes per float, each vector occupies 1.5 KB. 10,000 chunks ≈ 15 MB of vector data, well within available memory.

**Practical limits on a Pi 5 (8 GB)**:
- Up to ~50,000 chunks without impacting inference performance
- Above that, the HNSW index takes enough RAM (~75 MB for 50k × 384) to noticeably crowd the 8 GB

### Embedding Throughput

On a Raspberry Pi 5 (Cortex-A76, DOTPROD enabled), all-MiniLM-L6-v2 processes approximately:
- **50–100 documents per minute** for typical mixed-length text (Wikipedia summaries, arXiv abstracts)
- A full ingest run with 50 Wikipedia + 100 arXiv + local files typically completes in **8–15 minutes**
- Increasing `ARXIV_PAPER_COUNT` to 300 adds ~4–6 minutes to the nightly run

The ingest script uses `show_progress_bar=False` to keep logs clean in headless operation. Remove that flag during interactive testing to see per-batch progress.

### Improving Chunk Quality

The current 512-character chunks are coarse. For domain-specific document sets, consider:

1. **Semantic chunking**: split on double newlines or sentence boundaries rather than fixed character count. Requires `nltk` or `spacy`.
2. **Larger overlap**: increase `CHUNK_OVERLAP` from 50 to 100–150 characters for documents with dense cross-sentence dependencies.
3. **Document-type-specific strategies**: for arXiv abstracts (~1200 chars), the current chunk size results in 2–3 chunks per paper, which is appropriate. For long local documents, you may want larger `CHUNK_SIZE` (1024) to keep related content together.

Changes to `CHUNK_SIZE` or `CHUNK_OVERLAP` change the chunk content, which changes SHA-256 hashes, which changes all IDs. A re-ingest after changing these values will create new records alongside the old ones. To start clean: delete the collection and re-run.

```python
import chromadb
client = chromadb.PersistentClient(path="/home/merry/chromadb/")
client.delete_collection("knowledge_base")
```

Then run `python3 ingest.py --source all`.
