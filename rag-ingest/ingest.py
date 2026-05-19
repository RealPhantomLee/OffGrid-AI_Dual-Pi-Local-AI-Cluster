#!/usr/bin/env python3
"""
RAG Ingest Script
Ingests knowledge datasets into ChromaDB from Wikipedia, arXiv, and local docs.
Usage: python3 ingest.py [--source wikipedia|arxiv|local|all]
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import chromadb
import wikipedia
import arxiv
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_PATH = "/home/merry/chromadb/"
LOG_DIR = "/home/merry/rag-logs/"
DOCS_DIR = "/home/merry/Documents/"
COLLECTION_NAME = "knowledge_base"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
WIKI_ARTICLE_COUNT = 50
ARXIV_PAPER_COUNT = 100
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, datetime.now().strftime("%Y-%m-%d") + ".log")

    logger = logging.getLogger("rag-ingest")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    text = text.strip()
    if not text:
        return chunks
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def make_id(source: str, index: int, content: str) -> str:
    """Generate a deterministic ID based on source + content hash."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{source}_{index}_{digest}"


# ---------------------------------------------------------------------------
# Embedding + ChromaDB helpers
# ---------------------------------------------------------------------------
def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    source_tag: str,
    doc_index: int,
    text: str,
    metadata_base: dict,
) -> int:
    """Chunk text, embed, and upsert into ChromaDB. Returns number of chunks added."""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        chunk_id = make_id(source_tag, doc_index * 1000 + i, chunk)
        ids.append(chunk_id)
        documents.append(chunk)
        meta = {**metadata_base, "chunk_index": i, "total_chunks": len(chunks)}
        metadatas.append(meta)

    embeddings = model.encode(documents, show_progress_bar=False).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# Source: Wikipedia
# ---------------------------------------------------------------------------
def ingest_wikipedia(collection: chromadb.Collection, model: SentenceTransformer) -> int:
    logger.info("=== Ingesting Wikipedia featured articles ===")
    wikipedia.set_lang("en")

    featured_titles = wikipedia.random(WIKI_ARTICLE_COUNT)
    if isinstance(featured_titles, str):
        featured_titles = [featured_titles]

    total_chunks = 0
    for idx, title in enumerate(featured_titles):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            # Summary + first 2000 chars of full content
            raw = page.summary + "\n\n" + page.content[:2000]
            meta = {
                "source": "wikipedia",
                "title": page.title,
                "url": page.url,
                "ingested_at": datetime.now().isoformat(),
            }
            added = upsert_chunks(collection, model, "wiki", idx, raw, meta)
            total_chunks += added
            logger.debug(f"  [{idx+1}/{WIKI_ARTICLE_COUNT}] '{page.title}' -> {added} chunks")
        except wikipedia.exceptions.DisambiguationError as e:
            # Take the first option
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                raw = page.summary + "\n\n" + page.content[:2000]
                meta = {
                    "source": "wikipedia",
                    "title": page.title,
                    "url": page.url,
                    "ingested_at": datetime.now().isoformat(),
                }
                added = upsert_chunks(collection, model, "wiki", idx, raw, meta)
                total_chunks += added
            except Exception as inner_e:
                logger.warning(f"  Skipping '{title}' (disambiguation fallback failed): {inner_e}")
        except Exception as e:
            logger.warning(f"  Skipping '{title}': {e}")
        time.sleep(0.2)  # be polite to the API

    logger.info(f"Wikipedia: {total_chunks} total chunks upserted.")
    return total_chunks


# ---------------------------------------------------------------------------
# Source: arXiv
# ---------------------------------------------------------------------------
def ingest_arxiv(collection: chromadb.Collection, model: SentenceTransformer) -> int:
    logger.info("=== Ingesting arXiv abstracts ===")
    client = arxiv.Client(page_size=100, delay_seconds=1.0, num_retries=3)

    papers_per_cat = ARXIV_PAPER_COUNT // len(ARXIV_CATEGORIES)
    total_chunks = 0
    global_idx = 0

    for cat in ARXIV_CATEGORIES:
        search = arxiv.Search(
            query=f"cat:{cat}",
            max_results=papers_per_cat,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        try:
            results = list(client.results(search))
        except Exception as e:
            logger.warning(f"  arXiv query failed for {cat}: {e}")
            continue

        for paper in results:
            text = f"Title: {paper.title}\n\nAuthors: {', '.join(a.name for a in paper.authors)}\n\nAbstract: {paper.summary}"
            meta = {
                "source": "arxiv",
                "title": paper.title,
                "arxiv_id": paper.get_short_id(),
                "category": cat,
                "published": paper.published.isoformat() if paper.published else "",
                "url": paper.entry_id,
                "ingested_at": datetime.now().isoformat(),
            }
            added = upsert_chunks(collection, model, "arxiv", global_idx, text, meta)
            total_chunks += added
            logger.debug(f"  [{global_idx+1}] '{paper.title[:60]}' -> {added} chunks")
            global_idx += 1

    logger.info(f"arXiv: {total_chunks} total chunks upserted.")
    return total_chunks


# ---------------------------------------------------------------------------
# Source: Local documents
# ---------------------------------------------------------------------------
def iter_local_files(docs_dir: str) -> Generator[Path, None, None]:
    base = Path(docs_dir)
    if not base.exists():
        logger.warning(f"Documents directory not found: {docs_dir}")
        return
    for pattern in ("**/*.txt", "**/*.md"):
        yield from base.glob(pattern)


def ingest_local(collection: chromadb.Collection, model: SentenceTransformer) -> int:
    logger.info("=== Ingesting local documents ===")
    total_chunks = 0
    files = list(iter_local_files(DOCS_DIR))

    if not files:
        logger.info("No .txt or .md files found in Documents directory.")
        return 0

    for idx, fpath in enumerate(files):
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                logger.debug(f"  Skipping empty file: {fpath}")
                continue
            meta = {
                "source": "local",
                "filename": fpath.name,
                "filepath": str(fpath),
                "extension": fpath.suffix,
                "ingested_at": datetime.now().isoformat(),
            }
            added = upsert_chunks(collection, model, "local", idx, text, meta)
            total_chunks += added
            logger.debug(f"  [{idx+1}/{len(files)}] '{fpath.name}' -> {added} chunks")
        except Exception as e:
            logger.warning(f"  Failed to read '{fpath}': {e}")

    logger.info(f"Local docs: {total_chunks} total chunks upserted from {len(files)} files.")
    return total_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ingest knowledge sources into ChromaDB.")
    parser.add_argument(
        "--source",
        choices=["wikipedia", "arxiv", "local", "all"],
        default="all",
        help="Which source to ingest (default: all)",
    )
    args = parser.parse_args()

    logger.info(f"Starting RAG ingest — source: {args.source}")
    start_time = time.time()

    # Init ChromaDB
    os.makedirs(CHROMA_PATH, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = get_collection(chroma_client)
    logger.info(f"Connected to ChromaDB at {CHROMA_PATH}, collection '{COLLECTION_NAME}'")

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    logger.info("Model loaded.")

    total = 0

    if args.source in ("wikipedia", "all"):
        total += ingest_wikipedia(collection, model)

    if args.source in ("arxiv", "all"):
        total += ingest_arxiv(collection, model)

    if args.source in ("local", "all"):
        total += ingest_local(collection, model)

    elapsed = time.time() - start_time
    collection_count = collection.count()
    logger.info(
        f"Ingest complete in {elapsed:.1f}s. "
        f"Session chunks: {total}. "
        f"Total in collection: {collection_count}."
    )


if __name__ == "__main__":
    main()
