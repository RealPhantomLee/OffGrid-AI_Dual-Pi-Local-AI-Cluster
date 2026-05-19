#!/usr/bin/env bash
# Daily RAG ingestion runner — executed by systemd timer at 2 AM
set -euo pipefail

LOG_DIR="/home/<YOUR_USER>/rag-logs"
DATE="$(date +%Y-%m-%d)"
LOG_FILE="${LOG_DIR}/${DATE}.log"
ERR_FILE="${LOG_DIR}/errors.log"

mkdir -p "${LOG_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting RAG ingestion" >> "${LOG_FILE}"

if cd /home/<YOUR_USER>/rag-ingest && /home/<YOUR_USER>/venv/bin/python3 ingest.py --source all >> "${LOG_FILE}" 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ingestion completed OK" >> "${LOG_FILE}"

    # Quick ChromaDB sanity check
    /home/<YOUR_USER>/venv/bin/python3 - >> "${LOG_FILE}" 2>&1 << 'EOF'
import chromadb, sys
client = chromadb.PersistentClient(path="/home/<YOUR_USER>/chromadb")
try:
    col = client.get_collection("knowledge_base")
    count = col.count()
    results = col.query(query_texts=["artificial intelligence"], n_results=1)
    print(f"[ChromaDB] Collection has {count} documents. Test query OK.")
except Exception as e:
    print(f"[ChromaDB] WARNING: {e}", file=sys.stderr)
    sys.exit(1)
EOF

else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INGEST FAILED — see above" >> "${LOG_FILE}"
    echo "=== ${DATE} FAILURE ===" >> "${ERR_FILE}"
    tail -30 "${LOG_FILE}" >> "${ERR_FILE}"
    echo "" >> "${ERR_FILE}"
    exit 1
fi
