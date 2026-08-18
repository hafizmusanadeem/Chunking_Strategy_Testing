# Chunking strategy benchmark

This project now has a local benchmark that needs no vector database, API key, or LLM. It compares retrieval quality for four chunking strategies using TF-IDF and a labelled question set.

## Run the included example

```powershell
.\.venv\Scripts\python.exe benchmark_chunking.py --dataset data\sample_dataset.json
```

The ranked CSV is written to `results/chunking_benchmark.csv`. Choose the highest `recall@5`; use `mrr` as the tie-breaker, then lower latency.

## Evaluate your data

Create `data/my_dataset.json` using this shape:

```json
{
  "documents": [{"id": "unique-id", "content": "document text"}],
  "queries": [{"query": "user question", "answer": "exact supporting phrase", "document_id": "unique-id"}]
}
```

Use 20–50 realistic questions at minimum. The `answer` must be a short phrase copied exactly from the supporting text; this tells the benchmark which retrieved chunks are relevant. Then run:

```powershell
.\.venv\Scripts\python.exe benchmark_chunking.py --dataset data\my_dataset.json --sizes 128 256 512 --top-k 5
```

For code, Markdown, or manuals, pay particular attention to the `markdown` and `recursive` rows. For plain prose, start with `recursive` and `sentence`.
