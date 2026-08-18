# RAG chunking benchmark

This uses the actual RAG path: documents → chunking → Jina embeddings → Qdrant vector storage → query retrieval → metrics. Temporary local Qdrant storage is deleted after the run.

Create `data/my_dataset.json`:

```json
{"documents":[{"id":"policy-01","type":"markdown","content":"full text"}],"queries":[{"query":"What is the leave allowance?","evidence":"Employees receive 20 paid leave days.","document_id":"policy-01"}]}
```

`evidence` must be an exact supporting source passage. This labels relevance without relying on chunk indices, which vary by strategy. Use 20–50 representative queries.

```powershell
$env:JINA_API_KEY = "your-key"
.\.venv\Scripts\python.exe rag_chunking_benchmark.py --dataset data\my_dataset.json
```

Choose the strongest `recall@5`, then `mrr`; manually inspect the winning chunks before production use.
