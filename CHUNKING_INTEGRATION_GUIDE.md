# Chunking Strategy Integration Guide for RAG-Advanced-System

**Target Metrics:** 85%+ Citation Precision | 80%+ Faithfulness | P95 <2s Latency

**Your Stack:** LangChain + Qdrant + Jina Embeddings + LangSmith + NeMo Guardrails

---

## Phase 1: Testing Framework (Week 1-2)

### Step 1: Setup & Installation

```bash
# Install required packages
pip install langchain langchain-text-splitters langchain-community langsmith qdrant-client jina-embeddings-v2 pandas scikit-learn

# Verify Qdrant is running (Docker)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
```

### Step 2: Create Your Test Dataset

Create a file `data/test_documents.json`:

```json
{
  "documents": [
    {
      "id": "doc_001",
      "type": "general",
      "content": "Your sample document text here...",
      "queries": [
        {
          "query": "What is the main topic?",
          "relevant_chunk_indices": [0, 1, 2]
        }
      ]
    },
    {
      "id": "doc_002",
      "type": "code",
      "content": "def example():\n    pass",
      "queries": []
    }
  ]
}
```

### Step 3: Run Baseline Test

```python
from chunking_test_framework import ChunkingTestHarness, ChunkingConfig

# Initialize harness
harness = ChunkingTestHarness(
    qdrant_url="http://localhost:6333",
    embeddings_model="jina"
)

# Define baseline strategy
baseline = ChunkingConfig(
    name="Recursive-768-Baseline",
    chunk_size=768,
    chunk_overlap=100,
    strategy="recursive"
)

# Test on first document
sample_text = "Your document content..."
metrics = harness.test_strategy(
    config=baseline,
    sample_texts=[sample_text],
    query="Your test query?",
    relevant_chunk_indices=[0, 1]
)

print(f"✅ Baseline Precision@5: {metrics.retrieval_precision:.2%}")
print(f"✅ Citation Precision: {metrics.citation_precision:.2%}")
```

**Expected Output:**
```
==============================================================
Strategy: Recursive-768-Baseline
==============================================================
Chunks: 4 | Avg Size: 156.3 words
Precision@5: 60.00%
Faithfulness: 75.00%
Citation Precision: 70.00%
MRR@10: 50.00%
Latency: 245.32ms
Retrieved chunks: 5
```

---

## Phase 2: Comparative Testing (Week 2-3)

### Step 4: Run Full Comparison

```python
from chunking_test_framework import ChunkingConfig, ChunkingTestHarness

# Define all strategies to compare
strategies = [
    # Tier 1: Production-ready
    ChunkingConfig(
        name="Recursive-512-Tight",
        chunk_size=512,
        chunk_overlap=50,
        strategy="recursive"
    ),
    ChunkingConfig(
        name="Recursive-768-Standard",
        chunk_size=768,
        chunk_overlap=100,
        strategy="recursive"
    ),
    ChunkingConfig(
        name="Recursive-1024-Loose",
        chunk_size=1024,
        chunk_overlap=150,
        strategy="recursive"
    ),
    
    # Tier 2: Semantic (experimental)
    ChunkingConfig(
        name="Semantic-768-Aggressive",
        chunk_size=768,
        chunk_overlap=100,
        strategy="semantic",
        threshold=0.3  # Lower threshold = more splits
    ),
    ChunkingConfig(
        name="Semantic-768-Conservative",
        chunk_size=768,
        chunk_overlap=100,
        strategy="semantic",
        threshold=0.7  # Higher threshold = fewer splits
    ),
    
    # Tier 3: Baseline (should perform worst)
    ChunkingConfig(
        name="Character-768-Baseline",
        chunk_size=768,
        chunk_overlap=100,
        strategy="character"
    ),
]

# Load your test documents
import json
with open("data/test_documents.json") as f:
    test_data = json.load(f)

# Initialize harness
harness = ChunkingTestHarness()

# Run comparative test
print("\n🚀 Running Comparative Test Across All Strategies...")
results_df = harness.run_comparative_test(
    configs=strategies,
    sample_texts=[doc["content"] for doc in test_data["documents"]],
    test_queries=[
        (query["query"], query["relevant_chunk_indices"])
        for doc in test_data["documents"]
        for query in doc.get("queries", [])
    ]
)

# Display results
print("\n" + "="*120)
print("COMPARATIVE TEST RESULTS")
print("="*120)
print(results_df.to_string())

# Save results
results_df.to_csv("chunking_test_results.csv", index=False)

# Identify winner
print("\n" + "="*120)
print("ANALYSIS")
print("="*120)

# Sort by citation precision (your primary metric)
top_strategy = results_df.loc[results_df["citation_precision"].idxmax()]
print(f"\n🏆 Winner: {top_strategy['strategy_name']}")
print(f"   Citation Precision: {top_strategy['citation_precision']:.2%} ✓ {'PASS' if top_strategy['citation_precision'] >= 0.85 else 'FAIL'}")
print(f"   Faithfulness: {top_strategy['faithfulness_score']:.2%} ✓ {'PASS' if top_strategy['faithfulness_score'] >= 0.80 else 'FAIL'}")
print(f"   MRR@10: {top_strategy['mean_reciprocal_rank']:.2%}")
print(f"   Avg Chunk Size: {top_strategy['avg_chunk_size']:.1f} words")
print(f"   Num Chunks: {top_strategy['num_chunks']}")
print(f"   Latency: {top_strategy['avg_latency_ms']:.2f}ms")
```

### Step 5: Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("chunking_test_results.csv")

# Key insights
print("\n📊 KEY INSIGHTS:")
print("\n1. Citation Precision by Strategy:")
print(df[['strategy_name', 'citation_precision']].sort_values('citation_precision', ascending=False))

print("\n2. Faithfulness vs Latency Trade-off:")
print(df[['strategy_name', 'faithfulness_score', 'avg_latency_ms']].sort_values('faithfulness_score', ascending=False))

print("\n3. Chunk Statistics:")
print(df[['strategy_name', 'num_chunks', 'avg_chunk_size']].sort_values('num_chunks'))

# Visualization: Citation Precision Comparison
plt.figure(figsize=(12, 6))
df_sorted = df.sort_values('citation_precision', ascending=True)
plt.barh(df_sorted['strategy_name'], df_sorted['citation_precision'])
plt.axvline(x=0.85, color='green', linestyle='--', label='Target (85%)')
plt.xlabel('Citation Precision')
plt.title('Chunking Strategy Comparison: Citation Precision')
plt.legend()
plt.tight_layout()
plt.savefig('citation_precision_comparison.png')

# Visualization: Latency vs Precision
plt.figure(figsize=(12, 6))
plt.scatter(df['avg_latency_ms'], df['citation_precision'], s=200, alpha=0.6)
for idx, row in df.iterrows():
    plt.annotate(row['strategy_name'], 
                (row['avg_latency_ms'], row['citation_precision']),
                fontsize=8)
plt.axhline(y=0.85, color='green', linestyle='--', label='Citation Precision Target')
plt.axvline(x=2000, color='orange', linestyle='--', label='Latency Target (P95)')
plt.xlabel('Average Latency (ms)')
plt.ylabel('Citation Precision')
plt.title('Latency vs Citation Precision Trade-off')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('latency_vs_precision.png')
```

---

## Phase 3: Integration into Production (Week 3-4)

### Step 6: Implement Chosen Strategy in Your RAG Pipeline

**Before:** (Static chunking)
```python
# Old approach
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=768,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
```

**After:** (Dynamic, intelligent chunking)

Create `src/rag/chunking.py`:

```python
"""
Production Chunking Module
Integrates tested strategy into RAG pipeline
"""

from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)


@traceable(name="intelligent_chunking")
def intelligent_chunk_document(
    document: Document,
    strategy: str = "recursive",
    **kwargs
) -> List[Document]:
    """
    Intelligent chunking that adapts to document type.
    
    Args:
        document: Document to chunk
        strategy: One of: recursive, semantic, language_aware
        **kwargs: Additional parameters for the strategy
    
    Returns:
        List of chunked documents with metadata
    
    Example:
        >>> doc = Document(page_content="...", metadata={"type": "code"})
        >>> chunks = intelligent_chunk_document(doc, strategy="language_aware")
    """
    
    doc_type = document.metadata.get("type", "general")
    
    # Route to appropriate strategy
    if doc_type in ["code", "python", "javascript"]:
        return chunk_language_aware(document, **kwargs)
    elif doc_type in ["finance", "legal"]:
        return chunk_proposition_based(document, **kwargs)
    elif len(document.page_content.split()) > 5000:
        return chunk_semantic(document, **kwargs)
    else:
        return chunk_recursive(document, **kwargs)


def chunk_recursive(
    document: Document,
    chunk_size: int = 768,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Tier 1: Recursive chunking (your production default)
    Best for: General documents, mixed content
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = splitter.split_documents([document])
    
    # Add metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunking_strategy": "recursive",
            "chunk_id": i,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        })
    
    logger.info(f"Recursive chunking: {len(chunks)} chunks from {document.metadata.get('source', 'unknown')}")
    return chunks


def chunk_semantic(
    document: Document,
    chunk_size: int = 768,
    chunk_overlap: int = 100,
    threshold: float = 0.5
) -> List[Document]:
    """
    Tier 2: Semantic chunking with embedding-based splitting
    Best for: Large mixed-topic documents
    Adds ~100-200ms latency per document
    """
    from langchain_community.embeddings import JinaEmbeddings
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Step 1: Initial recursive split
    initial_chunks = chunk_recursive(document, chunk_size, chunk_overlap)
    
    if len(initial_chunks) <= 1:
        return initial_chunks
    
    # Step 2: Embed chunks
    embeddings = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")
    chunk_texts = [c.page_content for c in initial_chunks]
    embedded = np.array(embeddings.embed_documents(chunk_texts))
    
    # Step 3: Calculate similarities
    similarities = []
    for i in range(len(embedded) - 1):
        sim = cosine_similarity([embedded[i]], [embedded[i + 1]])[0][0]
        similarities.append(sim)
    
    # Step 4: Find split points (where similarity drops)
    threshold_val = np.percentile(similarities, threshold * 100)
    
    # Step 5: Merge/split chunks based on similarity
    final_chunks = [initial_chunks[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold_val:
            final_chunks.append(initial_chunks[i + 1])
        else:
            # Merge with previous
            final_chunks[-1].page_content += " " + initial_chunks[i + 1].page_content
    
    # Add metadata
    for i, chunk in enumerate(final_chunks):
        chunk.metadata.update({
            "chunking_strategy": "semantic",
            "chunk_id": i,
        })
    
    logger.info(f"Semantic chunking: {len(initial_chunks)} -> {len(final_chunks)} chunks")
    return final_chunks


def chunk_language_aware(
    document: Document,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    language: str = "python"
) -> List[Document]:
    """
    Tier 3: Language-aware chunking for code/markdown
    Best for: Code, markdown, structured documents
    """
    try:
        lang_enum = Language[language.upper()]
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents([document])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunking_strategy": "language_aware",
                "language": language,
                "chunk_id": i,
            })
        
        logger.info(f"Language-aware chunking ({language}): {len(chunks)} chunks")
        return chunks
    except KeyError:
        logger.warning(f"Language {language} not supported, falling back to recursive")
        return chunk_recursive(document, chunk_size, chunk_overlap)


def chunk_proposition_based(
    document: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 75
) -> List[Document]:
    """
    Tier 4: Proposition-based chunking (experimental)
    Best for: Finance, legal, high-stakes retrieval
    Requires LLM inference - use sparingly
    """
    # For now, use recursive as fallback
    # Full implementation would use LLM for proposition extraction
    logger.warning("Proposition chunking not yet implemented, using recursive fallback")
    return chunk_recursive(document, chunk_size, chunk_overlap)


# ============================================================================
# INTEGRATION WITH YOUR EXISTING RAG PIPELINE
# ============================================================================

@traceable(name="load_and_chunk")
def load_and_chunk_document(
    file_path: str,
    doc_type: str = "general"
) -> List[Document]:
    """
    End-to-end: Load document and apply intelligent chunking.
    
    This replaces your existing document loading + chunking step.
    """
    from langchain_community.document_loaders import PyPDFLoader
    
    # Load document
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    else:
        # Implement other loaders as needed
        with open(file_path) as f:
            docs = [Document(page_content=f.read())]
    
    # Add document type metadata
    for doc in docs:
        doc.metadata["type"] = doc_type
    
    # Apply intelligent chunking
    all_chunks = []
    for doc in docs:
        chunks = intelligent_chunk_document(doc)
        all_chunks.extend(chunks)
    
    logger.info(f"Loaded and chunked {file_path}: {len(all_chunks)} chunks")
    return all_chunks
```

### Step 7: Update Your RAG Loader Pipeline

In your main RAG file (e.g., `src/rag/main.py`):

```python
from src.rag.chunking import load_and_chunk_document
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import JinaEmbeddings

# Initialize embeddings
embeddings = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")

# Load and chunk documents
documents = []
for file_path, doc_type in [
    ("data/docs/finance.pdf", "finance"),
    ("data/docs/code.py", "code"),
    ("data/docs/general.md", "general"),
]:
    chunks = load_and_chunk_document(file_path, doc_type=doc_type)
    documents.extend(chunks)

# Create vector store
vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    url="http://localhost:6333",
    collection_name="rag_advanced_system"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Top-5 retrieval
)
```

---

## Phase 4: Observability & Monitoring (Week 4+)

### Step 8: Track Chunking Metrics in LangSmith

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Create evaluation dataset
evaluation_dataset = client.create_dataset(
    dataset_name="chunking_quality_evals",
    description="Evaluate retrieval quality with different chunking strategies"
)

# Add test cases
evaluation_dataset.add_examples([
    {
        "inputs": {"query": "What is the main topic?"},
        "outputs": {"expected_chunks": [0, 1, 2]},
        "metadata": {"strategy": "recursive-768"}
    },
    # ... more test cases
])

# Run evaluation
results = evaluate(
    lambda inputs: retriever.get_relevant_documents(inputs["query"]),
    data=evaluation_dataset,
    evaluators=[
        lambda inputs, outputs: {
            "citation_precision": calculate_precision(outputs),
            "faithfulness": calculate_faithfulness(outputs),
        }
    ],
    experiment_prefix="chunking-comparison"
)
```

### Step 9: Monitor in Production

```python
from langsmith.run_helpers import get_current_run_tree
import logging

logger = logging.getLogger(__name__)

@traceable
def rag_query(query: str) -> str:
    """Main RAG query with chunking metrics"""
    
    # Get run context
    run_tree = get_current_run_tree()
    
    # Retrieve chunks
    chunks = retriever.invoke(query)
    
    # Log chunking metrics
    run_tree.metadata = {
        "num_chunks_retrieved": len(chunks),
        "avg_chunk_size": sum(len(c.page_content.split()) for c in chunks) / len(chunks),
        "chunking_strategies": [c.metadata.get("chunking_strategy") for c in chunks],
        "citation_precision": calculate_citation_precision(chunks),
    }
    
    # Generate answer
    answer = llm_chain.invoke({"context": chunks, "query": query})
    
    return answer
```

---

## Phase 5: Iterate & Optimize (Ongoing)

### Step 10: Weekly Metrics Review

Create `notebooks/chunking_metrics_dashboard.ipynb`:

```python
import pandas as pd
from datetime import datetime, timedelta

# Query LangSmith for metrics
runs = client.list_runs(
    project_name="rag-advanced-system",
    filter='metadata.citation_precision is not None'
)

# Aggregate by chunking strategy
metrics_by_strategy = {}
for run in runs:
    strategy = run.metadata.get("chunking_strategies", ["unknown"])[0]
    if strategy not in metrics_by_strategy:
        metrics_by_strategy[strategy] = []
    
    metrics_by_strategy[strategy].append({
        "precision": run.metadata.get("citation_precision"),
        "latency": run.latency_ms,
        "timestamp": run.created_at
    })

# Create report
df = pd.DataFrame([
    {
        "strategy": strategy,
        "avg_precision": np.mean([m["precision"] for m in metrics]),
        "avg_latency": np.mean([m["latency"] for m in metrics]),
        "count": len(metrics)
    }
    for strategy, metrics in metrics_by_strategy.items()
])

print(df.sort_values("avg_precision", ascending=False))

# Alert if metrics degrade
for strategy in df["strategy"]:
    recent_precision = df[df["strategy"] == strategy]["avg_precision"].iloc[0]
    if recent_precision < 0.85:
        logger.warning(f"⚠️ Strategy {strategy} precision dropped below 85%: {recent_precision:.2%}")
```

---

## Checklist: Integration Complete?

- [ ] **Testing Framework**: `chunking_test_framework.py` integrated
- [ ] **Baseline Metrics**: Recorded initial performance
- [ ] **Comparative Test**: Ran all strategies, identified winner
- [ ] **Production Code**: Integrated chosen strategy into RAG pipeline
- [ ] **LangSmith Tracking**: Chunking metrics logged to LangSmith
- [ ] **Documentation**: Team trained on new chunking approach
- [ ] **Fallback Plan**: Have backup strategy if primary fails

---

## Expected Results by Week 4

| Metric | Target | Achieved |
|--------|--------|----------|
| Citation Precision | 85%+ | TBD |
| Faithfulness | 80%+ | TBD |
| P95 Latency | <2s | TBD |
| Avg Chunk Size | 100-200 words | TBD |
| Num Chunks/Doc | 5-20 | TBD |

---

## Troubleshooting

### Problem: Citation Precision < 85%
**Solution**: 
- Reduce chunk size to 512 tokens
- Add semantic post-processing (Tier 2)
- Check if query-chunk semantic alignment is poor

### Problem: Latency > 2s
**Solution**:
- Switch to recursive (remove semantic post-processing)
- Batch embeddings via Portkey
- Use smaller Jina model or embeddings cache

### Problem: Too Many Chunks
**Solution**:
- Increase chunk_size (768 → 1024)
- Reduce chunk_overlap (100 → 50)
- Use language-aware splitting (less fragmentation)

---

## Questions?

- See `chunking_test_framework.py` for full API
- Check LangSmith logs for per-document traces
- Reference YouTube transcripts in `data/` for examples

Good luck! 🚀
