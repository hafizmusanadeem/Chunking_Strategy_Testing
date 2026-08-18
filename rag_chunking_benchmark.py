"""Compare RAG chunking strategies using Jina embeddings and Qdrant.

Dataset shape: {"documents": [{"id": "...", "content": "..."}],
"queries": [{"query": "...", "evidence": "exact source passage", "document_id": "..."}]}.
Evidence, rather than chunk IDs, keeps relevance labels fair as strategies create different chunks.
"""
from __future__ import annotations

import argparse, csv, json, os, re, shutil, time, uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import JinaEmbeddings
from langchain_text_splitters import Language
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Config:
    name: str; strategy: str; chunk_size: int; chunk_overlap: int


def splitter(config: Config, language: str | None):
    if config.strategy == "recursive":
        return RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    if config.strategy == "character":
        return CharacterTextSplitter(separator=" ", chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    if config.strategy == "language":
        try:
            return RecursiveCharacterTextSplitter.from_language(Language[(language or "MARKDOWN").upper()], chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        except KeyError:
            return RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    raise ValueError(config.strategy)


def semantic_chunks(text: str, config: Config, embeddings: JinaEmbeddings) -> list[Document]:
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    limiter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    if len(sentences) < 3:
        return limiter.create_documents([text])
    vectors = np.asarray(embeddings.embed_documents(sentences))
    similarities = [cosine_similarity([vectors[i]], [vectors[i + 1]])[0][0] for i in range(len(vectors) - 1)]
    threshold = float(np.percentile(similarities, 25))
    groups, group = [], [sentences[0]]
    for i, score in enumerate(similarities):
        group.append(sentences[i + 1])
        if score <= threshold:
            groups.append(" ".join(group)); group = []
    if group: groups.append(" ".join(group))
    return [chunk for item in groups for chunk in limiter.create_documents([item])]


def chunk_all(documents: list[dict], config: Config, embeddings: JinaEmbeddings) -> list[Document]:
    output = []
    for source in documents:
        chunks = semantic_chunks(source["content"], config, embeddings) if config.strategy == "semantic" else splitter(config, source.get("language") or source.get("type")).create_documents([source["content"]])
        for index, chunk in enumerate(chunks):
            chunk.metadata = {"document_id": source["id"], "chunk_index": index, "strategy": config.name}
            output.append(chunk)
    return output


def evaluate(config: Config, documents: list[dict], queries: list[dict], embeddings: JinaEmbeddings, db_path: Path, top_k: int) -> dict:
    started = time.perf_counter(); chunks = chunk_all(documents, config, embeddings)
    vectors = embeddings.embed_documents([x.page_content for x in chunks])
    client = QdrantClient(path=str(db_path)); collection = f"chunk_eval_{uuid.uuid4().hex}"
    client.create_collection(collection, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
    client.upsert(collection, points=[PointStruct(id=i, vector=v, payload=chunk.metadata) for i, (v, chunk) in enumerate(zip(vectors, chunks))], wait=True)
    hits, ranks, precisions = [], [], []
    for item in queries:
        points = client.query_points(collection, query=embeddings.embed_query(item["query"]), limit=top_k).points
        relevant = [rank for rank, point in enumerate(points, 1) if item["evidence"].casefold() in chunks[int(point.id)].page_content.casefold() and (not item.get("document_id") or chunks[int(point.id)].metadata["document_id"] == item["document_id"])]
        hits.append(bool(relevant)); ranks.append(1 / relevant[0] if relevant else 0); precisions.append(len(relevant) / top_k)
    client.delete_collection(collection); client.close()
    return {"strategy": config.name, "chunks": len(chunks), "avg_chunk_words": round(mean(len(x.page_content.split()) for x in chunks), 1), f"recall@{top_k}": round(mean(hits), 4), f"precision@{top_k}": round(mean(precisions), 4), "mrr": round(mean(ranks), 4), "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--top-k", type=int, default=5); parser.add_argument("--qdrant-path", type=Path, default=Path(".qdrant-benchmark")); parser.add_argument("--output", type=Path, default=Path("results/rag_chunking_benchmark.csv")); args = parser.parse_args()
    if not os.getenv("JINA_API_KEY"): raise SystemExit("Set JINA_API_KEY before running; use the same embedding model as your production RAG pipeline.")
    data = json.loads(args.dataset.read_text(encoding="utf-8")); embeddings = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")
    configs = [Config("recursive-512", "recursive", 512, 75), Config("recursive-768", "recursive", 768, 100), Config("character-768", "character", 768, 100), Config("semantic-768", "semantic", 768, 100), Config("language-768", "language", 768, 100)]
    rows = [evaluate(config, data["documents"], data["queries"], embeddings, args.qdrant_path, args.top_k) for config in configs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    shutil.rmtree(args.qdrant_path, ignore_errors=True)
    print("\n".join(str(row) for row in rows)); print(f"Saved: {args.output}")


if __name__ == "__main__": main()
