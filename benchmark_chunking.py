r"""Local, reproducible chunking benchmark.

Run:
    .\.venv\Scripts\python.exe benchmark_chunking.py --dataset data\sample_dataset.json

Dataset schema:
{
  "documents": [{"id": "policy", "content": "...", "type": "markdown"}],
  "queries": [{"query": "...", "answer": "exact supporting phrase", "document_id": "policy"}]
}

The answer field is the smallest exact passage that must be retrieved.  It gives
the benchmark an objective relevance label without depending on an LLM judge.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Chunk:
    text: str
    document_id: str
    strategy: str
    index: int


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def fixed_word_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = _words(text)
    step = max(1, size - overlap)
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step)]


def recursive_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Fill chunks from paragraphs, then sentences, then words; retain boundaries."""
    units = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_count = 0

    def flush() -> None:
        nonlocal current, current_count
        if current:
            chunks.append("\n\n".join(current))
            trailing = _words(chunks[-1])[-overlap:] if overlap else []
            current, current_count = ([" ".join(trailing)] if trailing else []), len(trailing)

    for paragraph in units:
        pieces = [paragraph]
        if len(_words(paragraph)) > size:
            pieces = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
        for piece in pieces:
            count = len(_words(piece))
            if count > size:
                flush()
                chunks.extend(fixed_word_chunks(piece, size, overlap))
            elif current and current_count + count > size:
                flush()
                current.append(piece)
                current_count += count
            else:
                current.append(piece)
                current_count += count
    flush()
    return chunks


def sentence_chunks(text: str, size: int, overlap: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return _group_units(sentences, size, overlap)


def markdown_chunks(text: str, size: int, overlap: int) -> list[str]:
    sections = re.split(r"(?m)(?=^#{1,6}\s+)", text)
    result: list[str] = []
    for section in sections:
        if section.strip():
            result.extend(recursive_chunks(section, size, overlap))
    return result


def _group_units(units: Iterable[str], size: int, overlap: int) -> list[str]:
    return recursive_chunks("\n\n".join(units), size, overlap)


STRATEGIES: dict[str, Callable[[str, int, int], list[str]]] = {
    "fixed_word": fixed_word_chunks,
    "recursive": recursive_chunks,
    "sentence": sentence_chunks,
    "markdown": markdown_chunks,
}


def make_chunks(documents: list[dict], strategy: str, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    splitter = STRATEGIES[strategy]
    for document in documents:
        for text in splitter(document["content"], size, overlap):
            if text.strip():
                chunks.append(Chunk(text, document["id"], strategy, len(chunks)))
    return chunks


def benchmark(documents: list[dict], queries: list[dict], strategy: str, size: int, overlap: int, top_k: int) -> dict:
    started = time.perf_counter()
    chunks = make_chunks(documents, strategy, size, overlap)
    if not chunks:
        raise ValueError("No chunks were created; check the documents in the dataset.")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    hits = reciprocal_rank = 0.0
    evaluated = 0
    for item in queries:
        answer = item.get("answer", "").strip().lower()
        if not answer:
            continue
        evaluated += 1
        scores = (matrix @ vectorizer.transform([item["query"]]).T).toarray().ravel()
        ranked = scores.argsort()[::-1][:top_k]
        relevant = [rank + 1 for rank, index in enumerate(ranked)
                    if answer in chunks[index].text.lower()
                    and (not item.get("document_id") or chunks[index].document_id == item["document_id"])]
        if relevant:
            hits += 1
            reciprocal_rank += 1 / relevant[0]
    if not evaluated:
        raise ValueError("Each query needs a non-empty exact 'answer' field for evaluation.")
    sizes = [len(_words(chunk.text)) for chunk in chunks]
    return {
        "strategy": strategy, "chunk_size_words": size, "overlap_words": overlap,
        "chunks": len(chunks), "avg_chunk_words": round(sum(sizes) / len(sizes), 1),
        f"recall@{top_k}": round(hits / evaluated, 4), "mrr": round(reciprocal_rank / evaluated, 4),
        "queries": evaluated, "build_and_retrieve_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunking strategies on labeled Q&A.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/chunking_benchmark.csv"))
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows = []
    for strategy in STRATEGIES:
        for size in args.sizes:
            rows.append(benchmark(data["documents"], data["queries"], strategy, size,
                                  round(size * args.overlap_ratio), args.top_k))
    result = pd.DataFrame(rows).sort_values([f"recall@{args.top_k}", "mrr", "build_and_retrieve_ms"], ascending=[False, False, True])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
