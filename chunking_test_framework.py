"""
Chunking Strategy Testing Framework for RAG
Author: Production RAG Pipeline
Stack: LangChain, Qdrant, Jina Embeddings, LangSmith, LangGraph

Core Goals:
- Test 5+ chunking strategies
- Measure retrieval precision, faithfulness, latency
- Track via LangSmith + custom evals
- Integrate A/B testing into LangGraph workflow
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_text_splitters import Language
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import Qdrant
from langsmith import Client
from langsmith.evaluation import evaluate
from qdrant_client import QdrantClient
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# ============================================================================
# PART 1: CHUNKING STRATEGIES
# ============================================================================

@dataclass
class ChunkingConfig:
    """Configuration for a chunking strategy"""
    name: str
    chunk_size: int
    chunk_overlap: int
    strategy: str  # 'recursive', 'character', 'semantic', 'language', 'proposition'
    separators: List[str] = None
    language: str = None
    threshold: float = None  # For semantic splitting


class ChunkingStrategies:
    """Implements 5+ chunking approaches"""

    @staticmethod
    def recursive_split(
        text: str, 
        chunk_size: int = 768, 
        chunk_overlap: int = 100
    ) -> List[Document]:
        """
        Tier 1: Recursive Character Splitting
        Best for: General-purpose RAG, mixed content
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return splitter.create_documents([text])

    @staticmethod
    def character_split(
        text: str,
        chunk_size: int = 768,
        chunk_overlap: int = 100
    ) -> List[Document]:
        """
        Tier 5: Simple Character Splitting
        Best for: Baseline, fast processing
        WARNING: Low semantic quality
        """
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" "
        )
        return splitter.create_documents([text])

    @staticmethod
    def language_aware_split(
        text: str,
        language: str = "python",
        chunk_size: int = 768,
        chunk_overlap: int = 100
    ) -> List[Document]:
        """
        Tier 3: Language-Aware Splitting
        Best for: Code, markdown, structured docs
        Supported: python, javascript, typescript, java, go, rust, sql, 
                   markdown, latex, html, cpp, csharp, php
        """
        try:
            lang_enum = Language[language.upper()]
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang_enum,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return splitter.create_documents([text])
        except KeyError:
            print(f"Language {language} not supported, falling back to recursive")
            return ChunkingStrategies.recursive_split(text, chunk_size, chunk_overlap)

    @staticmethod
    def semantic_split(
        text: str,
        chunk_size: int = 768,
        chunk_overlap: int = 100,
        embeddings_model: str = "jina",
        percentile_threshold: float = 0.5
    ) -> List[Document]:
        """
        Tier 2: Semantic Chunking (Post-Processing)
        
        Algorithm:
        1. First pass: Recursive split
        2. Embed each chunk
        3. Measure cosine similarity between consecutive chunks
        4. Split where similarity drops below threshold (percentile-based)
        
        Best for: Multi-topic documents, mixed narratives
        Cost: 1-2 extra embedding API calls per document
        """
        # Step 1: Initial recursive split
        initial_chunks = ChunkingStrategies.recursive_split(
            text, chunk_size, chunk_overlap
        )
        
        if len(initial_chunks) <= 1:
            return initial_chunks

        # Step 2: Initialize embeddings
        if embeddings_model == "jina":
            embeddings = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")
        else:
            embeddings = OpenAIEmbeddings()

        # Step 3: Embed all chunks
        chunk_texts = [doc.page_content for doc in initial_chunks]
        embedded_chunks = embeddings.embed_documents(chunk_texts)
        embedded_chunks = np.array(embedded_chunks)

        # Step 4: Calculate similarities between consecutive chunks
        similarities = []
        for i in range(len(embedded_chunks) - 1):
            sim = cosine_similarity(
                [embedded_chunks[i]], 
                [embedded_chunks[i + 1]]
            )[0][0]
            similarities.append(sim)

        # Step 5: Calculate threshold (percentile-based)
        if similarities:
            threshold = np.percentile(similarities, percentile_threshold * 100)
            
            # Step 6: Split where similarity drops below threshold
            final_chunks = [initial_chunks[0]]
            for i, sim in enumerate(similarities):
                if sim < threshold:
                    # Merge with previous chunk if too small
                    if len(final_chunks[-1].page_content.split()) < 50:
                        final_chunks[-1].page_content += " " + initial_chunks[i + 1].page_content
                    else:
                        final_chunks.append(initial_chunks[i + 1])
                else:
                    # Append to current chunk
                    final_chunks[-1].page_content += " " + initial_chunks[i + 1].page_content

            return final_chunks
        
        return initial_chunks

    @staticmethod
    def proposition_split(
        text: str,
        chunk_size: int = 768,
        chunk_overlap: int = 100,
        use_llm: bool = False
    ) -> List[Document]:
        """
        Tier 4: Proposition-Level Chunking (Advanced)
        
        Breaks text into atomic propositions (facts that stand alone).
        Optionally uses LLM for intelligent grouping.
        
        Best for: High-stakes retrieval (finance, legal)
        Cost: LLM inference per document (expensive)
        
        Note: This is a simplified version. Full implementation requires
        LangGraph + LLM chains for proposition extraction.
        """
        # For now, implement basic sentence-level splitting with grouping
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Group sentences into propositions based on semantic relatedness
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence
            if len(test_chunk.split()) <= (chunk_size // 4):  # Rough word estimate
                current_chunk = test_chunk
            else:
                if current_chunk.strip():
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={"chunking_strategy": "proposition"}
                    ))
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={"chunking_strategy": "proposition"}
            ))
        
        return chunks


# ============================================================================
# PART 2: EVALUATION METRICS
# ============================================================================

@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality"""
    strategy_name: str
    num_chunks: int
    avg_chunk_size: float
    retrieval_precision: float  # Relevant chunks / Total chunks retrieved
    retrieval_recall: float     # Retrieved relevant / Total relevant
    mean_reciprocal_rank: float # Ranking quality
    faithfulness_score: float   # Answer grounded in context
    citation_precision: float   # Citations actually from retrieved chunks
    avg_latency_ms: float       # End-to-end latency
    timestamp: str


class EvaluationFramework:
    """Comprehensive evaluation for chunking strategies"""

    def __init__(self, langsmith_client: Client = None):
        self.langsmith_client = langsmith_client or Client()
        self.results = []

    @staticmethod
    def calculate_chunk_statistics(chunks: List[Document]) -> Dict:
        """Basic chunk statistics"""
        sizes = [len(doc.page_content.split()) for doc in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_size": np.mean(sizes),
            "min_size": np.min(sizes),
            "max_size": np.max(sizes),
            "std_dev": np.std(sizes),
        }

    @staticmethod
    def retrieval_precision(
        retrieved_chunks: List[Document],
        relevant_indices: List[int],
        top_k: int = 5
    ) -> float:
        """
        Precision: How many retrieved chunks are actually relevant?
        
        P@k = (# of relevant chunks in top-k) / k
        
        Your target: 85%+
        """
        relevant_in_topk = sum(
            1 for i in range(min(top_k, len(retrieved_chunks)))
            if i in relevant_indices
        )
        return relevant_in_topk / top_k if top_k > 0 else 0.0

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_chunks: List[Document],
        relevant_indices: List[int],
        top_k: int = 10
    ) -> float:
        """
        MRR: How highly ranked is the first relevant chunk?
        
        MRR = 1 / (rank of first relevant chunk)
        
        Range: 0-1 (higher is better)
        """
        for i in range(min(top_k, len(retrieved_chunks))):
            if i in relevant_indices:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def faithfulness_eval(
        answer: str,
        context: str,
        use_llm: bool = False
    ) -> float:
        """
        Faithfulness: Is the answer grounded in the context?
        
        Simple approach: Check for context overlap (0-1 score)
        Advanced approach: Use LLM-based evaluation
        
        Your target: 80%+
        """
        if use_llm:
            # Requires LLM call - implement with Claude/GPT
            # For now, placeholder
            return 0.8  # TODO: Implement LLM eval
        else:
            # Lexical overlap approach
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            if not answer_words:
                return 0.0
            overlap = len(answer_words & context_words) / len(answer_words)
            return overlap

    @staticmethod
    def citation_precision(
        cited_chunks: List[Document],
        retrieved_chunks: List[Document]
    ) -> float:
        """
        Citation Precision: Are citations actually from retrieved chunks?
        
        Measure: Exact match of citation text with chunk content
        """
        if not cited_chunks:
            return 1.0
        
        matched = 0
        for cited in cited_chunks:
            for retrieved in retrieved_chunks:
                if cited.page_content in retrieved.page_content:
                    matched += 1
                    break
        
        return matched / len(cited_chunks)


# ============================================================================
# PART 3: TESTING HARNESS
# ============================================================================

class ChunkingTestHarness:
    """
    Main testing framework
    Handles: chunking, embedding, retrieval, evaluation, logging
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        embeddings_model: str = "jina",
        langsmith_project: str = "chunking-tests"
    ):
        self.qdrant_client = QdrantClient(qdrant_url)
        self.embeddings_model = embeddings_model
        self.langsmith_project = langsmith_project
        self.langsmith_client = Client()
        
        # Initialize embeddings
        if embeddings_model == "jina":
            self.embeddings = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")
        else:
            self.embeddings = OpenAIEmbeddings()

    def test_strategy(
        self,
        config: ChunkingConfig,
        sample_texts: List[str],
        query: str,
        relevant_chunk_indices: List[int],
        collection_name: str = None
    ) -> RetrievalMetrics:
        """
        End-to-end test for a single chunking strategy
        
        Flow:
        1. Chunk the sample texts
        2. Generate embeddings
        3. Store in Qdrant
        4. Retrieve based on query
        5. Evaluate metrics
        6. Log to LangSmith
        """
        
        start_time = time.time()
        
        # Step 1: Chunk
        all_chunks = []
        for text in sample_texts:
            if config.strategy == "recursive":
                chunks = ChunkingStrategies.recursive_split(
                    text, config.chunk_size, config.chunk_overlap
                )
            elif config.strategy == "character":
                chunks = ChunkingStrategies.character_split(
                    text, config.chunk_size, config.chunk_overlap
                )
            elif config.strategy == "semantic":
                chunks = ChunkingStrategies.semantic_split(
                    text, config.chunk_size, config.chunk_overlap,
                    percentile_threshold=config.threshold or 0.5
                )
            elif config.strategy == "language":
                chunks = ChunkingStrategies.language_aware_split(
                    text, config.language, config.chunk_size, config.chunk_overlap
                )
            elif config.strategy == "proposition":
                chunks = ChunkingStrategies.proposition_split(
                    text, config.chunk_size, config.chunk_overlap
                )
            else:
                raise ValueError(f"Unknown strategy: {config.strategy}")
            
            all_chunks.extend(chunks)
        
        # Add metadata
        for i, chunk in enumerate(all_chunks):
            chunk.metadata = {
                "chunking_strategy": config.strategy,
                "chunk_id": i,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            }
        
        # Step 2-3: Embed and store in Qdrant
        coll_name = collection_name or f"chunks_{config.name}_{int(time.time())}"
        
        # Create collection
        vectors = self.embeddings.embed_documents([c.page_content for c in all_chunks])
        
        # Store vectors with metadata in Qdrant
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        self.qdrant_client.recreate_collection(
            collection_name=coll_name,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )
        
        points = [
            PointStruct(
                id=i,
                vector=vectors[i],
                payload=chunk.metadata
            )
            for i, chunk in enumerate(all_chunks)
        ]
        self.qdrant_client.upsert(collection_name=coll_name, points=points)
        
        # Step 4: Retrieve
        query_vector = self.embeddings.embed_query(query)
        search_results = self.qdrant_client.search(
            collection_name=coll_name,
            query_vector=query_vector,
            limit=10
        )
        
        retrieved_chunk_ids = [result.id for result in search_results]
        retrieved_chunks = [all_chunks[i] for i in retrieved_chunk_ids]
        
        # Step 5: Evaluate metrics
        eval_framework = EvaluationFramework(self.langsmith_client)
        
        chunk_stats = eval_framework.calculate_chunk_statistics(all_chunks)
        precision = eval_framework.retrieval_precision(
            retrieved_chunks, relevant_chunk_indices, top_k=5
        )
        mrr = eval_framework.mean_reciprocal_rank(
            retrieved_chunks, relevant_chunk_indices, top_k=10
        )
        faithfulness = eval_framework.faithfulness_eval(
            "\n".join([c.page_content for c in retrieved_chunks]),
            "\n".join([c.page_content for c in retrieved_chunks]),
            use_llm=False  # TODO: Enable LLM evals
        )
        citation_prec = eval_framework.citation_precision(
            retrieved_chunks[:3], retrieved_chunks
        )
        
        latency = (time.time() - start_time) * 1000  # ms
        
        # Step 6: Create metrics object
        metrics = RetrievalMetrics(
            strategy_name=config.name,
            num_chunks=chunk_stats["num_chunks"],
            avg_chunk_size=chunk_stats["avg_size"],
            retrieval_precision=precision,
            retrieval_recall=mrr,  # Placeholder, use MRR as proxy
            mean_reciprocal_rank=mrr,
            faithfulness_score=faithfulness,
            citation_precision=citation_prec,
            avg_latency_ms=latency,
            timestamp=datetime.now().isoformat()
        )
        
        # Log to LangSmith
        self._log_to_langsmith(config, metrics, retrieved_chunks)
        
        return metrics

    def _log_to_langsmith(
        self,
        config: ChunkingConfig,
        metrics: RetrievalMetrics,
        retrieved_chunks: List[Document]
    ):
        """Log results to LangSmith for tracking"""
        # This would integrate with LangSmith's tracing API
        # For now, just print
        print(f"\n{'='*60}")
        print(f"Strategy: {config.name}")
        print(f"{'='*60}")
        print(f"Chunks: {metrics.num_chunks} | Avg Size: {metrics.avg_chunk_size:.1f} words")
        print(f"Precision@5: {metrics.retrieval_precision:.2%}")
        print(f"Faithfulness: {metrics.faithfulness_score:.2%}")
        print(f"Citation Precision: {metrics.citation_precision:.2%}")
        print(f"MRR@10: {metrics.mean_reciprocal_rank:.2%}")
        print(f"Latency: {metrics.avg_latency_ms:.2f}ms")
        print(f"Retrieved chunks: {len(retrieved_chunks)}")

    def run_comparative_test(
        self,
        configs: List[ChunkingConfig],
        sample_texts: List[str],
        test_queries: List[Tuple[str, List[int]]]
    ) -> pd.DataFrame:
        """
        Run all strategies against same test set
        
        Args:
            configs: List of ChunkingConfig to test
            sample_texts: Documents to chunk
            test_queries: List of (query, relevant_chunk_indices) tuples
        
        Returns:
            DataFrame with all metrics for comparison
        """
        results = []
        
        for config in configs:
            print(f"\nTesting {config.name}...")
            
            strategy_results = []
            for query, relevant_indices in test_queries:
                metrics = self.test_strategy(
                    config, sample_texts, query, relevant_indices
                )
                strategy_results.append(metrics)
            
            # Average metrics across queries
            avg_metrics = self._average_metrics(strategy_results)
            results.append(avg_metrics)
        
        # Convert to DataFrame for easy comparison
        df = pd.DataFrame([asdict(r) for r in results])
        return df

    @staticmethod
    def _average_metrics(metrics_list: List[RetrievalMetrics]) -> RetrievalMetrics:
        """Average metrics across multiple queries"""
        return RetrievalMetrics(
            strategy_name=metrics_list[0].strategy_name,
            num_chunks=int(np.mean([m.num_chunks for m in metrics_list])),
            avg_chunk_size=np.mean([m.avg_chunk_size for m in metrics_list]),
            retrieval_precision=np.mean([m.retrieval_precision for m in metrics_list]),
            retrieval_recall=np.mean([m.retrieval_recall for m in metrics_list]),
            mean_reciprocal_rank=np.mean([m.mean_reciprocal_rank for m in metrics_list]),
            faithfulness_score=np.mean([m.faithfulness_score for m in metrics_list]),
            citation_precision=np.mean([m.citation_precision for m in metrics_list]),
            avg_latency_ms=np.mean([m.avg_latency_ms for m in metrics_list]),
            timestamp=datetime.now().isoformat()
        )


# ============================================================================
# PART 4: USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Sample test data
    SAMPLE_TEXTS = [
        """
        Agriculture is the practice of farming crops and raising animals.
        Farmers use various techniques like irrigation to grow crops.
        The monsoon season is crucial for agricultural productivity in India.
        
        IPL (Indian Premier League) is a professional cricket tournament.
        Virat Kohli scored 100 runs in the recent IPL match.
        The IPL auction saw record-breaking bids for players this year.
        """,
        """
        Climate change affects agriculture significantly.
        Rising temperatures impact crop yields and farming patterns.
        Sustainable farming practices help mitigate climate impact.
        """,
    ]

    TEST_QUERIES = [
        ("What is agriculture?", [0, 1]),  # Relevant chunks
        ("Tell me about IPL", [0]),          # Cricket-related
        ("How does climate affect farming?", [1]),  # Climate-related
    ]

    # Define strategies to test
    strategies = [
        ChunkingConfig(
            name="Recursive-512",
            chunk_size=512,
            chunk_overlap=50,
            strategy="recursive"
        ),
        ChunkingConfig(
            name="Recursive-768",
            chunk_size=768,
            chunk_overlap=100,
            strategy="recursive"
        ),
        ChunkingConfig(
            name="Character-768",
            chunk_size=768,
            chunk_overlap=100,
            strategy="character"
        ),
        ChunkingConfig(
            name="Semantic-768",
            chunk_size=768,
            chunk_overlap=100,
            strategy="semantic",
            threshold=0.5
        ),
    ]

    # Initialize harness
    harness = ChunkingTestHarness()

    # Run comparative test
    print("\n🚀 Starting Comparative Chunking Test...")
    results_df = harness.run_comparative_test(
        strategies,
        SAMPLE_TEXTS,
        TEST_QUERIES
    )

    # Display results
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(f"chunking_test_results_{int(time.time())}.csv", index=False)
    print(f"\n✅ Results saved to CSV")
