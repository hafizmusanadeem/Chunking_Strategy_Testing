"""
LangGraph Integration: Chunking Strategies in Production RAG
Stack: LangGraph, LangChain, Qdrant, Jina, LangSmith

This shows how to:
1. Use LangGraph for chunking workflow
2. A/B test different strategies in parallel
3. Track metrics via LangSmith
4. Make dynamic chunking decisions based on document type
"""

from typing import List, Dict, TypedDict, Literal
from enum import Enum
import json
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langsmith import Client
from langsmith.run_helpers import get_current_run_tree
import asyncio


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    RECURSIVE = "recursive"
    CHARACTER = "character"
    SEMANTIC = "semantic"
    LANGUAGE = "language"
    PROPOSITION = "proposition"


class DocumentState(TypedDict):
    """State for document processing in LangGraph"""
    
    # Input
    raw_text: str
    document_id: str
    document_type: Literal["general", "code", "markdown", "finance", "legal"]
    
    # Chunking decisions
    selected_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    
    # Processing results
    chunks: List[Document]
    chunk_stats: Dict
    
    # Evaluation (filled after retrieval test)
    metrics: Dict
    
    # Metadata
    timestamp: str
    trace_id: str


class RAGEvaluationState(TypedDict):
    """State for RAG evaluation pipeline"""
    
    # Document + chunking
    document_id: str
    chunks: List[Document]
    chunking_strategy: ChunkingStrategy
    
    # Retrieval
    query: str
    retrieved_chunks: List[Document]
    retrieval_scores: List[float]
    
    # Evaluation
    precision_at_5: float
    precision_at_10: float
    faithfulness_score: float
    citation_precision: float
    latency_ms: float
    
    # Results
    passed_evals: bool
    eval_report: Dict


# ============================================================================
# PART 1: ROUTER NODES (Decide chunking strategy based on doc type)
# ============================================================================

class DocumentTypeRouter:
    """
    Routes documents to appropriate chunking strategy
    based on document type and size
    """
    
    @staticmethod
    def route(state: DocumentState) -> Dict:
        """
        Decision logic:
        - Finance/Legal → Proposition (high precision needed)
        - Code/Markdown → Language-aware
        - General → Recursive (best balance)
        - Large mixed-topic → Semantic
        """
        
        doc_type = state["document_type"]
        text_length = len(state["raw_text"].split())
        
        if doc_type in ["finance", "legal"]:
            # High-stakes: Use proposition splitting
            strategy = ChunkingStrategy.PROPOSITION
            chunk_size = 512
            chunk_overlap = 75
            
        elif doc_type == "code":
            # Structured: Use language-aware splitting
            strategy = ChunkingStrategy.LANGUAGE
            chunk_size = 1024
            chunk_overlap = 100
            
        elif doc_type == "markdown":
            # Semi-structured: Use language-aware splitting
            strategy = ChunkingStrategy.LANGUAGE
            chunk_size = 768
            chunk_overlap = 100
            
        else:  # "general"
            # Mixed content
            if text_length > 5000:
                # Large mixed docs → add semantic post-processing
                strategy = ChunkingStrategy.SEMANTIC
                chunk_size = 768
                chunk_overlap = 100
            else:
                # Standard: Recursive splitting
                strategy = ChunkingStrategy.RECURSIVE
                chunk_size = 768
                chunk_overlap = 100
        
        return {
            "selected_strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# PART 2: CHUNKING NODES
# ============================================================================

class ChunkingNode:
    """
    Executes the selected chunking strategy
    Imported from chunking_test_framework.py
    """
    
    def __init__(self):
        from chunking_test_framework import ChunkingStrategies
        self.strategies = ChunkingStrategies
    
    async def chunk_recursive(self, text: str, state: DocumentState) -> List[Document]:
        """Execute recursive chunking"""
        chunks = self.strategies.recursive_split(
            text,
            chunk_size=state["chunk_size"],
            chunk_overlap=state["chunk_overlap"]
        )
        return chunks
    
    async def chunk_semantic(self, text: str, state: DocumentState) -> List[Document]:
        """Execute semantic chunking"""
        chunks = self.strategies.semantic_split(
            text,
            chunk_size=state["chunk_size"],
            chunk_overlap=state["chunk_overlap"],
            percentile_threshold=0.5
        )
        return chunks
    
    async def chunk_language(self, text: str, state: DocumentState) -> List[Document]:
        """Execute language-aware chunking"""
        language_map = {
            "code": "python",
            "markdown": "markdown",
            "general": "python"
        }
        chunks = self.strategies.language_aware_split(
            text,
            language=language_map.get(state["document_type"], "python"),
            chunk_size=state["chunk_size"],
            chunk_overlap=state["chunk_overlap"]
        )
        return chunks
    
    async def chunk_proposition(self, text: str, state: DocumentState) -> List[Document]:
        """Execute proposition chunking"""
        chunks = self.strategies.proposition_split(
            text,
            chunk_size=state["chunk_size"],
            chunk_overlap=state["chunk_overlap"]
        )
        return chunks
    
    async def execute(self, state: DocumentState) -> Dict:
        """Main chunking node"""
        
        strategy = state["selected_strategy"]
        text = state["raw_text"]
        
        # Route to appropriate chunking method
        if strategy == ChunkingStrategy.RECURSIVE:
            chunks = await self.chunk_recursive(text, state)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = await self.chunk_semantic(text, state)
        elif strategy == ChunkingStrategy.LANGUAGE:
            chunks = await self.chunk_language(text, state)
        elif strategy == ChunkingStrategy.PROPOSITION:
            chunks = await self.chunk_proposition(text, state)
        else:
            # Fallback
            chunks = await self.chunk_recursive(text, state)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                "document_id": state["document_id"],
                "chunking_strategy": strategy.value,
                "chunk_index": i,
                "chunk_size": state["chunk_size"],
                "chunk_overlap": state["chunk_overlap"],
            }
        
        # Calculate statistics
        chunk_stats = {
            "num_chunks": len(chunks),
            "avg_size": sum(len(c.page_content.split()) for c in chunks) / len(chunks) if chunks else 0,
            "min_size": min(len(c.page_content.split()) for c in chunks) if chunks else 0,
            "max_size": max(len(c.page_content.split()) for c in chunks) if chunks else 0,
        }
        
        return {
            "chunks": chunks,
            "chunk_stats": chunk_stats,
            "trace_id": str(get_current_run_tree().id) if get_current_run_tree() else "unknown"
        }


# ============================================================================
# PART 3: EVALUATION NODES
# ============================================================================

class EvaluationNode:
    """
    Evaluates chunking quality via retrieval tests
    """
    
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        from chunking_test_framework import EvaluationFramework
        self.eval_framework = EvaluationFramework()
    
    async def execute(
        self,
        state: RAGEvaluationState,
        query: str,
        relevant_chunk_indices: List[int]
    ) -> Dict:
        """
        Run evaluation:
        1. Embed chunks
        2. Simulate retrieval
        3. Calculate metrics
        4. Check against thresholds
        """
        
        retrieved_chunks = state.get("retrieved_chunks", [])
        
        # Calculate metrics
        precision_5 = self.eval_framework.retrieval_precision(
            retrieved_chunks, relevant_chunk_indices, top_k=5
        )
        precision_10 = self.eval_framework.retrieval_precision(
            retrieved_chunks, relevant_chunk_indices, top_k=10
        )
        faithfulness = self.eval_framework.faithfulness_eval(
            "\n".join([c.page_content for c in retrieved_chunks[:3]]),
            "\n".join([c.page_content for c in state.get("chunks", [])]),
            use_llm=False
        )
        citation_prec = self.eval_framework.citation_precision(
            retrieved_chunks[:3], retrieved_chunks
        )
        
        # Determine pass/fail
        # Targets: 85% citation precision, 80% faithfulness
        passed = (
            citation_prec >= 0.85 and
            faithfulness >= 0.80 and
            precision_5 >= 0.70
        )
        
        eval_report = {
            "precision_at_5": precision_5,
            "precision_at_10": precision_10,
            "faithfulness_score": faithfulness,
            "citation_precision": citation_prec,
            "passed_evals": passed,
            "thresholds": {
                "citation_precision_target": 0.85,
                "faithfulness_target": 0.80,
                "precision_at_5_target": 0.70
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "precision_at_5": precision_5,
            "precision_at_10": precision_10,
            "faithfulness_score": faithfulness,
            "citation_precision": citation_prec,
            "passed_evals": passed,
            "eval_report": eval_report,
        }


# ============================================================================
# PART 4: A/B TESTING GRAPH
# ============================================================================

class ABTestingGraph:
    """
    Parallel A/B testing of chunking strategies
    Tests multiple strategies on same document simultaneously
    """
    
    def __init__(self, langsmith_client: Client = None):
        self.langsmith_client = langsmith_client or Client()
        self.chunking_node = ChunkingNode()
        self.eval_node = EvaluationNode()
    
    def build_graph(self):
        """
        Build LangGraph for chunking strategy comparison
        
        Flow:
        1. Router: Suggest strategy based on doc type
        2. Parallel chunking: Test 3-4 strategies
        3. Parallel evaluation: Retrieve and measure metrics
        4. Comparison: Rank by citation precision
        5. Selection: Pick best strategy
        """
        
        graph = StateGraph(DocumentState)
        
        # Node 1: Router (decide strategy)
        def route_node(state: DocumentState) -> Dict:
            routing = DocumentTypeRouter.route(state)
            return routing
        
        # Node 2: Execute chunking
        async def chunk_node(state: DocumentState) -> Dict:
            return await self.chunking_node.execute(state)
        
        # Node 3: Evaluate (would be called after retrieval)
        async def eval_node(state: DocumentState) -> Dict:
            # This would be filled in based on actual retrieval results
            return {"metrics": {}}
        
        # Add nodes
        graph.add_node("router", route_node)
        graph.add_node("chunker", chunk_node)
        graph.add_node("evaluator", eval_node)
        
        # Edges
        graph.set_entry_point("router")
        graph.add_edge("router", "chunker")
        graph.add_edge("chunker", "evaluator")
        graph.add_edge("evaluator", END)
        
        return graph.compile()


# ============================================================================
# PART 5: PARALLEL A/B TEST (for multiple strategies)
# ============================================================================

class ParallelStrategyTester:
    """
    Tests multiple chunking strategies in parallel on the same document
    Results are compared for winner selection
    """
    
    def __init__(self):
        from chunking_test_framework import ChunkingStrategies, EvaluationFramework
        self.strategies_module = ChunkingStrategies
        self.eval_framework = EvaluationFramework()
    
    async def test_strategy(
        self,
        strategy: ChunkingStrategy,
        text: str,
        query: str,
        relevant_indices: List[int],
        **kwargs
    ) -> Dict:
        """Test a single strategy"""
        
        # Execute chunking
        if strategy == ChunkingStrategy.RECURSIVE:
            chunks = self.strategies_module.recursive_split(
                text,
                chunk_size=kwargs.get("chunk_size", 768),
                chunk_overlap=kwargs.get("chunk_overlap", 100)
            )
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = self.strategies_module.semantic_split(
                text,
                chunk_size=kwargs.get("chunk_size", 768),
                chunk_overlap=kwargs.get("chunk_overlap", 100)
            )
        else:
            chunks = self.strategies_module.recursive_split(text)
        
        # Simulate retrieval (in reality, would embed and search Qdrant)
        retrieved = chunks[:5]  # Placeholder
        
        # Evaluate
        metrics = {
            "strategy": strategy.value,
            "num_chunks": len(chunks),
            "avg_chunk_size": sum(len(c.page_content.split()) for c in chunks) / len(chunks) if chunks else 0,
            "precision_at_5": self.eval_framework.retrieval_precision(
                retrieved, relevant_indices, top_k=5
            ),
            "citation_precision": self.eval_framework.citation_precision(
                retrieved[:3], retrieved
            ),
        }
        
        return metrics
    
    async def run_parallel_test(
        self,
        text: str,
        query: str,
        relevant_indices: List[int],
        strategies: List[ChunkingStrategy],
        **kwargs
    ) -> Dict:
        """
        Run all strategies in parallel
        
        Returns:
            Dict with results for each strategy, winner, and recommendations
        """
        
        # Run all tests concurrently
        tasks = [
            self.test_strategy(strat, text, query, relevant_indices, **kwargs)
            for strat in strategies
        ]
        results = await asyncio.gather(*tasks)
        
        # Rank by citation precision
        ranked = sorted(results, key=lambda x: x["citation_precision"], reverse=True)
        
        # Prepare report
        report = {
            "timestamp": datetime.now().isoformat(),
            "results": ranked,
            "winner": ranked[0]["strategy"],
            "winner_metrics": ranked[0],
            "recommendations": self._generate_recommendations(ranked)
        }
        
        return report
    
    @staticmethod
    def _generate_recommendations(ranked_results: List[Dict]) -> Dict:
        """Generate actionable recommendations based on results"""
        
        winner = ranked_results[0]
        
        recommendations = {
            "primary_strategy": winner["strategy"],
            "reason": f"Best citation precision: {winner['citation_precision']:.2%}",
            "chunk_size_recommendation": winner.get("chunk_size", 768),
            "expected_metrics": {
                "citation_precision": winner["citation_precision"],
                "num_chunks": winner["num_chunks"],
                "avg_chunk_size": winner["avg_chunk_size"],
            },
            "runner_up": ranked_results[1]["strategy"] if len(ranked_results) > 1 else None,
            "fallback_strategy": ranked_results[2]["strategy"] if len(ranked_results) > 2 else None,
        }
        
        # Add specific notes
        if winner["citation_precision"] > 0.85:
            recommendations["note"] = "✅ Meets production target (85%+ citation precision)"
        else:
            recommendations["note"] = "⚠️ Below target - may need tuning"
        
        return recommendations


# ============================================================================
# PART 6: USAGE EXAMPLE
# ============================================================================

async def example_parallel_ab_test():
    """
    Example: A/B test chunking strategies on a document
    """
    
    sample_text = """
    Machine learning models require careful hyperparameter tuning.
    The learning rate controls optimization speed and stability.
    
    [Separate topic]
    
    Linear regression is a fundamental algorithm in machine learning.
    It models the relationship between input features and output targets.
    """
    
    strategies = [
        ChunkingStrategy.RECURSIVE,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.CHARACTER,
    ]
    
    tester = ParallelStrategyTester()
    
    report = await tester.run_parallel_test(
        text=sample_text,
        query="How does learning rate work?",
        relevant_indices=[0, 1],  # First two chunks
        strategies=strategies,
        chunk_size=256,
        chunk_overlap=50
    )
    
    print("\n" + "="*80)
    print("A/B TEST RESULTS")
    print("="*80)
    print(f"Winner: {report['winner']}")
    print(f"Citation Precision: {report['winner_metrics']['citation_precision']:.2%}")
    print(f"\nRecommendations:")
    for key, value in report['recommendations'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "-"*80)
    print("All Results (Ranked):")
    for i, result in enumerate(report['results'], 1):
        print(f"\n{i}. {result['strategy'].upper()}")
        print(f"   Citation Precision: {result['citation_precision']:.2%}")
        print(f"   Num Chunks: {result['num_chunks']}")
        print(f"   Avg Chunk Size: {result['avg_chunk_size']:.1f} words")


# ============================================================================
# USAGE IN MAIN RAG FLOW
# ============================================================================

if __name__ == "__main__":
    # Run example A/B test
    asyncio.run(example_parallel_ab_test())
