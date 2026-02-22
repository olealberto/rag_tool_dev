# ============================================================================
# 📁 run_evaluation.py - STANDALONE EVALUATION SCRIPT
# ============================================================================

"""
STANDALONE EVALUATION FOR PHASES 3, 4, AND 5
Run with: !python run_evaluation.py

Loads everything from disk - no recomputing embeddings or rebuilding graphs.

What it loads:
    - ./phase2_output/nih_research_abstracts.csv       (abstracts)
    - ./phase3_results/document_chunks_with_embeddings.csv  (chunks + embeddings)
    - ./phase3_results/faiss_index.bin                 (FAISS index)
    - ./phase5_knowledge_graph.gml                     (knowledge graph)

What it produces:
    - ./evaluation/test_set.json
    - ./evaluation/phase3_eval.json
    - ./evaluation/phase4_eval.json
    - ./evaluation/phase5_eval.json
    - ./evaluation/phase_comparison.json
    - ./evaluation/phase_comparison.png
"""

print("="*70)
print("🧪 RAG EVALUATION: PHASES 3, 4, AND 5")
print("="*70)

import os
import sys
import json
import time
import ast
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab

from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime
from collections import defaultdict

# ============ PATHS — all pre-computed assets ============

PATHS = {
    "abstracts":          "./phase2_output/nih_research_abstracts.csv",
    "chunks":             "./phase3_results/document_chunks_with_embeddings.csv",
    "chunks_no_emb":      "./phase3_results/document_chunks.csv",
    "faiss_index":        "./phase3_results/faiss_index.bin",
    "embeddings":         "./phase3_results/chunk_embeddings.npy",
    "graph_gml":          "./phase5_knowledge_graph.gml",
    "phase2_eval_set":    "./phase2_output/evaluation_set.json",
    "output_dir":         "./evaluation",
}

Path(PATHS["output_dir"]).mkdir(exist_ok=True)

# ============ GROUND TRUTH QUERIES ============
# Built from domain knowledge about NIH grant topics.
# Keyword matching against your abstracts determines relevant set per query.

SEED_QUERIES = [
    {
        "query_id": "Q01", "topic": "diabetes",
        "query": "diabetes prevention and management in community health settings",
        "keywords": ["diabetes", "diabetic", "hba1c", "glycemic", "insulin"]
    },
    {
        "query_id": "Q02", "topic": "CHW",
        "query": "community health worker interventions for underserved populations",
        "keywords": ["community health worker", "chw", "promotora", "lay health"]
    },
    {
        "query_id": "Q03", "topic": "behavioral_health",
        "query": "behavioral health integration in primary care",
        "keywords": ["behavioral health", "mental health", "depression", "anxiety",
                     "integrated care"]
    },
    {
        "query_id": "Q04", "topic": "cancer_screening",
        "query": "cancer screening programs in community health centers",
        "keywords": ["cancer", "screening", "mammography", "colonoscopy", "cervical"]
    },
    {
        "query_id": "Q05", "topic": "health_disparities",
        "query": "health disparities in minority and low income populations",
        "keywords": ["health disparit", "minority", "low-income", "underserved",
                     "racial", "ethnic"]
    },
    {
        "query_id": "Q06", "topic": "telehealth",
        "query": "telehealth and digital health interventions for chronic disease",
        "keywords": ["telehealth", "telemedicine", "digital health", "mobile health",
                     "mhealth"]
    },
    {
        "query_id": "Q07", "topic": "substance_use",
        "query": "substance use disorder treatment and opioid addiction",
        "keywords": ["substance use", "opioid", "addiction", "naloxone",
                     "buprenorphine"]
    },
    {
        "query_id": "Q08", "topic": "social_determinants",
        "query": "social determinants of health screening and referral programs",
        "keywords": ["social determinants", "food insecurity", "housing", "sdoh",
                     "social needs"]
    },
    {
        "query_id": "Q09", "topic": "hypertension",
        "query": "hypertension control and cardiovascular disease prevention",
        "keywords": ["hypertension", "blood pressure", "cardiovascular",
                     "heart disease"]
    },
    {
        "query_id": "Q10", "topic": "HIV",
        "query": "HIV prevention and treatment in high risk populations",
        "keywords": ["hiv", "aids", "prep", "antiretroviral"]
    },
    {
        "query_id": "Q11", "topic": "pediatric",
        "query": "pediatric health interventions in community settings",
        "keywords": ["pediatric", "children", "adolescent", "youth", "child health"]
    },
    {
        "query_id": "Q12", "topic": "Latino_health",
        "query": "Latino and Hispanic health promotion programs",
        "keywords": ["latino", "hispanic", "latinx", "spanish speaking", "promotora"]
    },
]


# ============ METRICS ============

class IRMetrics:
    """Standard Information Retrieval metrics"""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        top_k = retrieved[:k]
        return sum(1 for r in top_k if r in relevant) / k if k else 0.0

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        return sum(1 for r in top_k if r in relevant) / len(relevant)

    @staticmethod
    def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
        if not relevant or not retrieved:
            return 0.0
        hits, score = 0, 0.0
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                hits += 1
                score += hits / (i + 1)
        return score / len(relevant)

    @staticmethod
    def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        def dcg(lst, rel, k):
            return sum(
                1.0 / np.log2(i + 2)
                for i, d in enumerate(lst[:k]) if d in rel
            )
        actual = dcg(retrieved, relevant, k)
        ideal = dcg(list(relevant)[:k], relevant, k)
        return actual / ideal if ideal else 0.0

    @classmethod
    def compute_all(cls, retrieved: List[str],
                    relevant_ids: List[str]) -> Dict:
        relevant = set(relevant_ids)
        m = {}
        for k in [1, 3, 5, 10]:
            m[f"P@{k}"]    = cls.precision_at_k(retrieved, relevant, k)
            m[f"R@{k}"]    = cls.recall_at_k(retrieved, relevant, k)
            m[f"nDCG@{k}"] = cls.ndcg_at_k(retrieved, relevant, k)
        m["AP"]  = cls.average_precision(retrieved, relevant)
        m["RR"]  = cls.reciprocal_rank(retrieved, relevant)
        m["hits@5"] = sum(1 for r in retrieved[:5] if r in relevant)
        return m

    @staticmethod
    def aggregate(query_metrics: List[Dict]) -> Dict:
        if not query_metrics:
            return {}
        keys = [k for k in query_metrics[0] if isinstance(query_metrics[0][k], float)]
        agg = {}
        for k in keys:
            vals = [m[k] for m in query_metrics]
            agg[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4),
                "min":  round(float(np.min(vals)),  4),
                "max":  round(float(np.max(vals)),  4),
            }
        agg["MAP"] = agg.get("AP", {}).get("mean", 0)
        agg["MRR"] = agg.get("RR", {}).get("mean", 0)
        return agg


# ============ DATA LOADER ============

class DataLoader:
    """Loads all pre-computed assets from disk"""

    def __init__(self):
        self.abstracts_df   = None
        self.chunks_df      = None
        self.embeddings     = None
        self.faiss_index    = None
        self.graph          = None
        self.model          = None

    def load_all(self):
        print("\n📦 LOADING PRE-COMPUTED ASSETS")
        print("-"*50)
        self._load_abstracts()
        self._load_chunks()
        self._load_embeddings()
        self._load_faiss()
        self._load_graph()
        self._load_model()
        print("\n✅ All assets loaded")

    def _load_abstracts(self):
        path = PATHS["abstracts"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Abstracts not found: {path}\nRun phase2_api.py first")
        self.abstracts_df = pd.read_csv(path)
        print(f"  ✅ Abstracts:   {len(self.abstracts_df)} rows  ({path})")

    def _load_chunks(self):
        # Prefer chunks without embeddings for speed — embeddings loaded separately
        path = PATHS["chunks_no_emb"]
        if not os.path.exists(path):
            path = PATHS["chunks"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chunks not found. Run phase3_document_rag.py first")

        self.chunks_df = pd.read_csv(path)

        # Drop embedding column if it snuck in (saves memory)
        if "embedding" in self.chunks_df.columns:
            self.chunks_df = self.chunks_df.drop(columns=["embedding"])

        print(f"  ✅ Chunks:      {len(self.chunks_df)} rows  ({path})")

    def _load_embeddings(self):
        path = PATHS["embeddings"]
        if not os.path.exists(path):
            print(f"  ⚠️  Embeddings not found at {path} — vector search unavailable")
            return
        self.embeddings = np.load(path)
        print(f"  ✅ Embeddings:  {self.embeddings.shape}  ({path})")

    def _load_faiss(self):
        path = PATHS["faiss_index"]
        if not os.path.exists(path):
            print(f"  ⚠️  FAISS index not found at {path} — Phase 3 eval unavailable")
            return
        try:
            import faiss
            self.faiss_index = faiss.read_index(path)
            print(f"  ✅ FAISS:       {self.faiss_index.ntotal} vectors  ({path})")
        except ImportError:
            print("  ⚠️  faiss-cpu not installed — Phase 3 eval unavailable")

    def _load_graph(self):
        path = PATHS["graph_gml"]
        if not os.path.exists(path):
            print(f"  ⚠️  Graph not found at {path} — Phase 5 eval unavailable")
            return
        self.graph = nx.read_gml(path)
        print(f"  ✅ Graph:       {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges  ({path})")

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Use same model as Phase 3
            model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
            print(f"  ⏳ Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            print(f"  ✅ Model loaded")
        except Exception as e:
            print(f"  ⚠️  Could not load model: {e}")


# ============ GROUND TRUTH BUILDER ============

class GroundTruthBuilder:
    """Builds labeled test set from abstracts using keyword matching"""

    def __init__(self, abstracts_df: pd.DataFrame):
        self.df = abstracts_df.copy()
        self.df["_text"] = (
            self.df.get("abstract", self.df.get("text", pd.Series([""] * len(self.df))))
            .fillna("").str.lower()
        )
        self.id_col = "grant_id" if "grant_id" in self.df.columns else self.df.columns[0]

    def build(self, min_relevant: int = 2) -> List[Dict]:
        # Load cached if exists
        cache_path = Path(PATHS["output_dir"]) / "test_set.json"
        if cache_path.exists():
            with open(cache_path) as f:
                test_set = json.load(f)
            print(f"  📦 Loaded cached test set: {len(test_set)} queries")
            return test_set

        print(f"\n  Building test set from {len(self.df)} abstracts...")
        test_set = []

        for q in SEED_QUERIES:
            relevant_ids = self.df[
                self.df["_text"].apply(
                    lambda t: any(kw in t for kw in q["keywords"])
                )
            ][self.id_col].tolist()

            if len(relevant_ids) < min_relevant:
                print(f"    ⚠️  {q['query_id']} ({q['topic']}): "
                      f"only {len(relevant_ids)} relevant — skipping")
                continue

            test_set.append({
                "query_id":     q["query_id"],
                "query":        q["query"],
                "topic":        q["topic"],
                "relevant_ids": relevant_ids[:50],
                "relevant_count": len(relevant_ids),
            })
            print(f"    ✅ {q['query_id']} ({q['topic']}): {len(relevant_ids)} relevant")

        with open(cache_path, "w") as f:
            json.dump(test_set, f, indent=2)
        print(f"  💾 Saved test set: {len(test_set)} queries")
        return test_set


# ============ PHASE 3 EVALUATOR (FAISS) ============

class Phase3Evaluator:
    """Evaluates Phase 3 FAISS vector search using pre-built index"""

    def __init__(self, data: DataLoader):
        self.data = data

    def evaluate(self, test_set: List[Dict], top_k: int = 10) -> Dict:
        print(f"\n{'='*70}")
        print("🧪 PHASE 3 EVALUATION: FAISS VECTOR SEARCH")
        print(f"{'='*70}")

        if self.data.faiss_index is None or self.data.model is None:
            print("  ⚠️  FAISS index or model not available — skipping Phase 3 eval")
            return {}

        import faiss
        query_metrics = []
        retrieval_times = []

        for q in test_set:
            query_vec = self.data.model.encode([q["query"]])
            faiss.normalize_L2(query_vec)

            start = time.time()
            distances, indices = self.data.faiss_index.search(query_vec, top_k)
            retrieval_times.append(time.time() - start)

            # Map indices back to grant_ids
            retrieved_ids = []
            for idx in indices[0]:
                if 0 <= idx < len(self.data.chunks_df):
                    grant_id = self.data.chunks_df.iloc[idx].get("grant_id", "")
                    if grant_id:
                        retrieved_ids.append(str(grant_id))

            metrics = IRMetrics.compute_all(retrieved_ids, q["relevant_ids"])
            metrics["query_id"] = q["query_id"]
            metrics["topic"]    = q["topic"]
            query_metrics.append(metrics)

        agg = IRMetrics.aggregate(query_metrics)
        agg["avg_retrieval_time_s"] = round(float(np.mean(retrieval_times)), 4)
        agg["per_query"] = query_metrics

        print(f"  MAP:    {agg['MAP']:.4f}")
        print(f"  MRR:    {agg['MRR']:.4f}")
        print(f"  P@5:    {agg.get('P@5', {}).get('mean', 0):.4f}")
        print(f"  nDCG@5: {agg.get('nDCG@5', {}).get('mean', 0):.4f}")
        print(f"  Avg retrieval time: {agg['avg_retrieval_time_s']:.4f}s")

        return agg


# ============ PHASE 4 EVALUATOR (WEAVIATE HYBRID) ============

class Phase4Evaluator:
    """
    Evaluates Weaviate hybrid search.
    Starts its own Weaviate embedded instance, imports chunks, evaluates, closes.
    """

    def __init__(self, data: DataLoader):
        self.data   = data
        self.client = None

    def evaluate(self, test_set: List[Dict],
                 alpha_values: List[float] = None,
                 top_k: int = 10) -> Dict:

        if alpha_values is None:
            alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        print(f"\n{'='*70}")
        print("🧪 PHASE 4 EVALUATION: WEAVIATE HYBRID SEARCH")
        print(f"{'='*70}")

        # Start Weaviate
        try:
            import weaviate
            from weaviate.classes.query import MetadataQuery
            from weaviate.classes.config import Property, DataType

            print("  🚀 Starting Weaviate embedded...")
            self.client = weaviate.connect_to_embedded()
            assert self.client.is_ready()
            print("  ✅ Weaviate ready")
        except Exception as e:
            print(f"  ❌ Weaviate failed: {e} — skipping Phase 4 eval")
            return {}

        try:
            # Create schema and import chunks
            collection = self._setup_collection()
            if collection is None:
                return {}

            imported = self._import_chunks(collection)
            if imported == 0:
                print("  ❌ No chunks imported")
                return {}

            print(f"\n  Evaluating {len(alpha_values)} alpha values across "
                  f"{len(test_set)} queries...")

            results_by_alpha = {}

            for alpha in alpha_values:
                label = ("BM25 only" if alpha == 0.0
                         else "vector only" if alpha == 1.0
                         else f"hybrid α={alpha}")
                print(f"\n  📊 {label}...")

                query_metrics  = []
                retrieval_times = []

                for q in test_set:
                    query_vec = (
                        self.data.model.encode(q["query"]).tolist()
                        if self.data.model and alpha > 0 else None
                    )

                    start = time.time()
                    retrieved_ids = self._hybrid_search(
                        collection, q["query"], alpha, query_vec, top_k
                    )
                    retrieval_times.append(time.time() - start)

                    metrics = IRMetrics.compute_all(retrieved_ids, q["relevant_ids"])
                    metrics["query_id"] = q["query_id"]
                    metrics["topic"]    = q["topic"]
                    query_metrics.append(metrics)

                agg = IRMetrics.aggregate(query_metrics)
                agg["avg_retrieval_time_s"] = round(float(np.mean(retrieval_times)), 4)
                agg["alpha"]       = alpha
                agg["label"]       = label
                agg["per_query"]   = query_metrics
                results_by_alpha[alpha] = agg

                print(f"    MAP={agg['MAP']:.4f}  MRR={agg['MRR']:.4f}  "
                      f"P@5={agg.get('P@5',{}).get('mean',0):.4f}  "
                      f"nDCG@5={agg.get('nDCG@5',{}).get('mean',0):.4f}  "
                      f"time={agg['avg_retrieval_time_s']:.4f}s")

            # Best alpha by MAP
            optimal_alpha = max(results_by_alpha,
                                key=lambda a: results_by_alpha[a]["MAP"])
            print(f"\n  🎯 Optimal alpha: {optimal_alpha} "
                  f"(MAP={results_by_alpha[optimal_alpha]['MAP']:.4f})")

            self._print_alpha_table(results_by_alpha)

            return {
                "results_by_alpha": results_by_alpha,
                "optimal_alpha":    optimal_alpha,
                "optimal_metrics":  results_by_alpha[optimal_alpha],
                "test_set_size":    len(test_set),
            }

        finally:
            if self.client:
                self.client.close()
                print("\n  👋 Weaviate closed")

    def _setup_collection(self):
        try:
            from weaviate.classes.config import Property, DataType

            if self.client.collections.exists("EvalGrant"):
                self.client.collections.delete("EvalGrant")

            self.client.collections.create(
                name="EvalGrant",
                vectorizer_config=None,
                properties=[
                    Property(name="text",         data_type=DataType.TEXT),
                    Property(name="grantId",       data_type=DataType.TEXT),
                    Property(name="institute",     data_type=DataType.TEXT),
                    Property(name="year",          data_type=DataType.INT),
                    Property(name="isFQHCFocused", data_type=DataType.BOOL),
                    Property(name="sectionType",   data_type=DataType.TEXT),
                ]
            )
            return self.client.collections.get("EvalGrant")
        except Exception as e:
            print(f"  ❌ Schema creation failed: {e}")
            return None

    def _import_chunks(self, collection) -> int:
        """Import chunks with pre-computed embeddings — no recomputing"""
        print(f"\n  📤 Loading embeddings from disk...")

        # Load chunks with embeddings from the CSV Phase 3 saved
        chunks_path = PATHS["chunks"]
        if not os.path.exists(chunks_path):
            print(f"  ❌ Chunks with embeddings not found: {chunks_path}")
            return 0

        # Load in batches to manage memory (76MB CSV)
        print(f"  ⏳ Reading {chunks_path}...")
        chunks_df = pd.read_csv(chunks_path)
        print(f"  ✅ Loaded {len(chunks_df)} chunks")

        total   = 0
        failed  = 0

        with collection.batch.fixed_size(batch_size=100) as batch:
            for idx, row in chunks_df.iterrows():
                try:
                    vec = row.get("embedding")
                    if vec is None or (isinstance(vec, float) and np.isnan(vec)):
                        failed += 1
                        continue

                    if isinstance(vec, str):
                        vec = ast.literal_eval(vec)
                    elif isinstance(vec, np.ndarray):
                        vec = vec.tolist()

                    if not isinstance(vec, list) or len(vec) == 0:
                        failed += 1
                        continue

                    batch.add_object(
                        properties={
                            "text":         str(row.get("text", ""))[:5000],
                            "grantId":      str(row.get("grant_id", "")),
                            "institute":    str(row.get("institute", "")),
                            "year":         int(row.get("year", 2024)),
                            "isFQHCFocused": bool(row.get("is_fqhc_focused", False)),
                            "sectionType":  str(row.get("chunk_type",
                                                row.get("section_type", "general"))),
                        },
                        vector=vec
                    )
                    total += 1

                    if total % 500 == 0:
                        print(f"    Imported {total}/{len(chunks_df)} chunks...")

                except Exception as e:
                    failed += 1

        print(f"  ✅ Imported {total} chunks ({failed} failed)")
        return total

    def _hybrid_search(self, collection, query: str, alpha: float,
                       query_vec: list, top_k: int) -> List[str]:
        try:
            from weaviate.classes.query import MetadataQuery

            if alpha > 0 and query_vec:
                resp = collection.query.hybrid(
                    query=query, alpha=alpha, vector=query_vec,
                    limit=top_k, return_metadata=MetadataQuery(score=True)
                )
            else:
                resp = collection.query.hybrid(
                    query=query, alpha=0.0,
                    limit=top_k, return_metadata=MetadataQuery(score=True)
                )

            return [
                obj.properties.get("grantId", "")
                for obj in resp.objects
                if obj.properties.get("grantId")
            ]
        except Exception as e:
            return []

    def _print_alpha_table(self, results_by_alpha: Dict):
        print(f"\n  {'─'*72}")
        print(f"  {'Alpha':<8} {'Label':<18} {'MAP':<8} {'MRR':<8} "
              f"{'P@5':<8} {'R@5':<8} {'nDCG@5':<8} {'Time(s)':<8}")
        print(f"  {'─'*72}")
        for alpha in sorted(results_by_alpha.keys()):
            r = results_by_alpha[alpha]
            print(
                f"  {alpha:<8.2f} {r['label']:<18} "
                f"{r['MAP']:<8.4f} {r['MRR']:<8.4f} "
                f"{r.get('P@5',{}).get('mean',0):<8.4f} "
                f"{r.get('R@5',{}).get('mean',0):<8.4f} "
                f"{r.get('nDCG@5',{}).get('mean',0):<8.4f} "
                f"{r['avg_retrieval_time_s']:<8.4f}"
            )
        print(f"  {'─'*72}")


# ============ PHASE 5 EVALUATOR (KNOWLEDGE GRAPH) ============

class Phase5Evaluator:
    """
    Evaluates knowledge graph augmented retrieval.
    Uses the pre-saved GML graph — no Weaviate needed for graph queries.
    Starts its own Weaviate only for vector search baseline comparison.
    """

    def __init__(self, data: DataLoader):
        self.data  = data
        self.graph = data.graph

    def evaluate(self, test_set: List[Dict], top_k: int = 10) -> Dict:
        print(f"\n{'='*70}")
        print("🧪 PHASE 5 EVALUATION: KNOWLEDGE GRAPH AUGMENTATION")
        print(f"{'='*70}")

        if self.graph is None:
            print("  ⚠️  Graph not loaded — skipping Phase 5 eval")
            return {}

        if self.data.model is None:
            print("  ⚠️  Model not loaded — skipping Phase 5 eval")
            return {}

        # Print graph summary
        node_types = defaultdict(int)
        for _, d in self.graph.nodes(data=True):
            node_types[d.get("type", "unknown")] += 1
        print(f"\n  Graph loaded: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        print(f"  Node types: {dict(node_types)}")

        # Start Weaviate for vector baseline
        vector_available = False
        client = None
        collection = None

        try:
            import weaviate
            client = weaviate.connect_to_embedded()
            if client.is_ready():
                p4_eval = Phase4Evaluator(self.data)
                p4_eval.client = client
                collection = p4_eval._setup_collection()
                if collection:
                    imported = p4_eval._import_chunks(collection)
                    vector_available = imported > 0
                    print(f"  ✅ Weaviate ready for vector baseline "
                          f"({imported} chunks)")
        except Exception as e:
            print(f"  ⚠️  Weaviate unavailable: {e} — "
                  f"using FAISS for vector baseline")

        vector_metrics_list   = []
        expanded_metrics_list = []
        graph_only_metrics_list = []
        times = {"vector": [], "expanded": [], "graph_only": []}

        for q in test_set:
            query   = q["query"]
            rel_ids = q["relevant_ids"]
            topic   = q["topic"]

            print(f"\n  📝 {q['query_id']} ({topic}): {query[:55]}...")

            # 1. Vector search (FAISS if Weaviate unavailable)
            start = time.time()
            if vector_available and collection:
                query_vec = self.data.model.encode(query).tolist()
                vector_ids = self._weaviate_vector_search(
                    collection, query_vec, top_k
                )
            else:
                vector_ids = self._faiss_search(query, top_k)
            times["vector"].append(time.time() - start)

            vm = IRMetrics.compute_all(vector_ids, rel_ids)
            vm["query_id"], vm["topic"] = q["query_id"], topic
            vector_metrics_list.append(vm)

            # 2. Graph-expanded search
            start = time.time()
            expanded_ids = self._graph_expand(vector_ids[:3], vector_ids, top_k)
            times["expanded"].append(time.time() - start)

            em = IRMetrics.compute_all(expanded_ids, rel_ids)
            em["query_id"], em["topic"] = q["query_id"], topic
            expanded_metrics_list.append(em)

            # 3. Graph-only (condition/topic node traversal)
            start = time.time()
            graph_ids = self._graph_only_search(topic)
            times["graph_only"].append(time.time() - start)

            gm = IRMetrics.compute_all(graph_ids, rel_ids)
            gm["query_id"], gm["topic"] = q["query_id"], topic
            graph_only_metrics_list.append(gm)

            # Per-query output
            improved = "✅ improved" if em["AP"] > vm["AP"] else "──"
            print(f"    Vector:         P@5={vm['P@5']:.3f}  AP={vm['AP']:.3f}")
            print(f"    Graph-expanded: P@5={em['P@5']:.3f}  AP={em['AP']:.3f}  {improved}")
            print(f"    Graph-only:     P@5={gm['P@5']:.3f}  AP={gm['AP']:.3f}")

        if client:
            client.close()

        # Aggregate
        vector_agg   = IRMetrics.aggregate(vector_metrics_list)
        expanded_agg = IRMetrics.aggregate(expanded_metrics_list)
        graph_agg    = IRMetrics.aggregate(graph_only_metrics_list)

        vector_agg["avg_retrieval_time_s"]   = round(float(np.mean(times["vector"])), 4)
        expanded_agg["avg_retrieval_time_s"] = round(float(np.mean(times["expanded"])), 4)
        graph_agg["avg_retrieval_time_s"]    = round(float(np.mean(times["graph_only"])), 4)

        map_delta = expanded_agg["MAP"] - vector_agg["MAP"]
        mrr_delta = expanded_agg["MRR"] - vector_agg["MRR"]

        # Summary table
        print(f"\n  {'─'*65}")
        print(f"  {'Method':<22} {'MAP':<8} {'MRR':<8} {'P@5':<8} "
              f"{'nDCG@5':<8} {'Time(s)':<8}")
        print(f"  {'─'*65}")
        for label, agg in [("Vector only", vector_agg),
                            ("Graph expanded", expanded_agg),
                            ("Graph only", graph_agg)]:
            print(
                f"  {label:<22} "
                f"{agg['MAP']:<8.4f} "
                f"{agg['MRR']:<8.4f} "
                f"{agg.get('P@5',{}).get('mean',0):<8.4f} "
                f"{agg.get('nDCG@5',{}).get('mean',0):<8.4f} "
                f"{agg['avg_retrieval_time_s']:<8.4f}s"
            )
        print(f"  {'─'*65}")
        print(f"  Graph expansion MAP Δ: {map_delta:+.4f} "
              f"({'✅ positive' if map_delta > 0 else '❌ negative'})")

        return {
            "vector_metrics":         vector_agg,
            "graph_expanded_metrics": expanded_agg,
            "graph_only_metrics":     graph_agg,
            "map_improvement":        round(map_delta, 4),
            "mrr_improvement":        round(mrr_delta, 4),
            "graph_structure": {
                "nodes":      self.graph.number_of_nodes(),
                "edges":      self.graph.number_of_edges(),
                "node_types": dict(node_types),
            },
            "per_query": {
                "vector":   vector_metrics_list,
                "expanded": expanded_metrics_list,
                "graph":    graph_only_metrics_list,
            },
        }

    def _weaviate_vector_search(self, collection, query_vec: list,
                                top_k: int) -> List[str]:
        try:
            from weaviate.classes.query import MetadataQuery
            resp = collection.query.near_vector(
                near_vector=query_vec, limit=top_k,
                return_metadata=MetadataQuery(distance=True),
                return_properties=["grantId"]
            )
            return [o.properties.get("grantId", "") for o in resp.objects
                    if o.properties.get("grantId")]
        except Exception:
            return []

    def _faiss_search(self, query: str, top_k: int) -> List[str]:
        """FAISS fallback when Weaviate unavailable"""
        if self.data.faiss_index is None or self.data.model is None:
            return []
        try:
            import faiss
            vec = self.data.model.encode([query])
            faiss.normalize_L2(vec)
            _, indices = self.data.faiss_index.search(vec, top_k)
            ids = []
            for idx in indices[0]:
                if 0 <= idx < len(self.data.chunks_df):
                    gid = self.data.chunks_df.iloc[idx].get("grant_id", "")
                    if gid:
                        ids.append(str(gid))
            return ids
        except Exception:
            return []

    def _graph_expand(self, seed_ids: List[str],
                      base_ids: List[str], top_k: int) -> List[str]:
        """Expand vector results via graph traversal"""
        expanded = list(base_ids)
        seen     = set(base_ids)

        for grant_id in seed_ids:
            if grant_id not in self.graph:
                continue

            for neighbor in self.graph.neighbors(grant_id):
                if neighbor in seen:
                    continue
                ntype = self.graph.nodes[neighbor].get("type", "")

                if ntype == "grant":
                    seen.add(neighbor)
                    expanded.append(neighbor)
                elif ntype in ["condition", "intervention", "population"]:
                    # Traverse one more hop to reach grants
                    for second in self.graph.neighbors(neighbor):
                        if second not in seen:
                            stype = self.graph.nodes[second].get("type", "")
                            if stype == "grant":
                                seen.add(second)
                                expanded.append(second)

        return expanded[:top_k * 2]

    def _graph_only_search(self, topic: str) -> List[str]:
        """Find grants via topic node traversal"""
        TOPIC_MAP = {
            "diabetes":          [("COND_diabetes",)],
            "CHW":               [("INT_CHW",)],
            "behavioral_health": [("COND_depression",), ("INT_integrated_care",)],
            "cancer_screening":  [("COND_cancer",)],
            "health_disparities":[("POP_low_income",), ("POP_Medicaid",)],
            "telehealth":        [("INT_telehealth",)],
            "substance_use":     [("COND_substance_use",)],
            "social_determinants":[("COND_social_determinants",)],
            "hypertension":      [("COND_hypertension",)],
            "HIV":               [("COND_HIV",)],
            "pediatric":         [("POP_pediatric",)],
            "Latino_health":     [("POP_Latino",)],
        }

        hub_nodes = TOPIC_MAP.get(topic, [])
        grant_ids = []
        seen = set()

        for (node_id,) in hub_nodes:
            if node_id not in self.graph:
                continue
            for neighbor in self.graph.neighbors(node_id):
                ntype = self.graph.nodes[neighbor].get("type", "")
                if ntype == "grant" and neighbor not in seen:
                    seen.add(neighbor)
                    grant_ids.append(neighbor)

        return grant_ids


# ============ VISUALIZATION ============

def visualize_results(p3: Dict, p4: Dict, p5: Dict):
    """Six-panel comparison figure for capstone report"""

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("RAG System Evaluation: Phase 3 → 4 → 5",
                 fontsize=16, fontweight="bold")

    def get_mean(d, key):
        v = d.get(key, {})
        return v.get("mean", 0) if isinstance(v, dict) else float(v or 0)

    # ---- 1. Cross-phase metric comparison ----
    ax = axes[0, 0]
    metric_keys = ["MAP", "MRR", "P@5", "nDCG@5"]

    p3_vals = [
        p3.get("MAP", 0),
        p3.get("MRR", 0),
        get_mean(p3, "P@5"),
        get_mean(p3, "nDCG@5"),
    ]
    p4_opt = p4.get("optimal_metrics", {}) if p4 else {}
    p4_vals = [
        p4_opt.get("MAP", 0),
        p4_opt.get("MRR", 0),
        get_mean(p4_opt, "P@5"),
        get_mean(p4_opt, "nDCG@5"),
    ]
    p5_exp = p5.get("graph_expanded_metrics", {}) if p5 else {}
    p5_vals = [
        p5_exp.get("MAP", 0),
        p5_exp.get("MRR", 0),
        get_mean(p5_exp, "P@5"),
        get_mean(p5_exp, "nDCG@5"),
    ]

    x     = np.arange(len(metric_keys))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (vals, label) in enumerate([(p3_vals, "Phase 3 (FAISS)"),
                                        (p4_vals, "Phase 4 (Hybrid)"),
                                        (p5_vals, "Phase 5 (Graph)")]):
        ax.bar(x + i * width, vals, width, label=label,
               color=colors[i], alpha=0.85)

    ax.set_title("Retrieval Metrics by Phase")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_keys)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ---- 2. Phase 4 alpha tuning ----
    ax = axes[0, 1]
    if p4 and "results_by_alpha" in p4:
        rba     = p4["results_by_alpha"]
        alphas  = sorted(rba.keys())
        maps    = [rba[a]["MAP"] for a in alphas]
        mrrs    = [rba[a]["MRR"] for a in alphas]
        p5s     = [get_mean(rba[a], "P@5") for a in alphas]

        ax.plot(alphas, maps, "o-",  label="MAP",  color="#4C72B0", lw=2)
        ax.plot(alphas, mrrs, "s-",  label="MRR",  color="#DD8452", lw=2)
        ax.plot(alphas, p5s,  "^-",  label="P@5",  color="#55A868", lw=2)
        ax.axvline(x=p4.get("optimal_alpha", 0.5), color="red",
                   linestyle="--", alpha=0.7,
                   label=f"Optimal α={p4.get('optimal_alpha','?')}")
        ax.set_xlabel("Alpha  (0=BM25  →  1=Vector)")
        ax.set_ylabel("Score")
        ax.set_title("Phase 4: Alpha Tuning Curve")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
    else:
        ax.text(0.5, 0.5, "Phase 4 data\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Phase 4: Alpha Tuning")

    # ---- 3. Phase 5 per-query AP improvement ----
    ax = axes[0, 2]
    if p5 and "per_query" in p5:
        v_aps  = [m["AP"] for m in p5["per_query"].get("vector",   [])]
        e_aps  = [m["AP"] for m in p5["per_query"].get("expanded", [])]
        q_ids  = [m["query_id"] for m in p5["per_query"].get("vector", [])]
        deltas = [e - v for e, v in zip(e_aps, v_aps)]
        bar_c  = ["#55A868" if d > 0 else "#C44E52" for d in deltas]

        ax.bar(range(len(deltas)), deltas, color=bar_c, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(q_ids)))
        ax.set_xticklabels(q_ids, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("AP Improvement")
        ax.set_title("Phase 5: Graph Expansion AP Δ per Query")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Phase 5 data\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Phase 5: Graph AP Improvement")

    # ---- 4. Graph node composition ----
    ax = axes[1, 0]
    if p5 and "graph_structure" in p5:
        nt = p5["graph_structure"].get("node_types", {})
        if nt:
            node_colors = {
                "grant": "#4C72B0", "institute": "#DD8452", "year": "#55A868",
                "condition": "#C44E52", "intervention": "#8172B2",
                "population": "#937860", "fqhc_hub": "#DA8BC3"
            }
            bar_c = [node_colors.get(t, "#999999") for t in nt.keys()]
            ax.bar(nt.keys(), nt.values(), color=bar_c, alpha=0.85)
            ax.set_title("Knowledge Graph Node Types")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(0.5, 0.5, "Graph data\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Graph Node Types")

    # ---- 5. Vector vs Graph-expanded comparison ----
    ax = axes[1, 1]
    if p5:
        methods = ["Vector only", "Graph expanded", "Graph only"]
        aggs    = [
            p5.get("vector_metrics", {}),
            p5.get("graph_expanded_metrics", {}),
            p5.get("graph_only_metrics", {}),
        ]
        maps_cmp = [a.get("MAP", 0) for a in aggs]
        mrrs_cmp = [a.get("MRR", 0) for a in aggs]

        x2    = np.arange(len(methods))
        w2    = 0.35
        ax.bar(x2 - w2/2, maps_cmp, w2, label="MAP", color="#4C72B0", alpha=0.85)
        ax.bar(x2 + w2/2, mrrs_cmp, w2, label="MRR", color="#DD8452", alpha=0.85)
        ax.set_xticks(x2)
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Score")
        ax.set_title("Phase 5: Search Method Comparison")
        ax.legend()
        ax.set_ylim(0, max(max(maps_cmp), max(mrrs_cmp)) * 1.2 + 0.01)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Phase 5 data\nnot available",
                ha="center", va="center", transform=ax.transAxes)

    # ---- 6. Summary table ----
    ax = axes[1, 2]
    ax.axis("off")

    p4_alpha = p4.get("optimal_alpha", "N/A") if p4 else "N/A"
    p5_delta = p5.get("map_improvement", "N/A") if p5 else "N/A"
    p5_delta_str = f"{p5_delta:+.4f}" if isinstance(p5_delta, float) else str(p5_delta)

    rows = [
        ["Phase 3 MAP",      f"{p3.get('MAP', 0):.4f}"],
        ["Phase 4 MAP",      f"{p4_opt.get('MAP', 0):.4f}"],
        ["Phase 5 MAP",      f"{p5_exp.get('MAP', 0):.4f}"],
        ["Optimal Alpha",    str(p4_alpha)],
        ["Graph MAP Δ",      p5_delta_str],
        ["Graph Nodes",      str(p5.get("graph_structure", {}).get("nodes", "N/A") if p5 else "N/A")],
        ["Graph Edges",      str(p5.get("graph_structure", {}).get("edges", "N/A") if p5 else "N/A")],
        ["Queries Evaluated",str(len(p5.get("per_query", {}).get("vector", [])) if p5 else "N/A")],
    ]

    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.9)
    ax.set_title("Evaluation Summary", pad=20)

    plt.tight_layout()
    out_path = Path(PATHS["output_dir"]) / "phase_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n💾 Visualization saved to {out_path}")
    plt.close()


# ============ MAIN ============

def main():
    print(f"\n{'='*70}")
    print("🚀 STARTING EVALUATION")
    print(f"{'='*70}")

    start_time = time.time()

    # 1. Load all pre-computed assets
    data = DataLoader()
    data.load_all()

    # 2. Build ground truth test set
    print("\n📋 BUILDING GROUND TRUTH TEST SET")
    print("-"*50)
    gt = GroundTruthBuilder(data.abstracts_df)
    test_set = gt.build()

    if not test_set:
        print("❌ No test queries built — check abstracts CSV has 'abstract' column")
        sys.exit(1)

    # 3. Phase 3 evaluation (FAISS)
    p3_eval   = Phase3Evaluator(data)
    p3_results = p3_eval.evaluate(test_set)

    # 4. Phase 4 evaluation (Weaviate hybrid)
    p4_eval    = Phase4Evaluator(data)
    p4_results = p4_eval.evaluate(test_set)

    # 5. Phase 5 evaluation (Knowledge graph)
    p5_eval    = Phase5Evaluator(data)
    p5_results = p5_eval.evaluate(test_set)

    # 6. Save all results
    print(f"\n{'='*70}")
    print("💾 SAVING RESULTS")
    print(f"{'='*70}")

    out = Path(PATHS["output_dir"])

    def save(data_dict, filename):
        # Strip per_query before saving (large)
        clean = {k: v for k, v in data_dict.items() if k != "per_query"}
        with open(out / filename, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"  ✅ {out / filename}")

    if p3_results:
        save(p3_results, "phase3_eval.json")
    if p4_results:
        alpha_summary = {
            str(a): {"MAP": r["MAP"], "MRR": r["MRR"]}
            for a, r in p4_results.get("results_by_alpha", {}).items()
        }
        save({**p4_results, "alpha_summary": alpha_summary}, "phase4_eval.json")
    if p5_results:
        save(p5_results, "phase5_eval.json")

    # Cross-phase comparison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "phase3": {"MAP": p3_results.get("MAP", 0), "MRR": p3_results.get("MRR", 0)},
        "phase4": {"MAP": p4_results.get("optimal_metrics", {}).get("MAP", 0),
                   "optimal_alpha": p4_results.get("optimal_alpha")},
        "phase5": {"MAP": p5_results.get("graph_expanded_metrics", {}).get("MAP", 0),
                   "map_improvement": p5_results.get("map_improvement")},
    }
    with open(out / "phase_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  ✅ {out / 'phase_comparison.json'}")

    # 7. Visualize
    print(f"\n📊 GENERATING VISUALIZATION")
    print("-"*50)
    visualize_results(p3_results, p4_results, p5_results)

    # 8. Final summary
    elapsed = round(time.time() - start_time, 1)
    print(f"\n{'='*70}")
    print(f"✅ EVALUATION COMPLETE  ({elapsed}s)")
    print(f"{'='*70}")
    print(f"\n📁 Output files in {PATHS['output_dir']}/:")
    print(f"   • test_set.json")
    print(f"   • phase3_eval.json")
    print(f"   • phase4_eval.json")
    print(f"   • phase5_eval.json")
    print(f"   • phase_comparison.json")
    print(f"   • phase_comparison.png  ← use in capstone report")

    print(f"\n📊 QUICK RESULTS:")
    if p3_results:
        print(f"   Phase 3 MAP: {p3_results.get('MAP', 0):.4f}")
    if p4_results:
        print(f"   Phase 4 MAP: {p4_results.get('optimal_metrics', {}).get('MAP', 0):.4f}  "
              f"(optimal α={p4_results.get('optimal_alpha')})")
    if p5_results:
        delta = p5_results.get('map_improvement', 0)
        print(f"   Phase 5 MAP: {p5_results.get('graph_expanded_metrics', {}).get('MAP', 0):.4f}  "
              f"(graph Δ={delta:+.4f})")


if __name__ == "__main__":
    main()