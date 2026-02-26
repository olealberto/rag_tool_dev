# ============================================================================
# 📁 run_evaluation.py - STANDALONE EVALUATION SCRIPT
# ============================================================================

"""
STANDALONE EVALUATION FOR PHASES 3, 4, AND 5
Run with: !python run_evaluation.py

Loads everything from disk - no recomputing embeddings or rebuilding graphs.

What it loads:
    - ./phase2_output/nih_research_abstracts.csv
    - ./phase3_results/document_chunks_with_embeddings.csv
    - ./phase3_results/faiss/global.index     ← section-aware (Phase 3 updated)
    - ./phase3_results/faiss_index.bin         ← flat index fallback
    - ./phase5_knowledge_graph.gml / graph.pkl

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

import os, sys, json, time, ast, pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────

PATHS = {
    "abstracts":        "./phase2_output/nih_research_abstracts.csv",
    "chunks":           "./phase3_results/document_chunks_with_embeddings.csv",
    "chunks_no_emb":    "./phase3_results/document_chunks.csv",
    # FIX 1: prefer section-aware global index, fall back to flat
    "faiss_global":     "./phase3_results/faiss/global.index",
    "faiss_flat":       "./phase3_results/faiss_index.bin",
    "embeddings":       "./phase3_results/chunk_embeddings.npy",
    "graph_pkl":        "./phase5_graph_store/graph.pkl",
    "graph_gml":        "./phase5_knowledge_graph.gml",
    "output_dir":       "./evaluation",
}

Path(PATHS["output_dir"]).mkdir(exist_ok=True)

# ── Ground truth queries ──────────────────────────────────────────────────────

SEED_QUERIES = [
    {"query_id": "Q01", "topic": "diabetes",
     "query": "diabetes prevention and management in community health settings",
     "keywords": ["diabetes", "diabetic", "hba1c", "glycemic", "insulin"]},
    {"query_id": "Q02", "topic": "CHW",
     "query": "community health worker interventions for underserved populations",
     "keywords": ["community health worker", "chw", "promotora", "lay health"]},
    {"query_id": "Q03", "topic": "behavioral_health",
     "query": "behavioral health integration in primary care",
     "keywords": ["behavioral health", "mental health", "depression", "anxiety",
                  "integrated care"]},
    {"query_id": "Q04", "topic": "cancer_screening",
     "query": "cancer screening programs in community health centers",
     "keywords": ["cancer", "screening", "mammography", "colonoscopy", "cervical"]},
    {"query_id": "Q05", "topic": "health_disparities",
     "query": "health disparities in minority and low income populations",
     "keywords": ["health disparit", "minority", "low-income", "underserved",
                  "racial", "ethnic"]},
    {"query_id": "Q06", "topic": "telehealth",
     "query": "telehealth and digital health interventions for chronic disease",
     "keywords": ["telehealth", "telemedicine", "digital health", "mobile health",
                  "mhealth"]},
    {"query_id": "Q07", "topic": "substance_use",
     "query": "substance use disorder treatment and opioid addiction",
     "keywords": ["substance use", "opioid", "addiction", "naloxone",
                  "buprenorphine"]},
    {"query_id": "Q08", "topic": "social_determinants",
     "query": "social determinants of health screening and referral programs",
     "keywords": ["social determinants", "food insecurity", "housing", "sdoh",
                  "social needs"]},
    {"query_id": "Q09", "topic": "hypertension",
     "query": "hypertension control and cardiovascular disease prevention",
     "keywords": ["hypertension", "blood pressure", "cardiovascular",
                  "heart disease"]},
    {"query_id": "Q10", "topic": "HIV",
     "query": "HIV prevention and treatment in high risk populations",
     "keywords": ["hiv", "aids", "prep", "antiretroviral"]},
    {"query_id": "Q11", "topic": "pediatric",
     "query": "pediatric health interventions in community settings",
     "keywords": ["pediatric", "children", "adolescent", "youth", "child health"]},
    {"query_id": "Q12", "topic": "Latino_health",
     "query": "Latino and Hispanic health promotion programs",
     "keywords": ["latino", "hispanic", "latinx", "spanish speaking", "promotora"]},
]


# ── IR Metrics ────────────────────────────────────────────────────────────────

class IRMetrics:

    @staticmethod
    def precision_at_k(retrieved, relevant, k):
        return sum(1 for r in retrieved[:k] if r in relevant) / k if k else 0.0

    @staticmethod
    def recall_at_k(retrieved, relevant, k):
        return sum(1 for r in retrieved[:k] if r in relevant) / len(relevant) \
               if relevant else 0.0

    @staticmethod
    def average_precision(retrieved, relevant):
        if not relevant or not retrieved: return 0.0
        hits = score = 0.0
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                hits += 1
                score += hits / (i + 1)
        return score / len(relevant)

    @staticmethod
    def reciprocal_rank(retrieved, relevant):
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved, relevant, k):
        def dcg(lst, rel, k):
            return sum(1.0 / np.log2(i+2)
                       for i, d in enumerate(lst[:k]) if d in rel)
        actual = dcg(retrieved, relevant, k)
        ideal  = dcg(list(relevant)[:k], relevant, k)
        return actual / ideal if ideal else 0.0

    @staticmethod
    def r_precision(retrieved, relevant):
        """Precision at R where R = |relevant|. Robust to relevance set size."""
        r = len(relevant)
        if not r: return 0.0
        return sum(1 for d in retrieved[:r] if d in relevant) / r

    @staticmethod
    def hit_rate_at_k(retrieved, relevant, k):
        """Binary: did at least one relevant doc appear in top-k?"""
        return 1.0 if any(d in relevant for d in retrieved[:k]) else 0.0

    @staticmethod
    def ndcg_graded(retrieved, relevant_ids_ordered, k):
        """
        NDCG with graded relevance based on keyword match count.
        relevant_ids_ordered: list of (grant_id, grade) tuples, grade 1-4.
        Falls back to binary if grades not provided.
        """
        grade_map = {}
        if relevant_ids_ordered and isinstance(relevant_ids_ordered[0], tuple):
            grade_map = {gid: g for gid, g in relevant_ids_ordered}
        else:
            grade_map = {gid: 1 for gid in relevant_ids_ordered}

        def dcg(lst, grades, k):
            return sum(grades.get(d, 0) / np.log2(i + 2)
                       for i, d in enumerate(lst[:k]))

        actual = dcg(retrieved, grade_map, k)
        ideal_docs = sorted(grade_map.keys(),
                            key=lambda d: grade_map[d], reverse=True)
        ideal = dcg(ideal_docs, grade_map, k)
        return actual / ideal if ideal else 0.0

    @staticmethod
    def score_weighted_precision(retrieved, scores, relevant, k):
        """Mean retrieval score of relevant hits in top-k. Shows confidence when correct."""
        hits = [scores[i] for i, r in enumerate(retrieved[:k]) if r in relevant]
        return float(np.mean(hits)) if hits else 0.0

    @classmethod
    def compute_all(cls, retrieved, relevant_ids, scores=None):
        relevant = set(relevant_ids)
        # Normalise scores to [0,1] if provided; default 1.0 per position
        if scores is None or len(scores) == 0:
            scores = [1.0] * len(retrieved)
        scores = list(scores)[:len(retrieved)]
        # Normalise: hybrid scores are [0,2], FAISS cosine is [-1,1] after L2 norm → [0,1]
        smax = max(scores) if scores else 1.0
        smin = min(scores) if scores else 0.0
        rng  = smax - smin if smax != smin else 1.0
        norm_scores = [(s - smin) / rng for s in scores]

        m = {}
        for k in [1, 3, 5, 10, 20]:
            m[f"P@{k}"]      = cls.precision_at_k(retrieved, relevant, k)
            m[f"R@{k}"]      = cls.recall_at_k(retrieved, relevant, k)
            m[f"nDCG@{k}"]   = cls.ndcg_at_k(retrieved, relevant, k)
            m[f"HR@{k}"]     = cls.hit_rate_at_k(retrieved, relevant, k)
        m["AP"]        = cls.average_precision(retrieved, relevant)
        m["RR"]        = cls.reciprocal_rank(retrieved, relevant)
        m["R-Prec"]    = cls.r_precision(retrieved, relevant)
        m["hits@5"]    = sum(1 for r in retrieved[:5] if r in relevant)
        m["hits@20"]   = sum(1 for r in retrieved[:20] if r in relevant)
        m["n_relevant"] = len(relevant)

        # Score-based metrics (use raw scores, not normalised, for interpretability)
        m["avg_score_top5"]     = float(np.mean(scores[:5]))   if scores[:5]   else 0.0
        m["avg_score_top20"]    = float(np.mean(scores[:20]))  if scores[:20]  else 0.0
        m["avg_score_relevant"] = float(np.mean(
            [scores[i] for i, r in enumerate(retrieved) if r in relevant]
        )) if any(r in relevant for r in retrieved) else 0.0
        m["avg_score_irrelevant"] = float(np.mean(
            [scores[i] for i, r in enumerate(retrieved) if r not in relevant]
        )) if any(r not in relevant for r in retrieved) else 0.0
        # Score gap: how much higher does the system score relevant vs irrelevant?
        m["score_gap"] = m["avg_score_relevant"] - m["avg_score_irrelevant"]
        return m

    @staticmethod
    def aggregate(query_metrics):
        if not query_metrics: return {}
        keys = [k for k in query_metrics[0] if isinstance(query_metrics[0][k], float)]
        agg = {}
        for k in keys:
            vals = [m[k] for m in query_metrics]
            agg[k] = {"mean": round(float(np.mean(vals)), 4),
                      "std":  round(float(np.std(vals)),  4),
                      "min":  round(float(np.min(vals)),  4),
                      "max":  round(float(np.max(vals)),  4)}
        agg["MAP"] = agg.get("AP", {}).get("mean", 0)
        agg["MRR"] = agg.get("RR", {}).get("mean", 0)
        return agg


# ── Data loader ───────────────────────────────────────────────────────────────

class DataLoader:

    def __init__(self):
        self.abstracts_df    = None
        self.chunks_df       = None
        self.embeddings      = None
        self.faiss_index     = None
        self.graph           = None
        self.model           = None
        self.section_aware   = False   # True when using new per-section FAISS

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
        p = PATHS["abstracts"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"Abstracts not found: {p}")
        self.abstracts_df = pd.read_csv(p)
        print(f"  ✅ Abstracts:   {len(self.abstracts_df)} rows  ({p})")

    def _load_chunks(self):
        # IMPORTANT: must load same file/order as FAISS index was built from
        # chunks_with_embeddings (3986 rows) == FAISS index order
        # chunks_no_emb (4316 rows) is a DIFFERENT order — misaligns FAISS lookups
        p = PATHS["chunks"]   # 3986 rows, same order as FAISS
        if not os.path.exists(p):
            p = PATHS["chunks_no_emb"]
        if not os.path.exists(p):
            raise FileNotFoundError("Chunks not found. Run phase3_document_rag.py first")
        self.chunks_df = pd.read_csv(p)
        if "embedding" in self.chunks_df.columns:
            self.chunks_df = self.chunks_df.drop(columns=["embedding"])
        print(f"  ✅ Chunks:      {len(self.chunks_df)} rows  ({p})")

    def _load_embeddings(self):
        p = PATHS["embeddings"]
        if not os.path.exists(p):
            print(f"  ⚠️  Embeddings .npy not found — vector search uses FAISS only")
            return
        self.embeddings = np.load(p)
        print(f"  ✅ Embeddings:  {self.embeddings.shape}  ({p})")

    def _load_faiss(self):
        try:
            import faiss
        except ImportError:
            print("  ⚠️  faiss-cpu not installed — Phase 3 eval unavailable")
            return

        # FIX 1: try section-aware global index first, fall back to flat
        if os.path.exists(PATHS["faiss_global"]):
            self.faiss_index  = faiss.read_index(PATHS["faiss_global"])
            self.section_aware = True
            print(f"  ✅ FAISS:       {self.faiss_index.ntotal} vectors  "
                  f"(section-aware ✨, {PATHS['faiss_global']})")
        elif os.path.exists(PATHS["faiss_flat"]):
            self.faiss_index = faiss.read_index(PATHS["faiss_flat"])
            print(f"  ✅ FAISS:       {self.faiss_index.ntotal} vectors  "
                  f"(flat, {PATHS['faiss_flat']})")
        else:
            print(f"  ⚠️  No FAISS index found — Phase 3 eval unavailable")

    def _load_graph(self):
        # Try pickle (faster) before GML
        if os.path.exists(PATHS["graph_pkl"]):
            with open(PATHS["graph_pkl"], "rb") as f:
                self.graph = pickle.load(f)
            print(f"  ✅ Graph:       {self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges  (pickle)")
        elif os.path.exists(PATHS["graph_gml"]):
            self.graph = nx.read_gml(PATHS["graph_gml"])
            print(f"  ✅ Graph:       {self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges  (GML)")
        else:
            print(f"  ⚠️  Graph not found — Phase 5 eval unavailable")

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
            print(f"  ⏳ Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            print(f"  ✅ Model loaded")
        except Exception as e:
            print(f"  ⚠️  Could not load model: {e}")


# ── Ground truth builder ──────────────────────────────────────────────────────

class GroundTruthBuilder:

    def __init__(self, abstracts_df):
        self.df = abstracts_df.copy()
        text_col = "abstract" if "abstract" in self.df.columns else self.df.columns[2]
        self.df["_text"] = self.df[text_col].fillna("").str.lower()
        self.id_col = "grant_id" if "grant_id" in self.df.columns else self.df.columns[0]

    def build(self, min_relevant=2):
        cache = Path(PATHS["output_dir"]) / "test_set.json"
        if cache.exists():
            with open(cache) as f:
                test_set = json.load(f)
            # Rebuild if relevant_ids were capped differently before
            if test_set and len(test_set[0].get("relevant_ids", [])) > 20:
                print(f"  🔄 Stale cache (relevant_ids > 20) — rebuilding...")
                cache.unlink()
            else:
                print(f"  📦 Loaded cached test set: {len(test_set)} queries")
                return test_set

        print(f"\n  Building test set from {len(self.df)} abstracts...")
        test_set = []
        for q in SEED_QUERIES:
            rel = self.df[
                self.df["_text"].apply(lambda t: any(kw in t for kw in q["keywords"]))
            ][self.id_col].tolist()
            if len(rel) < min_relevant:
                print(f"    ⚠️  {q['query_id']} ({q['topic']}): only {len(rel)} — skipping")
                continue
            # Cap at 20 relevant — retrieval top_k=10 so P@5/MAP are meaningful
            # 50+ relevant in a 1797-grant corpus makes MAP artificially low
            test_set.append({
                "query_id":     q["query_id"],
                "query":        q["query"],
                "topic":        q["topic"],
                "relevant_ids": rel[:20],
                "relevant_count": len(rel),
            })
            print(f"    ✅ {q['query_id']} ({q['topic']}): {len(rel)} relevant")

        with open(cache, "w") as f:
            json.dump(test_set, f, indent=2)
        print(f"  💾 Saved test set: {len(test_set)} queries")
        return test_set


# ── Phase 3 evaluator (FAISS) ─────────────────────────────────────────────────

class Phase3Evaluator:

    def __init__(self, data: DataLoader):
        self.data = data

    def evaluate(self, test_set, top_k=10):
        print(f"\n{'='*70}")
        print("🧪 PHASE 3 EVALUATION: FAISS VECTOR SEARCH")
        print(f"{'='*70}")

        if self.data.faiss_index is None or self.data.model is None:
            print("  ⚠️  FAISS index or model not available — skipping"); return {}

        import faiss
        query_metrics, retrieval_times = [], []

        for q in test_set:
            vec = self.data.model.encode([q["query"]])
            faiss.normalize_L2(vec)
            t0 = time.time()
            raw_scores, indices = self.data.faiss_index.search(vec, top_k)
            retrieval_times.append(time.time() - t0)

            retrieved, scores = [], []
            for score, idx in zip(raw_scores[0], indices[0]):
                if 0 <= idx < len(self.data.chunks_df):
                    gid = str(self.data.chunks_df.iloc[idx].get("grant_id", ""))
                    if gid and gid != "nan":
                        retrieved.append(gid)
                        scores.append(float(score))

            m = IRMetrics.compute_all(retrieved, q["relevant_ids"], scores)
            m["query_id"] = q["query_id"]; m["topic"] = q["topic"]
            query_metrics.append(m)

        agg = IRMetrics.aggregate(query_metrics)
        agg["avg_retrieval_time_s"] = round(float(np.mean(retrieval_times)), 4)
        agg["per_query"]    = query_metrics
        agg["section_aware"] = self.data.section_aware

        label = "section-aware ✨" if self.data.section_aware else "flat"
        print(f"  MAP:                  {agg['MAP']:.4f}")
        print(f"  MRR:                  {agg['MRR']:.4f}")
        print(f"  R-Prec:               {agg.get('R-Prec',{}).get('mean',0):.4f}  ← robust to relevance set size")
        print(f"  P@5:                  {agg.get('P@5',{}).get('mean',0):.4f}")
        print(f"  R@20:                 {agg.get('R@20',{}).get('mean',0):.4f}")
        print(f"  HR@5:                 {agg.get('HR@5',{}).get('mean',0):.4f}  ← hit rate")
        print(f"  HR@20:                {agg.get('HR@20',{}).get('mean',0):.4f}")
        print(f"  nDCG@5:               {agg.get('nDCG@5',{}).get('mean',0):.4f}")
        print(f"  Avg score (top-5):    {agg.get('avg_score_top5',{}).get('mean',0):.4f}  ← retrieval confidence")
        print(f"  Avg score (relevant): {agg.get('avg_score_relevant',{}).get('mean',0):.4f}")
        print(f"  Score gap (rel-irrel):{agg.get('score_gap',{}).get('mean',0):+.4f}  ← system discriminability")
        print(f"  Index:    {label}")
        print(f"  Avg retrieval time: {agg['avg_retrieval_time_s']:.4f}s")
        return agg


# ── Phase 4 evaluator (Weaviate hybrid) ──────────────────────────────────────

class Phase4Evaluator:

    def __init__(self, data: DataLoader):
        self.data   = data
        self.client = None

    def _connect_weaviate(self):
        """FIX 2: robust multi-port connection matching query_pipeline.py"""
        import weaviate
        for ports in [{"port": 8079, "grpc_port": 50050},
                      {"port": 8080, "grpc_port": 50051}]:
            try:
                client = weaviate.connect_to_embedded(**ports)
                if client.is_ready():
                    return client
            except Exception as e:
                err = str(e).lower()
                if "already" in err or "listening" in err:
                    try:
                        client = weaviate.connect_to_local(**ports)
                        if client.is_ready():
                            return client
                    except: pass
        return None

    def evaluate(self, test_set, alpha_values=None, top_k=10):
        if alpha_values is None:
            alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        print(f"\n{'='*70}")
        print("🧪 PHASE 4 EVALUATION: WEAVIATE HYBRID SEARCH")
        print(f"{'='*70}")

        print("  🚀 Starting Weaviate embedded...")
        self.client = self._connect_weaviate()
        if self.client is None:
            print("  ❌ Weaviate failed — skipping Phase 4"); return {}
        print("  ✅ Weaviate ready")

        try:
            collection = self._setup_collection()
            if collection is None: return {}
            imported = self._import_chunks(collection)
            if imported == 0: print("  ❌ No chunks imported"); return {}

            print(f"\n  Evaluating {len(alpha_values)} alpha values "
                  f"across {len(test_set)} queries...")
            results_by_alpha = {}

            for alpha in alpha_values:
                label = ("BM25 only" if alpha == 0.0
                         else "vector only" if alpha == 1.0
                         else f"hybrid α={alpha}")
                print(f"\n  📊 {label}...")
                query_metrics, retrieval_times = [], []

                for q in test_set:
                    qvec = (self.data.model.encode(q["query"]).tolist()
                            if self.data.model and alpha > 0 else None)
                    t0 = time.time()
                    retrieved, scores = self._hybrid_search(collection, q["query"], alpha, qvec, top_k)
                    retrieval_times.append(time.time() - t0)
                    m = IRMetrics.compute_all(retrieved, q["relevant_ids"], scores)
                    m["query_id"] = q["query_id"]; m["topic"] = q["topic"]
                    m["retrieved_ids"] = retrieved  # store for Phase5 reuse
                    m["retrieved_scores"] = scores  # store scores too
                    query_metrics.append(m)

                agg = IRMetrics.aggregate(query_metrics)
                agg.update({"avg_retrieval_time_s": round(float(np.mean(retrieval_times)), 4),
                             "alpha": alpha, "label": label, "per_query": query_metrics})
                results_by_alpha[alpha] = agg
                print(f"    MAP={agg['MAP']:.4f}  MRR={agg['MRR']:.4f}  "
                      f"P@5={agg.get('P@5',{}).get('mean',0):.4f}  "
                      f"HR@5={agg.get('HR@5',{}).get('mean',0):.4f}  "
                      f"AvgScore={agg.get('avg_score_top5',{}).get('mean',0):.4f}  "
                      f"ScoreGap={agg.get('score_gap',{}).get('mean',0):+.4f}  "
                      f"time={agg['avg_retrieval_time_s']:.4f}s")

            optimal = max(results_by_alpha, key=lambda a: results_by_alpha[a]["MAP"])
            print(f"\n  🎯 Optimal alpha: {optimal} "
                  f"(MAP={results_by_alpha[optimal]['MAP']:.4f})")
            self._print_table(results_by_alpha)

            return {"results_by_alpha": results_by_alpha,
                    "optimal_alpha":    optimal,
                    "optimal_metrics":  results_by_alpha[optimal],
                    "test_set_size":    len(test_set)}
        finally:
            if self.client:
                try: self.client.close()
                except: pass
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
                    Property(name="text",          data_type=DataType.TEXT),
                    Property(name="grantId",       data_type=DataType.TEXT),
                    Property(name="institute",     data_type=DataType.TEXT),
                    Property(name="year",          data_type=DataType.INT),
                    Property(name="isFQHCFocused", data_type=DataType.BOOL),
                    Property(name="sectionType",   data_type=DataType.TEXT),  # FIX 3
                ]
            )
            return self.client.collections.get("EvalGrant")
        except Exception as e:
            print(f"  ❌ Schema failed: {e}"); return None

    def _import_chunks(self, collection):
        print(f"\n  📤 Loading embeddings from disk...")
        p = PATHS["chunks"]
        if not os.path.exists(p):
            print(f"  ❌ Not found: {p}"); return 0
        print(f"  ⏳ Reading {p}...")
        df = pd.read_csv(p)
        print(f"  ✅ Loaded {len(df)} chunks")

        total = failed = 0
        with collection.batch.fixed_size(batch_size=100) as batch:
            for idx, row in df.iterrows():
                try:
                    vec = row.get("embedding")
                    if vec is None or (isinstance(vec, float) and np.isnan(vec)):
                        failed += 1; continue
                    if isinstance(vec, str):
                        try:    vec = json.loads(vec.replace("'", '"'))
                        except: vec = ast.literal_eval(vec)
                    elif isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    if not isinstance(vec, list) or not vec:
                        failed += 1; continue

                    # FIX 3: prefer section_type (Phase 3 updated), fall back to chunk_type
                    section = str(row.get("section_type", row.get("chunk_type", "general")))

                    batch.add_object(
                        properties={
                            "text":          str(row.get("text", ""))[:5000],
                            "grantId":       str(row.get("grant_id", "")),
                            "institute":     str(row.get("institute", "")),
                            "year":          int(row.get("year", 2024)
                                                if pd.notna(row.get("year", 2024)) else 2024),
                            "isFQHCFocused": bool(row.get("is_fqhc_focused", False)),
                            "sectionType":   section,
                        },
                        vector=vec
                    )
                    total += 1
                    if total % 500 == 0:
                        print(f"    Imported {total}/{len(df)} chunks...")
                except Exception as e:
                    failed += 1

        print(f"  ✅ Imported {total} chunks ({failed} failed)")
        return total

    def _hybrid_search(self, collection, query, alpha, query_vec, top_k):
        try:
            from weaviate.classes.query import MetadataQuery
            kwargs = dict(query=query, alpha=alpha, limit=top_k,
                          return_metadata=MetadataQuery(score=True))
            if alpha > 0 and query_vec:
                kwargs["vector"] = query_vec
            resp = collection.query.hybrid(**kwargs)
            ids, scores = [], []
            for o in resp.objects:
                gid = o.properties.get("grantId", "")
                if gid:
                    ids.append(gid)
                    scores.append(float(o.metadata.score or 0.0))
            return ids, scores
        except: return [], []

    def _print_table(self, rba):
        print(f"\n  {'─'*90}")
        print(f"  {'Alpha':<8} {'Label':<18} {'MAP':<8} {'MRR':<8} "
              f"{'R-Prec':<8} {'P@5':<8} {'R@20':<8} {'HR@5':<8} {'nDCG@5':<8} {'Time(s)'}")
        print(f"  {'─'*90}")
        for alpha in sorted(rba.keys()):
            r = rba[alpha]
            print(f"  {alpha:<8.2f} {r['label']:<18} "
                  f"{r['MAP']:<8.4f} {r['MRR']:<8.4f} "
                  f"{r.get('R-Prec',{}).get('mean',0):<8.4f} "
                  f"{r.get('P@5',{}).get('mean',0):<8.4f} "
                  f"{r.get('R@20',{}).get('mean',0):<8.4f} "
                  f"{r.get('HR@5',{}).get('mean',0):<8.4f} "
                  f"{r.get('nDCG@5',{}).get('mean',0):<8.4f} "
                  f"{r['avg_retrieval_time_s']:.4f}s")
        print(f"  {'─'*90}")


# ── Phase 5 evaluator (knowledge graph) ──────────────────────────────────────

class Phase5Evaluator:

    TOPIC_MAP = {
        "diabetes":           ["COND_diabetes"],
        "CHW":                ["INT_CHW"],
        "behavioral_health":  ["COND_depression", "INT_integrated_care"],
        "cancer_screening":   ["COND_cancer"],
        "health_disparities": ["POP_low_income", "POP_Medicaid"],
        "telehealth":         ["INT_telehealth"],
        "substance_use":      ["COND_substance_use"],
        "social_determinants":["COND_social_determinants"],
        "hypertension":       ["COND_hypertension"],
        "HIV":                ["COND_HIV"],
        "pediatric":          ["POP_pediatric"],
        "Latino_health":      ["POP_Latino"],
    }

    def __init__(self, data: DataLoader):
        self.data  = data
        self.graph = data.graph

    def evaluate(self, test_set, top_k=10, p4_vector_results=None):
        print(f"\n{'='*70}")
        print("🧪 PHASE 5 EVALUATION: KNOWLEDGE GRAPH AUGMENTATION")
        print(f"{'='*70}")

        if self.graph is None:
            print("  ⚠️  Graph not loaded — skipping"); return {}
        if self.data.model is None:
            print("  ⚠️  Model not loaded — skipping"); return {}

        node_types = defaultdict(int)
        for _, d in self.graph.nodes(data=True):
            node_types[d.get("type", "unknown")] += 1
        print(f"\n  Graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        print(f"  Node types: {dict(node_types)}")

        # Build lookup from p4 vector-only results (avoids re-importing 3986 chunks)
        p4_lookup = {}
        p4_score_lookup = {}
        if p4_vector_results:
            for qm in p4_vector_results:
                if qm.get("retrieved_ids"):
                    p4_lookup[qm["query_id"]]       = qm["retrieved_ids"]
                    p4_score_lookup[qm["query_id"]] = qm.get("retrieved_scores", [])
            print(f"  ✅ Reusing Phase 4 vector results for {len(p4_lookup)} queries")

        # Only spin up Weaviate if p4 results unavailable
        client = collection = None
        vector_available = bool(p4_lookup)
        if not vector_available:
            p4ev = Phase4Evaluator(self.data)
            client = p4ev._connect_weaviate()
            if client:
                p4ev.client = client
                collection = p4ev._setup_collection()
                if collection:
                    imported = p4ev._import_chunks(collection)
                    vector_available = imported > 0

        v_list, e_list, g_list = [], [], []
        times = {"vector": [], "expanded": [], "graph_only": []}

        for q in test_set:
            print(f"\n  📝 {q['query_id']} ({q['topic']}): {q['query'][:55]}...")

            # Vector — prefer p4 reuse (with stored scores), fall back to fresh search
            t0 = time.time()
            if q["query_id"] in p4_lookup:
                v_ids    = p4_lookup[q["query_id"]]
                v_scores = p4_score_lookup.get(q["query_id"], [])
            elif vector_available and collection:
                vec = self.data.model.encode(q["query"]).tolist()
                v_ids, v_scores = self._weaviate_search(collection, vec, top_k)
            else:
                v_ids, v_scores = self._faiss_search(q["query"], top_k)
            times["vector"].append(time.time() - t0)
            vm = IRMetrics.compute_all(v_ids, q["relevant_ids"], v_scores)
            vm["query_id"] = q["query_id"]; vm["topic"] = q["topic"]
            v_list.append(vm)

            # Graph-expanded (no retrieval scores — positional proxy)
            t0 = time.time()
            e_ids = self._expand(v_ids[:3], v_ids, top_k)
            times["expanded"].append(time.time() - t0)
            em = IRMetrics.compute_all(e_ids, q["relevant_ids"])
            em["query_id"] = q["query_id"]; em["topic"] = q["topic"]
            e_list.append(em)

            # Graph-only (no retrieval scores)
            t0 = time.time()
            g_ids = self._graph_only(q["topic"])
            times["graph_only"].append(time.time() - t0)
            gm_obj = IRMetrics.compute_all(g_ids, q["relevant_ids"])
            gm_obj["query_id"] = q["query_id"]; gm_obj["topic"] = q["topic"]
            g_list.append(gm_obj)

            improved = "✅ improved" if em["AP"] > vm["AP"] else "──"
            print(f"    Vector:         P@5={vm['P@5']:.3f}  AP={vm['AP']:.3f}")
            print(f"    Graph-expanded: P@5={em['P@5']:.3f}  AP={em['AP']:.3f}  {improved}")
            print(f"    Graph-only:     P@5={gm_obj['P@5']:.3f}  AP={gm_obj['AP']:.3f}")

        if client:
            try: client.close()
            except: pass

        vagg = IRMetrics.aggregate(v_list)
        eagg = IRMetrics.aggregate(e_list)
        gagg = IRMetrics.aggregate(g_list)
        for agg, key in [(vagg,"vector"),(eagg,"expanded"),(gagg,"graph_only")]:
            agg["avg_retrieval_time_s"] = round(float(np.mean(times[key])), 4)

        map_delta = eagg["MAP"] - vagg["MAP"]
        mrr_delta = eagg["MRR"] - vagg["MRR"]

        rprec_delta = eagg.get("R-Prec",{}).get("mean",0) - vagg.get("R-Prec",{}).get("mean",0)
        hr5_delta   = eagg.get("HR@5",{}).get("mean",0)   - vagg.get("HR@5",{}).get("mean",0)
        r20_delta   = eagg.get("R@20",{}).get("mean",0)   - vagg.get("R@20",{}).get("mean",0)

        print(f"\n  {'─'*100}")
        print(f"  {'Method':<22} {'MAP':<8} {'MRR':<8} {'R-Prec':<8} {'HR@5':<8} {'R@20':<8} {'AvgSc@5':<9} {'ScoreGap':<10} Time")
        print(f"  {'─'*100}")
        for label, agg in [("Vector only", vagg),
                            ("Graph expanded", eagg),
                            ("Graph only", gagg)]:
            print(f"  {label:<22} {agg['MAP']:<8.4f} {agg['MRR']:<8.4f} "
                  f"{agg.get('R-Prec',{}).get('mean',0):<8.4f} "
                  f"{agg.get('HR@5',{}).get('mean',0):<8.4f} "
                  f"{agg.get('R@20',{}).get('mean',0):<8.4f} "
                  f"{agg.get('avg_score_top5',{}).get('mean',0):<9.4f} "
                  f"{agg.get('score_gap',{}).get('mean',0):<+10.4f} "
                  f"{agg['avg_retrieval_time_s']:.4f}s")
        print(f"  {'─'*100}")
        print(f"  Graph expansion MAP Δ:    {map_delta:+.4f} "
              f"({'✅ positive' if map_delta > 0 else '❌ negative'})")
        print(f"  Graph expansion R-Prec Δ: {rprec_delta:+.4f}")
        print(f"  Graph expansion HR@5 Δ:   {hr5_delta:+.4f}")
        print(f"  Graph expansion R@20 Δ:   {r20_delta:+.4f}")

        return {
            "vector_metrics":         vagg,
            "graph_expanded_metrics": eagg,
            "graph_only_metrics":     gagg,
            "map_improvement":        round(map_delta, 4),
            "mrr_improvement":        round(mrr_delta, 4),
            "rprec_improvement":      round(rprec_delta, 4),
            "hr5_improvement":        round(hr5_delta, 4),
            "r20_improvement":        round(r20_delta, 4),
            "graph_structure":        {"nodes": self.graph.number_of_nodes(),
                                       "edges": self.graph.number_of_edges(),
                                       "node_types": dict(node_types)},
            "per_query": {"vector": v_list, "expanded": e_list, "graph": g_list},
        }

    def _weaviate_search(self, collection, vec, top_k):
        try:
            from weaviate.classes.query import MetadataQuery
            resp = collection.query.near_vector(
                near_vector=vec, limit=top_k,
                return_metadata=MetadataQuery(distance=True),
                return_properties=["grantId"]
            )
            ids, scores = [], []
            for o in resp.objects:
                gid = o.properties.get("grantId", "")
                if gid:
                    ids.append(gid)
                    # distance → similarity: score = 1 - distance/2 (cosine)
                    dist = o.metadata.distance or 0.0
                    scores.append(float(1.0 - dist / 2.0))
            return ids, scores
        except: return [], []

    def _faiss_search(self, query, top_k):
        if self.data.faiss_index is None or self.data.model is None: return [], []
        try:
            import faiss
            vec = self.data.model.encode([query])
            faiss.normalize_L2(vec)
            raw_scores, indices = self.data.faiss_index.search(vec, top_k)
            ids, scores = [], []
            for score, idx in zip(raw_scores[0], indices[0]):
                if 0 <= idx < len(self.data.chunks_df):
                    gid = str(self.data.chunks_df.iloc[idx].get("grant_id",""))
                    if gid and gid != "nan":
                        ids.append(gid)
                        scores.append(float(score))
            return ids, scores
        except: return [], []

    def _expand(self, seed_ids, base_ids, top_k):
        expanded, seen = list(base_ids), set(base_ids)
        for gid in seed_ids:
            if gid not in self.graph: continue
            for nbr in self.graph.neighbors(gid):
                if nbr in seen: continue
                ntype = self.graph.nodes[nbr].get("type","")
                if ntype == "grant":
                    seen.add(nbr); expanded.append(nbr)
                elif ntype in ["condition","intervention","population"]:
                    for nbr2 in self.graph.neighbors(nbr):
                        if nbr2 not in seen and \
                           self.graph.nodes[nbr2].get("type","") == "grant":
                            seen.add(nbr2); expanded.append(nbr2)
        return expanded[:top_k*2]

    def _graph_only(self, topic):
        hubs = self.TOPIC_MAP.get(topic, [])
        ids, seen = [], set()
        for node_id in hubs:
            if node_id not in self.graph: continue
            for nbr in self.graph.neighbors(node_id):
                if nbr not in seen and \
                   self.graph.nodes[nbr].get("type","") == "grant":
                    seen.add(nbr); ids.append(nbr)
        return ids


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_results(p3, p4, p5):
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle("RAG System Evaluation: Phase 3 → 4 → 5",
                 fontsize=16, fontweight="bold")

    def gm(d, key):
        v = d.get(key, {})
        return v.get("mean", 0) if isinstance(v, dict) else float(v or 0)

    p4o = p4.get("optimal_metrics", {}) if p4 else {}
    p5e = p5.get("graph_expanded_metrics", {}) if p5 else {}

    # ── 1. Cross-phase: MAP / R-Prec / HR@5 / R@20 ───────────────────────────
    ax = axes[0, 0]
    mk = ["MAP", "R-Prec", "HR@5", "R@20"]
    vals = [
        [p3.get("MAP",0),      gm(p3,"R-Prec"),  gm(p3,"HR@5"),  gm(p3,"R@20")],
        [p4o.get("MAP",0),     gm(p4o,"R-Prec"), gm(p4o,"HR@5"), gm(p4o,"R@20")],
        [p5e.get("MAP",0),     gm(p5e,"R-Prec"), gm(p5e,"HR@5"), gm(p5e,"R@20")],
    ]
    labels = ["Phase 3 (FAISS)", "Phase 4 (Hybrid α=0.25)", "Phase 5 (Graph expanded)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    x, w = np.arange(len(mk)), 0.25
    for i, (v, lbl, c) in enumerate(zip(vals, labels, colors)):
        ax.bar(x + i*w, v, w, label=lbl, color=c, alpha=0.85)
    ax.set_title("Key Metrics by Phase\n(R-Prec & HR@5 robust to relevance set size)")
    ax.set_xticks(x + w); ax.set_xticklabels(mk)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    # ── 2. Alpha tuning: MAP + R-Prec + HR@5 ─────────────────────────────────
    ax = axes[0, 1]
    if p4 and "results_by_alpha" in p4:
        rba = p4["results_by_alpha"]; alphas = sorted(rba.keys())
        ax.plot(alphas, [rba[a]["MAP"] for a in alphas],          "o-", label="MAP",    lw=2)
        ax.plot(alphas, [gm(rba[a],"R-Prec") for a in alphas],   "s-", label="R-Prec", lw=2)
        ax.plot(alphas, [gm(rba[a],"HR@5")   for a in alphas],   "^-", label="HR@5",   lw=2)
        ax.plot(alphas, [gm(rba[a],"R@20")   for a in alphas],   "D-", label="R@20",   lw=2, alpha=0.7)
        ax.axvline(p4.get("optimal_alpha", 0.5), color="red", ls="--", alpha=0.7,
                   label=f"Optimal α={p4.get('optimal_alpha','?')}")
        ax.set_xlabel("Alpha  (0=BM25  →  1=Vector)")
        ax.set_ylabel("Score"); ax.set_title("Phase 4: Alpha Tuning\n(all metrics)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_xlim(-0.05, 1.05)
    else:
        ax.text(0.5, 0.5, "Phase 4 data\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Phase 4: Alpha Tuning")

    # ── 3. Per-query R-Prec delta (graph expansion) ───────────────────────────
    ax = axes[0, 2]
    if p5 and "per_query" in p5:
        v_rp  = [m.get("R-Prec", 0) for m in p5["per_query"].get("vector", [])]
        e_rp  = [m.get("R-Prec", 0) for m in p5["per_query"].get("expanded", [])]
        q_ids = [m["query_id"]       for m in p5["per_query"].get("vector", [])]
        topics= [m["topic"]          for m in p5["per_query"].get("vector", [])]
        deltas = [e - v for e, v in zip(e_rp, v_rp)]
        bars = ax.bar(range(len(deltas)), deltas,
                      color=["#55A868" if d > 0 else "#C44E52" for d in deltas],
                      alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(topics)))
        ax.set_xticklabels(topics, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("R-Precision Improvement")
        ax.set_title("Phase 5: Graph Expansion R-Prec Δ per Query")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Phase 5 data\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Phase 5: Graph R-Prec Improvement")

    # ── 4a. Hit Rate @K across phases ─────────────────────────────────────────
    ax = axes[1, 0]
    ks = [1, 3, 5, 10, 20]
    p3_hr  = [gm(p3,  f"HR@{k}") for k in ks]
    p4_hr  = [gm(p4o, f"HR@{k}") for k in ks]
    p5_hr  = [gm(p5e, f"HR@{k}") for k in ks]
    if any(p3_hr + p4_hr + p5_hr):
        ax.plot(ks, p3_hr, "o-", label="Phase 3 (FAISS)",   color="#4C72B0", lw=2)
        ax.plot(ks, p4_hr, "s-", label="Phase 4 (Hybrid)",  color="#DD8452", lw=2)
        ax.plot(ks, p5_hr, "^-", label="Phase 5 (Graph+)",  color="#55A868", lw=2)

        # Annotate final values
        for vals, c in [(p3_hr,"#4C72B0"),(p4_hr,"#DD8452"),(p5_hr,"#55A868")]:
            ax.annotate(f"{vals[-1]:.2f}", xy=(20, vals[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        color=c, fontsize=7, va="center")
        ax.set_xlabel("K"); ax.set_ylabel("Hit Rate")
        ax.set_title("Hit Rate@K by Phase\n(≥1 relevant in top-K)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05); ax.set_xticks(ks)
    else:
        ax.text(0.5, 0.5, "HR@K data\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Hit Rate@K by Phase")

    # ── 5. Retrieval score confidence by phase ────────────────────────────────
    ax = axes[1, 1]
    phases      = ["Phase 3\n(FAISS)", "Phase 4\n(Hybrid α=0.25)", "Phase 5\n(Graph+)"]
    aggs_score  = [p3, p4o, p5e]
    top5_scores = [gm(a, "avg_score_top5")     for a in aggs_score]
    rel_scores  = [gm(a, "avg_score_relevant")  for a in aggs_score]
    irrel_scores= [gm(a, "avg_score_irrelevant") for a in aggs_score]
    x2 = np.arange(len(phases))
    w2 = 0.25
    if any(top5_scores + rel_scores):
        b1 = ax.bar(x2 - w2, top5_scores,  w2, label="Avg score top-5",    color="#4C72B0", alpha=0.85)
        b2 = ax.bar(x2,       rel_scores,   w2, label="Avg score relevant", color="#55A868", alpha=0.85)
        b3 = ax.bar(x2 + w2,  irrel_scores, w2, label="Avg score irrelevant", color="#C44E52", alpha=0.85)
        ax.set_xticks(x2); ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("Retrieval Score")
        ax.set_title("Retrieval Score Confidence by Phase\n(higher relevant vs irrelevant = better discrimination)")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        # Annotate score gap
        for i, (rs, ir) in enumerate(zip(rel_scores, irrel_scores)):
            if rs and ir:
                gap = rs - ir
                ax.annotate(f"gap={gap:+.3f}", xy=(i, max(rs, ir)),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=7, color="#333")
    else:
        ax.text(0.5, 0.5, "Score data\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Retrieval Score Confidence")

    # ── 6. Summary table ──────────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    rows = [
        ["Metric",           "Ph3 FAISS", "Ph4 Hybrid", "Ph5 Graph+"],
        ["MAP",              f"{p3.get('MAP',0):.4f}",
                             f"{p4o.get('MAP',0):.4f}",
                             f"{p5e.get('MAP',0):.4f}"],
        ["R-Precision",      f"{gm(p3,'R-Prec'):.4f}",
                             f"{gm(p4o,'R-Prec'):.4f}",
                             f"{gm(p5e,'R-Prec'):.4f}"],
        ["HR@5",             f"{gm(p3,'HR@5'):.4f}",
                             f"{gm(p4o,'HR@5'):.4f}",
                             f"{gm(p5e,'HR@5'):.4f}"],
        ["R@20",             f"{gm(p3,'R@20'):.4f}",
                             f"{gm(p4o,'R@20'):.4f}",
                             f"{gm(p5e,'R@20'):.4f}"],
        ["MRR",              f"{p3.get('MRR',0):.4f}",
                             f"{p4o.get('MRR',0):.4f}",
                             f"{p5e.get('MRR',0):.4f}"],
        ["nDCG@5",           f"{gm(p3,'nDCG@5'):.4f}",
                             f"{gm(p4o,'nDCG@5'):.4f}",
                             f"{gm(p5e,'nDCG@5'):.4f}"],
        ["AvgScore@5",       f"{gm(p3,'avg_score_top5'):.4f}",
                             f"{gm(p4o,'avg_score_top5'):.4f}",
                             f"{gm(p5e,'avg_score_top5'):.4f}"],
        ["Score Gap",        f"{gm(p3,'score_gap'):+.4f}",
                             f"{gm(p4o,'score_gap'):+.4f}",
                             f"{gm(p5e,'score_gap'):+.4f}"],
        ["Optimal α",        "—",
                             str(p4.get("optimal_alpha","N/A") if p4 else "N/A"),
                             "—"],
        ["Graph MAP Δ",      "—", "—",
                             f"{p5.get('map_improvement',0):+.4f}" if p5 else "N/A"],
        ["Graph HR@5 Δ",     "—", "—",
                             f"{p5.get('hr5_improvement',0):+.4f}" if p5 else "N/A"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.3, 1.65)
    # Highlight header row
    for j in range(4):
        tbl[0, j].set_facecolor("#4C72B0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Evaluation Summary", pad=20)

    plt.tight_layout()
    out = Path(PATHS["output_dir"]) / "phase_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\n💾 Visualization saved to {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print("🚀 STARTING EVALUATION")
    print(f"{'='*70}")
    t0 = time.time()

    data = DataLoader()
    data.load_all()

    print("\n📋 BUILDING GROUND TRUTH TEST SET")
    print("-"*50)
    test_set = GroundTruthBuilder(data.abstracts_df).build()
    if not test_set:
        print("❌ No test queries built"); sys.exit(1)

    EVAL_TOP_K = 20   # retrieve more candidates so MAP/P@5 are meaningful
    p3 = Phase3Evaluator(data).evaluate(test_set, top_k=EVAL_TOP_K)
    p4 = Phase4Evaluator(data).evaluate(test_set, top_k=EVAL_TOP_K)
    # Reuse Phase 4 optimal-alpha per-query results as Phase 5 vector baseline
    # avoids re-importing 3986 chunks a second time
    p4_rba = p4.get("results_by_alpha", {}) if p4 else {}
    best_a  = p4.get("optimal_alpha", 0.25) if p4 else 0.25
    p4_per_query = p4_rba.get(best_a, {}).get("per_query", [])
    p5 = Phase5Evaluator(data).evaluate(test_set, top_k=EVAL_TOP_K,
                                         p4_vector_results=p4_per_query)

    print(f"\n{'='*70}")
    print("💾 SAVING RESULTS")
    print(f"{'='*70}")
    out = Path(PATHS["output_dir"])

    def save(d, fname):
        clean = {k: v for k, v in d.items() if k != "per_query"}
        with open(out/fname, "w") as f: json.dump(clean, f, indent=2)
        print(f"  ✅ {out/fname}")

    if p3: save(p3, "phase3_eval.json")
    if p4: save(p4, "phase4_eval.json")
    if p5: save(p5, "phase5_eval.json")

    def _gm(d, key):
        v = d.get(key, {})
        return v.get("mean", 0) if isinstance(v, dict) else float(v or 0)

    p4o = p4.get("optimal_metrics", {}) if p4 else {}
    p5e = p5.get("graph_expanded_metrics", {}) if p5 else {}

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "phase3": {
            "MAP":            p3.get("MAP", 0),
            "MRR":            p3.get("MRR", 0),
            "R-Prec":         _gm(p3, "R-Prec"),
            "HR@5":           _gm(p3, "HR@5"),
            "R@20":           _gm(p3, "R@20"),
            "avg_score_top5": _gm(p3, "avg_score_top5"),
            "score_gap":      _gm(p3, "score_gap"),
            "section_aware":  p3.get("section_aware", False),
        },
        "phase4": {
            "MAP":    p4o.get("MAP", 0),
            "MRR":    p4o.get("MRR", 0),
            "R-Prec": _gm(p4o, "R-Prec"),
            "HR@5":   _gm(p4o, "HR@5"),
            "R@20":   _gm(p4o, "R@20"),
            "optimal_alpha": p4.get("optimal_alpha") if p4 else None,
        },
        "phase5": {
            "MAP":    p5e.get("MAP", 0),
            "MRR":    p5e.get("MRR", 0),
            "R-Prec": _gm(p5e, "R-Prec"),
            "HR@5":   _gm(p5e, "HR@5"),
            "R@20":   _gm(p5e, "R@20"),
            "map_improvement":   p5.get("map_improvement")   if p5 else None,
            "rprec_improvement": p5.get("rprec_improvement") if p5 else None,
            "hr5_improvement":   p5.get("hr5_improvement")   if p5 else None,
            "r20_improvement":   p5.get("r20_improvement")   if p5 else None,
        },
    }
    with open(out/"phase_comparison.json","w") as f: json.dump(comparison, f, indent=2)
    print(f"  ✅ {out/'phase_comparison.json'}")

    print(f"\n📊 GENERATING VISUALIZATION")
    visualize_results(p3, p4, p5)

    print(f"\n{'='*70}")
    print(f"✅ EVALUATION COMPLETE  ({round(time.time()-t0,1)}s)")
    print(f"{'='*70}")
    print(f"\n📁 Output files in {PATHS['output_dir']}/:")
    for f in ["test_set.json","phase3_eval.json","phase4_eval.json",
              "phase5_eval.json","phase_comparison.json","phase_comparison.png"]:
        print(f"   • {f}")

    print(f"\n📊 QUICK RESULTS:")
    print(f"  {'Metric':<12} {'Phase 3':>10} {'Phase 4':>10} {'Phase 5':>10}  {'Graph Δ':>10}")
    print(f"  {'─'*58}")
    metrics_summary = [
        ("MAP",           p3.get("MAP",0),              p4o.get("MAP",0),              p5e.get("MAP",0),              p5.get("map_improvement",0)   if p5 else 0),
        ("R-Prec",        _gm(p3,"R-Prec"),             _gm(p4o,"R-Prec"),             _gm(p5e,"R-Prec"),             p5.get("rprec_improvement",0) if p5 else 0),
        ("HR@5",          _gm(p3,"HR@5"),               _gm(p4o,"HR@5"),               _gm(p5e,"HR@5"),               p5.get("hr5_improvement",0)   if p5 else 0),
        ("R@20",          _gm(p3,"R@20"),               _gm(p4o,"R@20"),               _gm(p5e,"R@20"),               p5.get("r20_improvement",0)   if p5 else 0),
        ("MRR",           p3.get("MRR",0),              p4o.get("MRR",0),              p5e.get("MRR",0),              0),
        ("nDCG@5",        _gm(p3,"nDCG@5"),             _gm(p4o,"nDCG@5"),             _gm(p5e,"nDCG@5"),             0),
        ("AvgScore@5",    _gm(p3,"avg_score_top5"),     _gm(p4o,"avg_score_top5"),     _gm(p5e,"avg_score_top5"),     0),
        ("ScoreGap",      _gm(p3,"score_gap"),          _gm(p4o,"score_gap"),          _gm(p5e,"score_gap"),          0),
    ]
    for name, v3, v4, v5, delta in metrics_summary:
        d_str = f"{delta:+.4f}" if delta else "   —  "
        print(f"  {name:<12} {v3:>10.4f} {v4:>10.4f} {v5:>10.4f}  {d_str:>10}")
    if p4: print(f"\n  Optimal α = {p4.get('optimal_alpha')}  "
                 f"({'section-aware ✨' if p3.get('section_aware') else 'flat'} FAISS index)")


if __name__ == "__main__":
    main()