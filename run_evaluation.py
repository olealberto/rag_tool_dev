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

    @classmethod
    def compute_all(cls, retrieved, relevant_ids):
        relevant = set(relevant_ids)
        m = {}
        for k in [1, 3, 5, 10]:
            m[f"P@{k}"]    = cls.precision_at_k(retrieved, relevant, k)
            m[f"R@{k}"]    = cls.recall_at_k(retrieved, relevant, k)
            m[f"nDCG@{k}"] = cls.ndcg_at_k(retrieved, relevant, k)
        m["AP"]     = cls.average_precision(retrieved, relevant)
        m["RR"]     = cls.reciprocal_rank(retrieved, relevant)
        m["hits@5"] = sum(1 for r in retrieved[:5] if r in relevant)
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
        p = PATHS["chunks_no_emb"]
        if not os.path.exists(p):
            p = PATHS["chunks"]
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
            test_set.append({
                "query_id":     q["query_id"],
                "query":        q["query"],
                "topic":        q["topic"],
                "relevant_ids": rel[:50],
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
            _, indices = self.data.faiss_index.search(vec, top_k)
            retrieval_times.append(time.time() - t0)

            retrieved = []
            for idx in indices[0]:
                if 0 <= idx < len(self.data.chunks_df):
                    gid = str(self.data.chunks_df.iloc[idx].get("grant_id", ""))
                    if gid and gid != "nan":
                        retrieved.append(gid)

            m = IRMetrics.compute_all(retrieved, q["relevant_ids"])
            m["query_id"] = q["query_id"]; m["topic"] = q["topic"]
            query_metrics.append(m)

        agg = IRMetrics.aggregate(query_metrics)
        agg["avg_retrieval_time_s"] = round(float(np.mean(retrieval_times)), 4)
        agg["per_query"]    = query_metrics
        agg["section_aware"] = self.data.section_aware

        label = "section-aware ✨" if self.data.section_aware else "flat"
        print(f"  MAP:    {agg['MAP']:.4f}")
        print(f"  MRR:    {agg['MRR']:.4f}")
        print(f"  P@5:    {agg.get('P@5',{}).get('mean',0):.4f}")
        print(f"  nDCG@5: {agg.get('nDCG@5',{}).get('mean',0):.4f}")
        print(f"  Index:  {label}")
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
                    retrieved = self._hybrid_search(collection, q["query"], alpha, qvec, top_k)
                    retrieval_times.append(time.time() - t0)
                    m = IRMetrics.compute_all(retrieved, q["relevant_ids"])
                    m["query_id"] = q["query_id"]; m["topic"] = q["topic"]
                    query_metrics.append(m)

                agg = IRMetrics.aggregate(query_metrics)
                agg.update({"avg_retrieval_time_s": round(float(np.mean(retrieval_times)), 4),
                             "alpha": alpha, "label": label, "per_query": query_metrics})
                results_by_alpha[alpha] = agg
                print(f"    MAP={agg['MAP']:.4f}  MRR={agg['MRR']:.4f}  "
                      f"P@5={agg.get('P@5',{}).get('mean',0):.4f}  "
                      f"nDCG@5={agg.get('nDCG@5',{}).get('mean',0):.4f}  "
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
            return [o.properties.get("grantId", "") for o in resp.objects
                    if o.properties.get("grantId")]
        except: return []

    def _print_table(self, rba):
        print(f"\n  {'─'*72}")
        print(f"  {'Alpha':<8} {'Label':<18} {'MAP':<8} {'MRR':<8} "
              f"{'P@5':<8} {'R@5':<8} {'nDCG@5':<8} {'Time(s)'}")
        print(f"  {'─'*72}")
        for alpha in sorted(rba.keys()):
            r = rba[alpha]
            print(f"  {alpha:<8.2f} {r['label']:<18} "
                  f"{r['MAP']:<8.4f} {r['MRR']:<8.4f} "
                  f"{r.get('P@5',{}).get('mean',0):<8.4f} "
                  f"{r.get('R@5',{}).get('mean',0):<8.4f} "
                  f"{r.get('nDCG@5',{}).get('mean',0):<8.4f} "
                  f"{r['avg_retrieval_time_s']:.4f}s")
        print(f"  {'─'*72}")


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

    def evaluate(self, test_set, top_k=10):
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

        # Weaviate for vector baseline (optional)
        client = collection = None
        vector_available = False
        p4 = Phase4Evaluator(self.data)
        client = p4._connect_weaviate()
        if client:
            p4.client = client
            collection = p4._setup_collection()
            if collection:
                imported = p4._import_chunks(collection)
                vector_available = imported > 0

        v_list, e_list, g_list = [], [], []
        times = {"vector": [], "expanded": [], "graph_only": []}

        for q in test_set:
            print(f"\n  📝 {q['query_id']} ({q['topic']}): {q['query'][:55]}...")

            # Vector
            t0 = time.time()
            if vector_available and collection:
                vec = self.data.model.encode(q["query"]).tolist()
                v_ids = self._weaviate_search(collection, vec, top_k)
            else:
                v_ids = self._faiss_search(q["query"], top_k)
            times["vector"].append(time.time() - t0)
            vm = IRMetrics.compute_all(v_ids, q["relevant_ids"])
            vm["query_id"] = q["query_id"]; vm["topic"] = q["topic"]
            v_list.append(vm)

            # Graph-expanded
            t0 = time.time()
            e_ids = self._expand(v_ids[:3], v_ids, top_k)
            times["expanded"].append(time.time() - t0)
            em = IRMetrics.compute_all(e_ids, q["relevant_ids"])
            em["query_id"] = q["query_id"]; em["topic"] = q["topic"]
            e_list.append(em)

            # Graph-only
            t0 = time.time()
            g_ids = self._graph_only(q["topic"])
            times["graph_only"].append(time.time() - t0)
            gm = IRMetrics.compute_all(g_ids, q["relevant_ids"])
            gm["query_id"] = q["query_id"]; gm["topic"] = q["topic"]
            g_list.append(gm)

            improved = "✅ improved" if em["AP"] > vm["AP"] else "──"
            print(f"    Vector:         P@5={vm['P@5']:.3f}  AP={vm['AP']:.3f}")
            print(f"    Graph-expanded: P@5={em['P@5']:.3f}  AP={em['AP']:.3f}  {improved}")
            print(f"    Graph-only:     P@5={gm['P@5']:.3f}  AP={gm['AP']:.3f}")

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

        print(f"\n  {'─'*65}")
        print(f"  {'Method':<22} {'MAP':<8} {'MRR':<8} {'P@5':<8} {'nDCG@5':<8} {'Time(s)'}")
        print(f"  {'─'*65}")
        for label, agg in [("Vector only", vagg),
                            ("Graph expanded", eagg),
                            ("Graph only", gagg)]:
            print(f"  {label:<22} {agg['MAP']:<8.4f} {agg['MRR']:<8.4f} "
                  f"{agg.get('P@5',{}).get('mean',0):<8.4f} "
                  f"{agg.get('nDCG@5',{}).get('mean',0):<8.4f} "
                  f"{agg['avg_retrieval_time_s']:.4f}s")
        print(f"  {'─'*65}")
        print(f"  Graph expansion MAP Δ: {map_delta:+.4f} "
              f"({'✅ positive' if map_delta > 0 else '❌ negative'})")

        return {
            "vector_metrics":         vagg,
            "graph_expanded_metrics": eagg,
            "graph_only_metrics":     gagg,
            "map_improvement":        round(map_delta, 4),
            "mrr_improvement":        round(mrr_delta, 4),
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
            return [o.properties.get("grantId","") for o in resp.objects
                    if o.properties.get("grantId")]
        except: return []

    def _faiss_search(self, query, top_k):
        if self.data.faiss_index is None or self.data.model is None: return []
        try:
            import faiss
            vec = self.data.model.encode([query])
            faiss.normalize_L2(vec)
            _, indices = self.data.faiss_index.search(vec, top_k)
            ids = []
            for idx in indices[0]:
                if 0 <= idx < len(self.data.chunks_df):
                    gid = str(self.data.chunks_df.iloc[idx].get("grant_id",""))
                    if gid and gid != "nan": ids.append(gid)
            return ids
        except: return []

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
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("RAG System Evaluation: Phase 3 → 4 → 5",
                 fontsize=16, fontweight="bold")

    def gm(d, key):
        v = d.get(key, {})
        return v.get("mean", 0) if isinstance(v, dict) else float(v or 0)

    # 1. Cross-phase metrics
    ax = axes[0, 0]
    mk = ["MAP", "MRR", "P@5", "nDCG@5"]
    p4o = p4.get("optimal_metrics",{}) if p4 else {}
    p5e = p5.get("graph_expanded_metrics",{}) if p5 else {}
    vals = [
        [p3.get("MAP",0), p3.get("MRR",0), gm(p3,"P@5"), gm(p3,"nDCG@5")],
        [p4o.get("MAP",0), p4o.get("MRR",0), gm(p4o,"P@5"), gm(p4o,"nDCG@5")],
        [p5e.get("MAP",0), p5e.get("MRR",0), gm(p5e,"P@5"), gm(p5e,"nDCG@5")],
    ]
    labels = ["Phase 3 (FAISS)", "Phase 4 (Hybrid)", "Phase 5 (Graph)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    x, w  = np.arange(len(mk)), 0.25
    for i, (v, lbl, c) in enumerate(zip(vals, labels, colors)):
        ax.bar(x + i*w, v, w, label=lbl, color=c, alpha=0.85)
    ax.set_title("Retrieval Metrics by Phase"); ax.set_xticks(x+w)
    ax.set_xticklabels(mk); ax.set_ylabel("Score"); ax.set_ylim(0,1)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # 2. Alpha tuning
    ax = axes[0, 1]
    if p4 and "results_by_alpha" in p4:
        rba = p4["results_by_alpha"]; alphas = sorted(rba.keys())
        ax.plot(alphas, [rba[a]["MAP"] for a in alphas], "o-", label="MAP", lw=2)
        ax.plot(alphas, [rba[a]["MRR"] for a in alphas], "s-", label="MRR", lw=2)
        ax.plot(alphas, [gm(rba[a],"P@5") for a in alphas], "^-", label="P@5", lw=2)
        ax.axvline(p4.get("optimal_alpha",0.5), color="red", ls="--", alpha=0.7,
                   label=f"Optimal α={p4.get('optimal_alpha','?')}")
        ax.set_xlabel("Alpha  (0=BM25  →  1=Vector)")
        ax.set_ylabel("Score"); ax.set_title("Phase 4: Alpha Tuning")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(-0.05, 1.05)
    else:
        ax.text(0.5,0.5,"Phase 4 data\nnot available",ha="center",va="center",
                transform=ax.transAxes)
        ax.set_title("Phase 4: Alpha Tuning")

    # 3. Per-query AP delta
    ax = axes[0, 2]
    if p5 and "per_query" in p5:
        v_aps = [m["AP"] for m in p5["per_query"].get("vector",[])]
        e_aps = [m["AP"] for m in p5["per_query"].get("expanded",[])]
        q_ids = [m["query_id"] for m in p5["per_query"].get("vector",[])]
        deltas = [e - v for e, v in zip(e_aps, v_aps)]
        ax.bar(range(len(deltas)), deltas,
               color=["#55A868" if d > 0 else "#C44E52" for d in deltas], alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(q_ids))); ax.set_xticklabels(q_ids, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("AP Improvement"); ax.set_title("Phase 5: Graph Expansion AP Δ per Query")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5,0.5,"Phase 5 data\nnot available",ha="center",va="center",
                transform=ax.transAxes)
        ax.set_title("Phase 5: Graph AP Improvement")

    # 4. Graph node types
    ax = axes[1, 0]
    if p5 and "graph_structure" in p5:
        nt = p5["graph_structure"].get("node_types", {})
        nc = {"grant":"#4C72B0","institute":"#DD8452","year":"#55A868",
              "condition":"#C44E52","intervention":"#8172B2",
              "population":"#937860","fqhc_hub":"#DA8BC3"}
        if nt:
            ax.bar(nt.keys(), nt.values(),
                   color=[nc.get(t,"#999") for t in nt.keys()], alpha=0.85)
            ax.set_title("Knowledge Graph Node Types"); ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(0.5,0.5,"Graph data\nnot available",ha="center",va="center",
                transform=ax.transAxes)
        ax.set_title("Graph Node Types")

    # 5. Method comparison
    ax = axes[1, 1]
    if p5:
        methods = ["Vector only","Graph expanded","Graph only"]
        aggs = [p5.get("vector_metrics",{}), p5.get("graph_expanded_metrics",{}),
                p5.get("graph_only_metrics",{})]
        maps_c = [a.get("MAP",0) for a in aggs]
        mrrs_c = [a.get("MRR",0) for a in aggs]
        x2, w2 = np.arange(len(methods)), 0.35
        ax.bar(x2-w2/2, maps_c, w2, label="MAP", color="#4C72B0", alpha=0.85)
        ax.bar(x2+w2/2, mrrs_c, w2, label="MRR", color="#DD8452", alpha=0.85)
        ax.set_xticks(x2); ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Score"); ax.set_title("Phase 5: Method Comparison"); ax.legend()
        top = max(max(maps_c), max(mrrs_c)) * 1.2 + 0.01
        ax.set_ylim(0, top if top > 0.01 else 0.1); ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5,0.5,"Phase 5 data\nnot available",ha="center",va="center",
                transform=ax.transAxes)

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")
    delta = p5.get("map_improvement",0) if p5 else 0
    rows = [
        ["Phase 3 MAP",     f"{p3.get('MAP',0):.4f}"],
        ["Index type",      "section-aware" if p3.get("section_aware") else "flat"],
        ["Phase 4 MAP",     f"{p4o.get('MAP',0):.4f}"],
        ["Optimal α",       str(p4.get("optimal_alpha","N/A") if p4 else "N/A")],
        ["Phase 5 MAP",     f"{p5e.get('MAP',0):.4f}"],
        ["Graph MAP Δ",     f"{delta:+.4f}" if isinstance(delta,float) else "N/A"],
        ["Graph nodes",     str(p5.get("graph_structure",{}).get("nodes","N/A") if p5 else "N/A")],
        ["Queries eval'd",  str(len(p5["per_query"].get("vector",[]) if p5 and "per_query" in p5 else []))],
    ]
    tbl = ax.table(cellText=rows, colLabels=["Metric","Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.9)
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

    p3 = Phase3Evaluator(data).evaluate(test_set)
    p4 = Phase4Evaluator(data).evaluate(test_set)
    p5 = Phase5Evaluator(data).evaluate(test_set)

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

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "phase3": {"MAP": p3.get("MAP",0), "MRR": p3.get("MRR",0),
                   "section_aware": p3.get("section_aware",False)},
        "phase4": {"MAP": p4.get("optimal_metrics",{}).get("MAP",0) if p4 else 0,
                   "optimal_alpha": p4.get("optimal_alpha") if p4 else None},
        "phase5": {"MAP": p5.get("graph_expanded_metrics",{}).get("MAP",0) if p5 else 0,
                   "map_improvement": p5.get("map_improvement") if p5 else None},
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
    if p3: print(f"   Phase 3 MAP: {p3.get('MAP',0):.4f}  "
                 f"({'section-aware ✨' if p3.get('section_aware') else 'flat'} FAISS)")
    if p4: print(f"   Phase 4 MAP: {p4.get('optimal_metrics',{}).get('MAP',0):.4f}  "
                 f"(optimal α={p4.get('optimal_alpha')})")
    if p5:
        d = p5.get("map_improvement",0)
        print(f"   Phase 5 MAP: {p5.get('graph_expanded_metrics',{}).get('MAP',0):.4f}  "
              f"(graph Δ={d:+.4f})")


if __name__ == "__main__":
    main()