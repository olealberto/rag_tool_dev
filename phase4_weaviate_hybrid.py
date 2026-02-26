# ============================================================================
# 📁 phase4_weaviate_hybrid.py - WEAVIATE V4 HYBRID RAG (SECTION-AWARE)
# ============================================================================

"""
PHASE 4: WEAVIATE V4 HYBRID SEARCH WITH SECTION-AWARE QUERIES

Upgrades from original:
  - sectionType property in schema (maps from Phase 3 section_type)
  - Section-prefixed query embedding (matches Phase 3 chunk embedding style)
  - Optional section_filter on hybrid search
  - Robust Weaviate connection (embedded → fallback to local, two port configs)
  - Proper evaluation: P@5, MAP, MRR across alpha values
  - Evaluation saved to phase4_results.json for paper
"""

print("=" * 70)
print("🎯 PHASE 4: WEAVIATE V4 HYBRID-RAG (Section-Aware)")
print("=" * 70)

import sys, os, ast, json, time
sys.path.append('.')

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    import weaviate
    from weaviate.classes.query import MetadataQuery, Filter
    from weaviate.classes.config import Property, DataType
    print(f"✅ Weaviate v{weaviate.__version__} loaded")
except ImportError:
    print("📦 Run: pip install weaviate-client")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️  Run: pip install sentence-transformers")

# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION_NAME  = "GrantChunk"
EMBEDDING_MODEL  = "pritamdeka/S-PubMedBert-MS-MARCO"
CHUNKS_CSV       = "./phase3_results/document_chunks_with_embeddings.csv"
RESULTS_JSON     = "./phase4_results.json"

# Must match Phase 3 SECTION_PREFIXES exactly
SECTION_PREFIXES = {
    "specific_aims":    "specific aims:",
    "significance":     "significance and innovation:",
    "innovation":       "innovation:",
    "approach":         "approach and methods:",
    "methods":          "research methods:",
    "background":       "background:",
    "project_summary":  "project summary:",
    "project_narrative":"project narrative:",
    "human_subjects":   "human subjects:",
    "bibliography":     "references:",
    "preamble":         "introduction:",
    "general":          "grant text:",
    "semantic_chunk":   "grant section:",
}

HIGH_VALUE_SECTIONS = {
    "specific_aims", "significance", "innovation",
    "approach", "methods", "background", "project_summary",
}


# ── Weaviate connection ───────────────────────────────────────────────────────

class WeaviateManager:
    """
    Robust Weaviate connection for Colab:
    Tries embedded on port 8079, falls back to 8080,
    falls back to connecting to already-running instance.
    """

    def __init__(self):
        self.client     = None
        self.collection = None

    def start(self) -> bool:
        print("\n🔌 Starting Weaviate...")

        for ports in [{"port": 8079, "grpc_port": 50050},
                      {"port": 8080, "grpc_port": 50051}]:
            try:
                self.client = weaviate.connect_to_embedded(**ports)
                print(f"  ✅ Embedded Weaviate started on port {ports['port']}")
                break
            except Exception as e:
                err = str(e).lower()
                if "already" in err or "listening" in err:
                    try:
                        self.client = weaviate.connect_to_local(**ports)
                        print(f"  ✅ Connected to existing Weaviate on port {ports['port']}")
                        break
                    except:
                        continue
                continue
        else:
            print("  ❌ Could not connect to Weaviate on any port")
            return False

        if not self.client.is_ready():
            print("  ❌ Weaviate not ready")
            return False

        return self._ensure_collection()

    def _ensure_collection(self) -> bool:
        try:
            if self.client.collections.exists(COLLECTION_NAME):
                print(f"  ✅ Collection {COLLECTION_NAME} already exists")
            else:
                print(f"  📐 Creating collection: {COLLECTION_NAME}")
                self.client.collections.create(
                    name=COLLECTION_NAME,
                    vectorizer_config=None,
                    properties=[
                        Property(name="text",          data_type=DataType.TEXT),
                        Property(name="grantId",       data_type=DataType.TEXT),
                        Property(name="title",         data_type=DataType.TEXT),
                        Property(name="institution",   data_type=DataType.TEXT),
                        Property(name="year",          data_type=DataType.INT),
                        Property(name="institute",     data_type=DataType.TEXT),
                        Property(name="isFQHCFocused", data_type=DataType.BOOL),
                        Property(name="chunkIndex",    data_type=DataType.INT),
                        Property(name="chunkType",     data_type=DataType.TEXT),
                        Property(name="sectionType",   data_type=DataType.TEXT),  # ← NEW
                        Property(name="dataSource",    data_type=DataType.TEXT),
                        Property(name="wordCount",     data_type=DataType.INT),
                        Property(name="isHighValue",   data_type=DataType.BOOL),
                    ]
                )
                print(f"  ✅ Collection created with sectionType property")

            self.collection = self.client.collections.get(COLLECTION_NAME)
            return True
        except Exception as e:
            print(f"  ❌ Collection error: {e}")
            return False

    def is_populated(self) -> Tuple[bool, int]:
        if not self.collection:
            return False, 0
        try:
            resp  = self.collection.aggregate.over_all()
            count = getattr(resp, "total_count", 0) or 0
            if count > 0:
                print(f"  ℹ️  Collection has {count} objects")
                return True, count
            # Fallback check
            res = self.collection.query.fetch_objects(limit=1)
            if hasattr(res, "objects") and res.objects:
                return True, -1
            return False, 0
        except:
            return False, 0

    def import_chunks(self, chunks_df: pd.DataFrame, batch_size: int = 100) -> int:
        print(f"\n📤 Importing {len(chunks_df)} chunks into Weaviate...")
        total, failed = 0, 0

        with self.collection.batch.fixed_size(batch_size=batch_size) as batch:
            for idx, row in chunks_df.iterrows():
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

                    section = str(row.get("section_type", "general"))
                    batch.add_object(
                        properties={
                            "text":          str(row.get("text", ""))[:5000],
                            "grantId":       str(row.get("grant_id", "")),
                            "title":         str(row.get("title", ""))[:200],
                            "institution":   str(row.get("institution", ""))[:200],
                            "year":          int(row.get("year", 2024) if pd.notna(row.get("year", 2024)) else 2024),
                            "institute":     str(row.get("institute", ""))[:100],
                            "isFQHCFocused": bool(row.get("is_fqhc_focused", False)),
                            "chunkIndex":    int(row.get("chunk_index", 0) if pd.notna(row.get("chunk_index", 0)) else 0),
                            "chunkType":     str(row.get("chunk_type", "general"))[:50],
                            "sectionType":   section,                          # ← NEW
                            "dataSource":    str(row.get("data_source", ""))[:50],
                            "wordCount":     int(row.get("word_count", 0) if pd.notna(row.get("word_count", 0)) else 0),
                            "isHighValue":   bool(section in HIGH_VALUE_SECTIONS),
                        },
                        vector=vec
                    )
                    total += 1
                    if total % 500 == 0:
                        print(f"    {total}/{len(chunks_df)}...")

                except Exception as e:
                    failed += 1
                    if failed <= 3:
                        print(f"  ⚠️  Row {idx}: {e}")

        print(f"  ✅ Imported {total} chunks ({failed} failed)")
        return total

    def hybrid_search(self,
                      query: str,
                      query_vector: List[float],
                      alpha: float = 0.5,
                      top_k: int = 10,
                      section_filter: Optional[str] = None,
                      fqhc_only: bool = False) -> List[Dict]:
        """
        Hybrid BM25 + vector search with optional section and FQHC filters.

        Args:
            query:          Natural language query text
            query_vector:   Pre-computed query embedding (section-prefixed)
            alpha:          0.0 = BM25 only, 1.0 = vector only, 0.5 = balanced
            top_k:          Number of results
            section_filter: Restrict to section e.g. "specific_aims", "methods"
            fqhc_only:      If True, only return FQHC-focused grants
        """
        if not self.collection:
            return []
        try:
            # Build filter
            filters = None
            if section_filter and fqhc_only:
                filters = (Filter.by_property("sectionType").equal(section_filter) &
                           Filter.by_property("isFQHCFocused").equal(True))
            elif section_filter:
                filters = Filter.by_property("sectionType").equal(section_filter)
            elif fqhc_only:
                filters = Filter.by_property("isFQHCFocused").equal(True)

            kwargs = dict(
                query=query,
                alpha=alpha,
                limit=top_k,
                return_metadata=MetadataQuery(score=True),
            )
            if alpha > 0 and query_vector:
                kwargs["vector"] = query_vector
            if filters is not None:
                kwargs["filters"] = filters

            resp    = self.collection.query.hybrid(**kwargs)
            results = []
            for obj in resp.objects:
                p     = obj.properties
                score = getattr(obj.metadata, "score", 0) or 0
                results.append({
                    "grant_id":     str(p.get("grantId", "")),
                    "title":        str(p.get("title", "")),
                    "institution":  str(p.get("institution", "")),
                    "year":         p.get("year", ""),
                    "section_type": str(p.get("sectionType", "")),
                    "is_fqhc":      bool(p.get("isFQHCFocused", False)),
                    "is_high_value":bool(p.get("isHighValue", False)),
                    "text":         str(p.get("text", ""))[:300],
                    "score":        round(min(float(score), 1.0), 4),
                    "alpha":        alpha,
                    "section_filter": section_filter or "none",
                })
            return results
        except Exception as e:
            print(f"  ⚠️  Search error: {e}")
            return []

    def close(self):
        if self.client:
            try: self.client.close()
            except: pass
            print("  👋 Weaviate closed")
            self.client = self.collection = None


# ── Section-aware query encoder ───────────────────────────────────────────────

class SectionAwareQueryEncoder:
    """
    Encodes queries with the same section prefix used in Phase 3.
    Ensures query vectors are in the same embedding space as chunk vectors.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"\n🧠 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"  ✅ Model loaded")

    def encode(self, query: str, section_filter: Optional[str] = None) -> List[float]:
        """
        Encode query with section prefix if section_filter is set.
        Matches Phase 3 embedding style exactly.
        """
        if section_filter:
            prefix = SECTION_PREFIXES.get(section_filter, "grant text:")
            text   = f"{prefix} {query}"
        else:
            text = query
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def encode_batch(self, queries: List[str],
                     section_filter: Optional[str] = None) -> List[List[float]]:
        if section_filter:
            prefix = SECTION_PREFIXES.get(section_filter, "grant text:")
            texts  = [f"{prefix} {q}" for q in queries]
        else:
            texts = queries
        embs = self.model.encode(texts, normalize_embeddings=True)
        return embs.tolist()


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Phase4Evaluator:
    """
    Evaluates hybrid search across alpha values.
    Metrics: P@5, MAP, MRR — standard IR metrics for the paper.
    """

    def __init__(self, wm: WeaviateManager, encoder: SectionAwareQueryEncoder):
        self.wm      = wm
        self.encoder = encoder

    def _precision_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        return sum(1 for r in retrieved[:k] if r in relevant) / k

    def _average_precision(self, retrieved: List[str], relevant: set) -> float:
        hits, ap = 0, 0.0
        for i, r in enumerate(retrieved, 1):
            if r in relevant:
                hits += 1
                ap   += hits / i
        return ap / max(len(relevant), 1)

    def _reciprocal_rank(self, retrieved: List[str], relevant: set) -> float:
        for i, r in enumerate(retrieved, 1):
            if r in relevant:
                return 1.0 / i
        return 0.0

    def evaluate_alpha(self, test_queries: List[Dict],
                       alphas: List[float] = None,
                       top_k: int = 5) -> Dict:
        """
        Evaluate hybrid search across alpha values.
        Returns metrics dict keyed by alpha.
        """
        if alphas is None:
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

        print(f"\n🧪 Evaluating {len(test_queries)} queries × {len(alphas)} alpha values…")
        all_metrics = {}

        for alpha in alphas:
            p_at_1, p_at_3, p_at_5, aps, rrs, times = [], [], [], [], [], []

            for qd in test_queries:
                query       = qd["query"]
                relevant    = set(qd.get("relevant_grant_ids", []))
                section     = qd.get("section_filter")

                vec = self.encoder.encode(query, section_filter=section)

                t0      = time.time()
                results = self.wm.hybrid_search(
                    query, vec, alpha=alpha, top_k=top_k,
                    section_filter=section
                )
                times.append(time.time() - t0)

                retrieved = [r["grant_id"] for r in results]
                p_at_1.append(self._precision_at_k(retrieved, relevant, 1))
                p_at_3.append(self._precision_at_k(retrieved, relevant, 3))
                p_at_5.append(self._precision_at_k(retrieved, relevant, 5))
                aps.append(self._average_precision(retrieved, relevant))
                rrs.append(self._reciprocal_rank(retrieved, relevant))

            all_metrics[alpha] = {
                "p@1":      round(float(np.mean(p_at_1)), 4),
                "p@3":      round(float(np.mean(p_at_3)), 4),
                "p@5":      round(float(np.mean(p_at_5)), 4),
                "map":      round(float(np.mean(aps)), 4),
                "mrr":      round(float(np.mean(rrs)), 4),
                "avg_time": round(float(np.mean(times)), 4),
            }
            m = all_metrics[alpha]
            print(f"  α={alpha:.2f} → P@5={m['p@5']:.3f}  MAP={m['map']:.3f}  "
                  f"MRR={m['mrr']:.3f}  t={m['avg_time']:.3f}s")

        # Find optimal alpha by MAP
        optimal = max(all_metrics, key=lambda a: all_metrics[a]["map"])
        print(f"\n  ✅ Optimal α={optimal} (MAP={all_metrics[optimal]['map']:.3f})")
        return all_metrics, optimal

    def section_comparison(self, query: str,
                           sections: List[str] = None,
                           alpha: float = 0.5,
                           top_k: int = 5) -> Dict:
        """
        Run the same query across different section filters and compare results.
        Useful for demo and paper — shows section-aware retrieval in action.
        """
        if sections is None:
            sections = [None, "specific_aims", "significance", "methods", "background"]

        print(f"\n📊 Section comparison: '{query}'  α={alpha}")
        comparison = {}

        for section in sections:
            label = section or "all sections"
            vec   = self.encoder.encode(query, section_filter=section)
            results = self.wm.hybrid_search(
                query, vec, alpha=alpha, top_k=top_k,
                section_filter=section
            )
            comparison[label] = results
            top_ids = [r["grant_id"] for r in results[:3]]
            print(f"  [{label}]: {top_ids}")

        return comparison


# ── Default test queries ──────────────────────────────────────────────────────

def build_eval_queries(chunks_csv: str = CHUNKS_CSV) -> List[Dict]:
    """
    Build evaluation queries from the actual chunk corpus.
    Uses FQHC-focused chunks as ground truth.
    """
    try:
        df = pd.read_csv(chunks_csv, usecols=[
            "grant_id", "section_type", "is_fqhc_focused", "text", "data_source"
        ])
    except Exception as e:
        print(f"⚠️  Could not load chunks for eval: {e}")
        return _default_queries()

    fqhc = df[df["is_fqhc_focused"] == True].copy()
    if fqhc.empty:
        return _default_queries()

    # Build one query per FQHC grant, targeting its highest-value section
    seen_grants, queries = set(), []
    for _, row in fqhc.iterrows():
        gid = row["grant_id"]
        if gid in seen_grants:
            continue
        seen_grants.add(gid)

        text_lower = str(row.get("text", "")).lower()
        if "diabetes" in text_lower:
            q = "diabetes management community health center"
        elif "hypertension" in text_lower or "blood pressure" in text_lower:
            q = "hypertension control underserved populations"
        elif "depression" in text_lower or "mental health" in text_lower:
            q = "behavioral health integration primary care"
        elif "hiv" in text_lower or "aids" in text_lower:
            q = "HIV prevention safety-net clinic"
        elif "cancer" in text_lower:
            q = "cancer screening community health center"
        elif "asthma" in text_lower:
            q = "asthma management pediatric FQHC"
        else:
            q = "community health center underserved populations intervention"

        section = row.get("section_type")
        queries.append({
            "query":               q,
            "relevant_grant_ids":  [gid],
            "section_filter":      section if section in HIGH_VALUE_SECTIONS else None,
        })

        if len(queries) >= 50:
            break

    print(f"✅ Built {len(queries)} eval queries from corpus")
    return queries if queries else _default_queries()


def _default_queries() -> List[Dict]:
    return [
        {"query": "diabetes prevention community health workers FQHC",
         "relevant_grant_ids": [], "section_filter": "specific_aims"},
        {"query": "behavioral health integration primary care underserved",
         "relevant_grant_ids": [], "section_filter": "significance"},
        {"query": "telehealth rural community health center implementation",
         "relevant_grant_ids": [], "section_filter": "methods"},
        {"query": "cancer screening medically underserved populations",
         "relevant_grant_ids": [], "section_filter": None},
        {"query": "community health worker chronic disease management",
         "relevant_grant_ids": [], "section_filter": "specific_aims"},
    ]


# ── Main run function ─────────────────────────────────────────────────────────

def run_phase4(force_reimport: bool = False):
    """
    Run Phase 4 hybrid search evaluation.

    Args:
        force_reimport: Re-import chunks even if Weaviate already populated.
    """
    print("\n🚀 STARTING PHASE 4: SECTION-AWARE HYBRID RAG")
    t_start = time.time()

    # 1. Connect to Weaviate
    wm = WeaviateManager()
    if not wm.start():
        print("❌ Weaviate failed — cannot continue")
        return None

    # 2. Load chunks CSV
    print(f"\n📂 Loading chunks from {CHUNKS_CSV}…")
    try:
        chunks_df = pd.read_csv(CHUNKS_CSV)
        print(f"  ✅ {len(chunks_df)} chunks loaded")
    except FileNotFoundError:
        print(f"  ❌ {CHUNKS_CSV} not found — run Phase 3 first")
        wm.close(); return None

    # 3. Import if needed
    populated, count = wm.is_populated()
    if not populated or force_reimport:
        print("  📤 Importing chunks…")
        imported = wm.import_chunks(chunks_df)
        if imported == 0:
            print("  ❌ Import failed"); wm.close(); return None
        print(f"  ✅ Imported {imported} chunks")
    else:
        print(f"  ✅ Weaviate already has {count} objects — skipping import")

    # 4. Load embedding model
    if not EMBEDDING_AVAILABLE:
        print("❌ sentence-transformers not available"); wm.close(); return None
    encoder = SectionAwareQueryEncoder(EMBEDDING_MODEL)

    # 5. Build evaluation queries
    test_queries = build_eval_queries(CHUNKS_CSV)

    # 6. Evaluate across alpha values
    evaluator = Phase4Evaluator(wm, encoder)
    alpha_metrics, optimal_alpha = evaluator.evaluate_alpha(
        test_queries, alphas=[0.0, 0.25, 0.5, 0.75, 1.0], top_k=5
    )

    # 7. Section comparison demo
    print("\n📊 SECTION COMPARISON DEMO")
    evaluator.section_comparison(
        "diabetes management community health worker intervention",
        alpha=optimal_alpha
    )

    # 8. Save results
    results = {
        "timestamp":      datetime.now().isoformat(),
        "model":          EMBEDDING_MODEL,
        "chunks_total":   len(chunks_df),
        "eval_queries":   len(test_queries),
        "alpha_metrics":  {str(k): v for k, v in alpha_metrics.items()},
        "optimal_alpha":  optimal_alpha,
        "runtime_s":      round(time.time() - t_start, 1),
        "section_indexes_used": True,
    }

    os.makedirs(os.path.dirname(RESULTS_JSON) or ".", exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # 9. Print summary
    print("\n" + "=" * 70)
    print("✅ PHASE 4 COMPLETE")
    print("=" * 70)
    print(f"\n📊 Alpha Tuning Results:")
    print(f"  {'α':>6}  {'P@1':>6}  {'P@3':>6}  {'P@5':>6}  {'MAP':>6}  {'MRR':>6}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
    for alpha in sorted(alpha_metrics):
        m = alpha_metrics[alpha]
        star = " ←" if alpha == optimal_alpha else ""
        print(f"  {alpha:>6.2f}  {m['p@1']:>6.3f}  {m['p@3']:>6.3f}  "
              f"{m['p@5']:>6.3f}  {m['map']:>6.3f}  {m['mrr']:>6.3f}{star}")

    print(f"\n  Optimal α = {optimal_alpha}")
    print(f"\n📁 Results saved to {RESULTS_JSON}")
    print(f"\n🔗 Phase 5 / query_pipeline.py: set DEFAULT_ALPHA = {optimal_alpha}")

    wm.close()
    return results


# ── Interactive search (for notebook use) ────────────────────────────────────

def interactive_search(wm: WeaviateManager = None,
                       encoder: SectionAwareQueryEncoder = None):
    """Quick interactive search — call from notebook after run_phase4()."""
    if wm is None:
        wm = WeaviateManager()
        wm.start()
    if encoder is None:
        encoder = SectionAwareQueryEncoder()

    print("\n💬 Interactive hybrid search")
    print("  Commands: /aims, /significance, /methods, /fqhc, /quit")

    while True:
        try:
            ui = input("\nQuery > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not ui: continue
        if ui.lower() in ["/quit", "quit", "exit"]: break

        section = None
        if ui.startswith("/aims "):       section, ui = "specific_aims", ui[6:]
        elif ui.startswith("/significance "): section, ui = "significance", ui[14:]
        elif ui.startswith("/methods "):   section, ui = "methods", ui[9:]
        elif ui.startswith("/fqhc "):
            vec     = encoder.encode(ui[6:])
            results = wm.hybrid_search(ui[6:], vec, fqhc_only=True)
            _print(results); continue

        vec     = encoder.encode(ui, section_filter=section)
        results = wm.hybrid_search(ui, vec, section_filter=section)
        _print(results)


def _print(results: List[Dict]):
    print(f"\n{'─'*70}\n📋 {len(results)} results\n{'─'*70}")
    for i, r in enumerate(results, 1):
        hv   = "⭐" if r.get("is_high_value") else "  "
        fqhc = "🏥" if r.get("is_fqhc") else "  "
        print(f"\n{i}. {hv}{fqhc} [{r['score']:.4f}] "
              f"section:{r.get('section_type','?')}  α:{r.get('alpha','?')}")
        print(f"   Grant: {r.get('grant_id','')}")
        if r.get("title"): print(f"   Title: {r['title'][:70]}")
        if r.get("text"):  print(f"   Text:  {r['text'][:200]}…")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reimport", action="store_true",
                        help="Force reimport chunks even if Weaviate populated")
    args, _ = parser.parse_known_args()
    run_phase4(force_reimport=args.reimport)