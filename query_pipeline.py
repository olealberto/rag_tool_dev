# ============================================================================
# 📁 query_pipeline.py - UNIFIED QUERY PIPELINE
# ============================================================================

"""
UNIFIED QUERY PIPELINE: Hybrid RAG + Knowledge Graph

Single entry point for the full system. Boots Weaviate, loads the knowledge
graph, and answers grant-matching queries using:
    - Phase 4: Weaviate hybrid search (BM25 + vector)
    - Phase 5: Knowledge graph expansion

Run:
    !python query_pipeline.py                         # interactive mode
    !python query_pipeline.py --query "diabetes CHW"  # single query
    !python query_pipeline.py --setup-only            # just load, no query

Demo usage in Colab:
    from query_pipeline import GrantQueryPipeline
    pipeline = GrantQueryPipeline()
    pipeline.setup()
    results = pipeline.query("diabetes prevention community health workers")
"""

print("="*70)
print("🚀 GRANT QUERY PIPELINE: Hybrid RAG + Knowledge Graph")
print("="*70)

import os
import sys
import ast
import json
import time
import argparse
import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

# ============ PATHS ============

PATHS = {
    "abstracts":   "./phase2_output/nih_research_abstracts.csv",
    "chunks":      "./phase3_results/document_chunks_with_embeddings.csv",
    "graph_pkl":   "./phase5_graph_store/graph.pkl",
    "graph_gml":   "./phase5_knowledge_graph.gml",
}

COLLECTION_NAME = "GrantChunk"
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
DEFAULT_ALPHA   = 0.25   # optimal from evaluation
DEFAULT_TOP_K   = 10


# ============ WEAVIATE MANAGER ============

class WeaviateManager:

    def __init__(self):
        self.client     = None
        self.collection = None

    def start(self) -> bool:
        try:
            import weaviate
            from weaviate.classes.config import Property, DataType

            print("\n🔌 Starting Weaviate embedded...")
            try:
                self.client = weaviate.connect_to_embedded(
                    port=8080, grpc_port=50051
                )
                print("  ✅ Weaviate ready")
            except Exception as embed_err:
                err_str = str(embed_err)
                if "8079" in err_str or "already" in err_str.lower() or "listening" in err_str.lower():
                    print("  ℹ️  Weaviate already running — connecting to existing instance")
                    self.client = weaviate.connect_to_local(port=8079, grpc_port=50050)
                    print("  ✅ Weaviate ready (existing instance)")
                else:
                    raise embed_err

            # Create collection if it doesn't exist
            if not self.client.collections.exists(COLLECTION_NAME):
                print(f"  📐 Creating collection: {COLLECTION_NAME}")
                self.client.collections.create(
                    name=COLLECTION_NAME,
                    vectorizer_config=None,
                    properties=[
                        Property(name="text",         data_type=DataType.TEXT),
                        Property(name="grantId",      data_type=DataType.TEXT),
                        Property(name="title",        data_type=DataType.TEXT),
                        Property(name="institution",  data_type=DataType.TEXT),
                        Property(name="year",         data_type=DataType.INT),
                        Property(name="isFQHC",       data_type=DataType.BOOL),
                        Property(name="chunkIndex",   data_type=DataType.INT),
                        Property(name="chunkType",    data_type=DataType.TEXT),
                    ]
                )

            self.collection = self.client.collections.get(COLLECTION_NAME)
            return True

        except Exception as e:
            print(f"  ❌ Weaviate failed: {e}")
            return False

    def is_populated(self) -> bool:
        try:
            resp = self.collection.aggregate.over_all(total_count=True)
            count = resp.total_count or 0
            print(f"  ℹ️  Collection has {count} objects")
            return count > 0
        except Exception:
            return False

    def import_chunks(self, chunks_df: pd.DataFrame) -> int:
        print(f"\n  📤 Importing {len(chunks_df)} chunks into Weaviate...")
        total, failed = 0, 0

        with self.collection.batch.fixed_size(batch_size=200) as batch:
            for _, row in chunks_df.iterrows():
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
                            "text":        str(row.get("text", ""))[:5000],
                            "grantId":     str(row.get("grant_id", "")),
                            "title":       str(row.get("title", "")),
                            "institution": str(row.get("institution", "")),
                            "year":        int(row.get("year", 2024) or 2024),
                            "isFQHC":      bool(row.get("has_fqhc_terms", False)),
                            "chunkIndex":  int(row.get("chunk_index", 0) or 0),
                            "chunkType":   str(row.get("chunk_type", "abstract")),
                        },
                        vector=vec
                    )
                    total += 1
                    if total % 500 == 0:
                        print(f"    Imported {total}/{len(chunks_df)}...")

                except Exception:
                    failed += 1

        print(f"  ✅ Imported {total} chunks ({failed} failed)")
        return total

    def hybrid_search(self, query: str, query_vector: List[float],
                      alpha: float = DEFAULT_ALPHA,
                      top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        try:
            from weaviate.classes.query import MetadataQuery

            if alpha > 0 and query_vector:
                resp = self.collection.query.hybrid(
                    query=query,
                    alpha=alpha,
                    vector=query_vector,
                    limit=top_k,
                    return_metadata=MetadataQuery(score=True)
                )
            else:
                resp = self.collection.query.hybrid(
                    query=query,
                    alpha=0.0,
                    limit=top_k,
                    return_metadata=MetadataQuery(score=True)
                )

            results = []
            for obj in resp.objects:
                p = obj.properties
                results.append({
                    "grant_id":    p.get("grantId", ""),
                    "title":       p.get("title", ""),
                    "institution": p.get("institution", ""),
                    "year":        p.get("year", ""),
                    "is_fqhc":     p.get("isFQHC", False),
                    "text":        p.get("text", "")[:300],
                    "score":       round(obj.metadata.score or 0, 4),
                    "source":      "hybrid_search",
                })
            return results

        except Exception as e:
            print(f"  ⚠️  Search error: {e}")
            return []

    def close(self):
        if self.client:
            self.client.close()
            print("  👋 Weaviate closed")


# ============ GRAPH QUERY ENGINE ============

class GraphQueryEngine:

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def expand(self, seed_grant_ids: List[str],
               top_k: int = 5) -> List[Dict]:
        """
        Expand seed results via graph traversal.
        Traverses condition/intervention/population hub nodes
        to find related grants the vector search missed.
        """
        expanded  = []
        seen      = set(seed_grant_ids)

        for grant_id in seed_grant_ids[:3]:  # expand from top 3
            if grant_id not in self.graph:
                continue

            for neighbor in self.graph.neighbors(grant_id):
                ntype = self.graph.nodes[neighbor].get("type", "")

                # Traverse through semantic hub nodes
                if ntype in ["condition", "intervention", "population"]:
                    hub_name = self.graph.nodes[neighbor].get("name", neighbor)

                    for second_hop in self.graph.neighbors(neighbor):
                        if second_hop in seen:
                            continue
                        if self.graph.nodes[second_hop].get("type") != "grant":
                            continue

                        seen.add(second_hop)
                        edge = self.graph.get_edge_data(grant_id, neighbor, {})

                        expanded.append({
                            "grant_id":    second_hop,
                            "title":       "",
                            "institution": self.graph.nodes[second_hop].get(
                                               "institution", ""),
                            "year":        self.graph.nodes[second_hop].get(
                                               "year", ""),
                            "is_fqhc":     bool(self.graph.nodes[second_hop].get(
                                               "is_fqhc_focused", False)),
                            "text":        "",
                            "score":       0.0,
                            "source":      f"graph_expansion ({hub_name})",
                            "shared_hub":  hub_name,
                        })

        return expanded[:top_k]

    def rfp_match(self, conditions: List[str] = None,
                  interventions: List[str]    = None,
                  populations: List[str]      = None,
                  fqhc_only: bool             = False) -> List[Dict]:
        """
        Find grants via direct graph traversal — no vector search.
        Useful for structured RFP matching.
        """
        scores = defaultdict(float)

        def traverse(prefix, items, weight):
            for item in (items or []):
                node_id = f"{prefix}{item}"
                if node_id not in self.graph:
                    # Try partial match
                    matches = [n for n in self.graph.nodes
                               if n.startswith(prefix) and item.lower() in n.lower()]
                    if matches:
                        node_id = matches[0]
                    else:
                        continue
                for neighbor in self.graph.neighbors(node_id):
                    if self.graph.nodes[neighbor].get("type") == "grant":
                        scores[neighbor] += weight

        traverse("COND_", conditions,    1.5)
        traverse("INT_",  interventions, 1.0)
        traverse("POP_",  populations,   1.0)

        if fqhc_only and "FQHC_HUB" in self.graph:
            fqhc = {n for n in self.graph.neighbors("FQHC_HUB")
                    if self.graph.nodes[n].get("type") == "grant"}
            scores = {k: v for k, v in scores.items() if k in fqhc}

        results = []
        for gid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            nd = self.graph.nodes.get(gid, {})
            results.append({
                "grant_id":    gid,
                "title":       "",
                "institution": nd.get("institution", nd.get("institute", "")),
                "year":        nd.get("year", ""),
                "is_fqhc":     bool(nd.get("is_fqhc_focused", False)),
                "text":        "",
                "score":       round(score, 2),
                "source":      "graph_rfp_match",
            })

        return results

    def graph_stats(self) -> Dict:
        node_types = defaultdict(int)
        for _, d in self.graph.nodes(data=True):
            node_types[d.get("type", "unknown")] += 1
        return {
            "nodes":      self.graph.number_of_nodes(),
            "edges":      self.graph.number_of_edges(),
            "node_types": dict(node_types),
        }


# ============ RESULT ENRICHER ============

class ResultEnricher:
    """Joins retrieval results back to abstracts CSV for full metadata"""

    def __init__(self, abstracts_df: pd.DataFrame):
        self.abstracts = abstracts_df.set_index("grant_id")

    def enrich(self, results: List[Dict]) -> List[Dict]:
        enriched = []
        for r in results:
            gid = r.get("grant_id", "")
            if gid in self.abstracts.index:
                row = self.abstracts.loc[gid]
                # If duplicate grant_ids exist, loc returns a DataFrame — take first row
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                r["title"]            = r.get("title") or str(row.get("title", ""))
                r["institution"]      = r.get("institution") or str(row.get("institution", ""))
                r["year"]             = r.get("year") or int(row.get("year", 0) or 0)
                r["is_fqhc"]          = bool(row.get("has_fqhc_terms", False))
                r["abstract_snippet"] = str(row.get("abstract", ""))[:200]
            enriched.append(r)
        return enriched


# ============ GRANT QUERY PIPELINE ============

class GrantQueryPipeline:
    """
    Main pipeline class. Call setup() once, then query() as many times as needed.
    Weaviate stays alive between queries.

    Usage:
        pipeline = GrantQueryPipeline()
        pipeline.setup()

        results = pipeline.query("diabetes prevention community health workers")
        results = pipeline.query("telehealth for hypertension in rural populations")
        results = pipeline.rfp_query(
            conditions=["diabetes"],
            interventions=["CHW"],
            populations=["Latino"],
        )

        pipeline.close()
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, top_k: int = DEFAULT_TOP_K):
        self.alpha      = alpha
        self.top_k      = top_k
        self.weaviate   = WeaviateManager()
        self.graph_engine = None
        self.enricher   = None
        self.model      = None
        self._ready     = False

    def setup(self) -> bool:
        """
        Boot everything. Call once before querying.
        Takes ~60s on first run (Weaviate import).
        Subsequent runs are faster if Weaviate is already populated.
        """
        print("\n" + "="*70)
        print("⚙️  SETTING UP QUERY PIPELINE")
        print("="*70)

        start = time.time()

        # 1. Load abstracts
        print("\n📦 Loading abstracts...")
        if not os.path.exists(PATHS["abstracts"]):
            print(f"  ❌ Abstracts not found: {PATHS['abstracts']}")
            print("     Run phase2_api.py first")
            return False
        abstracts_df = pd.read_csv(PATHS["abstracts"])
        self.enricher = ResultEnricher(abstracts_df)
        print(f"  ✅ {len(abstracts_df)} abstracts loaded")

        # 2. Load embedding model
        print("\n🧠 Loading embedding model...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"  ✅ {EMBEDDING_MODEL}")
        except Exception as e:
            print(f"  ❌ Model load failed: {e}")
            return False

        # 3. Start Weaviate
        if not self.weaviate.start():
            return False

        # 4. Import chunks if Weaviate is empty
        if not self.weaviate.is_populated():
            print("\n  📥 Weaviate is empty — importing chunks...")
            chunks_path = PATHS["chunks"]
            if not os.path.exists(chunks_path):
                print(f"  ❌ Chunks not found: {chunks_path}")
                print("     Run phase3_document_rag.py first")
                return False

            print(f"  ⏳ Reading {chunks_path} (this is a large file)...")
            chunks_df = pd.read_csv(chunks_path)

            # Merge title/institution from abstracts if missing
            if "title" not in chunks_df.columns:
                meta = abstracts_df[["grant_id", "title", "institution",
                                     "has_fqhc_terms"]].copy()
                chunks_df = chunks_df.merge(meta, on="grant_id", how="left")

            imported = self.weaviate.import_chunks(chunks_df)
            if imported == 0:
                print("  ❌ No chunks imported")
                return False
        else:
            print("  ✅ Weaviate already populated — skipping import")

        # 5. Load knowledge graph
        print("\n🕸️  Loading knowledge graph...")
        graph = self._load_graph()
        if graph is None:
            print("  ⚠️  No graph found — run phase5_knowledge_graph.py first")
            print("       Graph expansion will be unavailable")
        else:
            self.graph_engine = GraphQueryEngine(graph)
            stats = self.graph_engine.graph_stats()
            print(f"  ✅ Graph: {stats['nodes']} nodes, {stats['edges']} edges")
            print(f"     Node types: {stats['node_types']}")

        elapsed = round(time.time() - start, 1)
        self._ready = True
        print(f"\n✅ Pipeline ready ({elapsed}s)")
        print(f"   Alpha: {self.alpha} (BM25/vector blend)")
        print(f"   Top-K: {self.top_k}")
        print(f"   Graph expansion: {'enabled' if self.graph_engine else 'disabled'}")

        return True

    def query(self, query_text: str,
              alpha: float       = None,
              top_k: int         = None,
              use_graph: bool    = True,
              fqhc_only: bool    = False,
              verbose: bool      = True) -> List[Dict]:
        """
        Run a hybrid search query with optional graph expansion.

        Args:
            query_text: Natural language query
            alpha:      BM25/vector blend (0=BM25 only, 1=vector only)
                        Defaults to optimal alpha from evaluation (0.25)
            top_k:      Number of results to return
            use_graph:  Whether to expand results via knowledge graph
            fqhc_only:  Filter to FQHC-relevant grants only
            verbose:    Print results to console

        Returns:
            List of result dicts with grant_id, title, institution,
            score, source, abstract_snippet
        """
        if not self._ready:
            print("❌ Pipeline not set up. Call setup() first.")
            return []

        alpha = alpha if alpha is not None else self.alpha
        top_k = top_k if top_k is not None else self.top_k

        if verbose:
            print(f"\n{'='*70}")
            print(f"🔍 QUERY: {query_text}")
            print(f"   α={alpha}  top_k={top_k}  graph={'on' if use_graph else 'off'}"
                  f"  fqhc_only={fqhc_only}")
            print(f"{'='*70}")

        start = time.time()

        # 1. Embed query
        query_vec = self.model.encode(query_text).tolist()

        # 2. Hybrid search (Phase 4)
        hybrid_results = self.weaviate.hybrid_search(
            query=query_text,
            query_vector=query_vec,
            alpha=alpha,
            top_k=top_k
        )

        if verbose:
            print(f"\n📊 Hybrid search: {len(hybrid_results)} results "
                  f"(α={alpha}, {time.time()-start:.2f}s)")

        # 3. Graph expansion (Phase 5)
        graph_results = []
        if use_graph and self.graph_engine and hybrid_results:
            seed_ids     = [r["grant_id"] for r in hybrid_results]
            graph_results = self.graph_engine.expand(seed_ids, top_k=5)

            if verbose:
                print(f"🕸️  Graph expansion: {len(graph_results)} additional results")

        # 4. Combine and deduplicate
        all_results  = hybrid_results + graph_results
        seen, unique = set(), []
        for r in all_results:
            gid = r["grant_id"]
            if gid and gid not in seen:
                seen.add(gid)
                unique.append(r)

        # 5. Filter FQHC if requested
        if fqhc_only:
            unique = [r for r in unique if r.get("is_fqhc")]
            if verbose:
                print(f"🏥 FQHC filter: {len(unique)} results")

        # 6. Enrich with abstract metadata
        unique = self.enricher.enrich(unique)

        # 7. Print results
        if verbose:
            self._print_results(unique, query_text)

        return unique

    def rfp_query(self,
                  conditions: List[str]    = None,
                  interventions: List[str] = None,
                  populations: List[str]   = None,
                  free_text: str           = None,
                  fqhc_only: bool          = False,
                  verbose: bool            = True) -> List[Dict]:
        """
        Structured RFP matching query.
        Combines graph traversal (structured) + hybrid search (free text).

        Args:
            conditions:    Health conditions e.g. ["diabetes", "hypertension"]
            interventions: Interventions e.g. ["CHW", "telehealth"]
            populations:   Target populations e.g. ["Latino", "pediatric"]
            free_text:     Additional free text query for hybrid search
            fqhc_only:     Restrict to FQHC-relevant grants

        Example:
            results = pipeline.rfp_query(
                conditions=["diabetes"],
                interventions=["CHW"],
                populations=["Latino"],
                fqhc_only=True
            )
        """
        if not self._ready:
            print("❌ Pipeline not set up. Call setup() first.")
            return []

        if verbose:
            print(f"\n{'='*70}")
            print(f"🎯 RFP QUERY")
            if conditions:    print(f"   Conditions:    {conditions}")
            if interventions: print(f"   Interventions: {interventions}")
            if populations:   print(f"   Populations:   {populations}")
            if free_text:     print(f"   Free text:     {free_text}")
            print(f"   FQHC only: {fqhc_only}")
            print(f"{'='*70}")

        all_results = []

        # Graph-based structured match
        if self.graph_engine and any([conditions, interventions, populations]):
            graph_results = self.graph_engine.rfp_match(
                conditions=conditions,
                interventions=interventions,
                populations=populations,
                fqhc_only=fqhc_only
            )
            all_results.extend(graph_results)
            if verbose:
                print(f"\n🕸️  Graph match: {len(graph_results)} results")

        # Hybrid search on free text or auto-generated query
        search_query = free_text or " ".join(
            (conditions or []) + (interventions or []) + (populations or [])
        )
        if search_query:
            query_vec     = self.model.encode(search_query).tolist()
            hybrid_results = self.weaviate.hybrid_search(
                query=search_query,
                query_vector=query_vec,
                alpha=self.alpha,
                top_k=self.top_k
            )
            if verbose:
                print(f"📊 Hybrid search: {len(hybrid_results)} results")
            all_results.extend(hybrid_results)

        # Deduplicate — graph results first (they scored via structured match)
        seen, unique = set(), []
        for r in all_results:
            gid = r["grant_id"]
            if gid and gid not in seen:
                seen.add(gid)
                unique.append(r)

        if fqhc_only:
            unique = [r for r in unique if r.get("is_fqhc")]

        unique = self.enricher.enrich(unique)

        if verbose:
            self._print_results(unique, search_query)

        return unique

    def compare_approaches(self, query_text: str) -> Dict:
        """
        Show side-by-side comparison of BM25, vector, hybrid, and graph-expanded.
        Useful for demo.
        """
        if not self._ready:
            print("❌ Not set up.")
            return {}

        print(f"\n{'='*70}")
        print(f"📊 APPROACH COMPARISON: '{query_text}'")
        print(f"{'='*70}")

        query_vec = self.model.encode(query_text).tolist()

        results = {}
        for label, alpha in [("BM25 only (α=0.0)", 0.0),
                              ("Hybrid (α=0.25)", 0.25),
                              ("Hybrid (α=0.5)", 0.5),
                              ("Vector only (α=1.0)", 1.0)]:
            r = self.weaviate.hybrid_search(query_text, query_vec,
                                            alpha=alpha, top_k=5)
            results[label] = r
            ids = [x["grant_id"][:20] for x in r[:3]]
            print(f"\n  {label}:")
            for i, x in enumerate(r[:3], 1):
                print(f"    {i}. [{x['score']:.4f}] {x['grant_id']} — "
                      f"{x['title'][:50] or x['text'][:50]}...")

        # Graph expansion on top of best hybrid
        if self.graph_engine:
            hybrid = self.weaviate.hybrid_search(query_text, query_vec,
                                                 alpha=0.25, top_k=5)
            seed_ids = [r["grant_id"] for r in hybrid]
            expanded = self.graph_engine.expand(seed_ids, top_k=5)
            results["Graph expansion"] = expanded
            print(f"\n  Graph expansion (additional grants via hub nodes):")
            for i, x in enumerate(expanded[:3], 1):
                print(f"    {i}. {x['grant_id']} — via {x.get('shared_hub', '?')}")

        return results

    def close(self):
        """Shut down Weaviate. Call when done."""
        self.weaviate.close()
        self._ready = False

    # ---- Helpers ----

    def _load_graph(self) -> Optional[nx.Graph]:
        import pickle
        # Prefer pickle (faster)
        if os.path.exists(PATHS["graph_pkl"]):
            try:
                with open(PATHS["graph_pkl"], "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        # Fall back to GML
        if os.path.exists(PATHS["graph_gml"]):
            try:
                return nx.read_gml(PATHS["graph_gml"])
            except Exception:
                pass
        return None

    def _print_results(self, results: List[Dict], query: str):
        print(f"\n{'─'*70}")
        print(f"📋 RESULTS ({len(results)} total)")
        print(f"{'─'*70}")

        for i, r in enumerate(results[:self.top_k], 1):
            source_label = {
                "hybrid_search":   "🔍 Hybrid",
                "graph_rfp_match": "🎯 Graph",
            }.get(r.get("source", ""), f"🕸️  {r.get('source', '')}")

            print(f"\n{i}. {source_label}  [score: {r.get('score', 0):.4f}]")
            print(f"   Grant ID:    {r.get('grant_id', 'N/A')}")
            if r.get("title"):
                print(f"   Title:       {r['title'][:70]}")
            if r.get("institution"):
                print(f"   Institution: {r['institution'][:60]}")
            if r.get("year"):
                print(f"   Year:        {r['year']}")
            if r.get("is_fqhc"):
                print(f"   ✅ FQHC-relevant")
            if r.get("abstract_snippet"):
                print(f"   Abstract:    {r['abstract_snippet'][:150]}...")


# ============ INTERACTIVE MODE ============

def interactive_mode(pipeline: GrantQueryPipeline):
    print("\n" + "="*70)
    print("💬 INTERACTIVE QUERY MODE")
    print("="*70)
    print("Commands:")
    print("  <query>              — hybrid + graph search")
    print("  /rfp <query>         — structured RFP match")
    print("  /compare <query>     — compare all approaches")
    print("  /fqhc <query>        — FQHC-only results")
    print("  /alpha <0-1> <query> — custom alpha")
    print("  /quit                — exit")
    print()

    while True:
        try:
            user_input = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
            break

        elif user_input.startswith("/compare "):
            q = user_input[9:].strip()
            pipeline.compare_approaches(q)

        elif user_input.startswith("/fqhc "):
            q = user_input[6:].strip()
            pipeline.query(q, fqhc_only=True)

        elif user_input.startswith("/alpha "):
            parts = user_input.split(" ", 2)
            if len(parts) == 3:
                try:
                    alpha = float(parts[1])
                    q     = parts[2]
                    pipeline.query(q, alpha=alpha)
                except ValueError:
                    print("  Usage: /alpha 0.5 your query here")

        elif user_input.startswith("/rfp "):
            q = user_input[5:].strip()
            pipeline.rfp_query(free_text=q)

        else:
            pipeline.query(user_input)


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(
        description="Grant Query Pipeline: Hybrid RAG + Knowledge Graph"
    )
    parser.add_argument("--query",      type=str, help="Single query to run")
    parser.add_argument("--alpha",      type=float, default=DEFAULT_ALPHA,
                        help=f"BM25/vector blend (default: {DEFAULT_ALPHA})")
    parser.add_argument("--top-k",      type=int, default=DEFAULT_TOP_K,
                        help=f"Results per query (default: {DEFAULT_TOP_K})")
    parser.add_argument("--fqhc-only",  action="store_true",
                        help="Filter to FQHC-relevant grants")
    parser.add_argument("--no-graph",   action="store_true",
                        help="Disable graph expansion")
    parser.add_argument("--compare",    action="store_true",
                        help="Compare BM25 vs hybrid vs vector vs graph")
    parser.add_argument("--setup-only", action="store_true",
                        help="Set up pipeline but don't query")
    args = parser.parse_args()

    pipeline = GrantQueryPipeline(alpha=args.alpha, top_k=args.top_k)
    ok = pipeline.setup()

    if not ok:
        print("❌ Setup failed")
        sys.exit(1)

    if args.setup_only:
        print("\n✅ Setup complete (--setup-only mode)")
        pipeline.close()
        return

    try:
        if args.query and args.compare:
            pipeline.compare_approaches(args.query)

        elif args.query:
            pipeline.query(
                args.query,
                use_graph=not args.no_graph,
                fqhc_only=args.fqhc_only
            )

        else:
            # Interactive mode
            interactive_mode(pipeline)

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()