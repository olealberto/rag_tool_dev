# ============================================================================
# 📁 query_pipeline.py - UNIFIED QUERY PIPELINE (COLAB-READY)
# ============================================================================

"""
UNIFIED QUERY PIPELINE: Hybrid RAG + Knowledge Graph

Single entry point for the full system. Boots Weaviate, loads the knowledge
graph, and answers grant-matching queries using:
    - Phase 4: Weaviate hybrid search (BM25 + vector)
    - Phase 5: Knowledge graph expansion

Run in Colab:
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
import pickle

from pathlib import Path
from typing import List, Dict, Optional, Tuple
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
        """Start Weaviate embedded or connect to existing instance"""
        try:
            import weaviate
            from weaviate.classes.config import Property, DataType
        except ImportError:
            print("  ❌ weaviate-client not installed. Run: pip install weaviate-client")
            return False

        print("\n🔌 Starting Weaviate...")
        
        # Try multiple port combinations for flexibility
        port_configs = [
            {"port": 8079, "grpc_port": 50050},  # Default
            {"port": 8080, "grpc_port": 50051},  # Alternative
        ]
        
        connected = False
        for ports in port_configs:
            try:
                # Try embedded first
                self.client = weaviate.connect_to_embedded(**ports)
                print(f"  ✅ Weaviate embedded started on port {ports['port']}")
                connected = True
                break
            except Exception as embed_err:
                err_str = str(embed_err).lower()
                # Check if ports are already in use (Weaviate already running)
                if "address already in use" in err_str or "already listening" in err_str:
                    try:
                        self.client = weaviate.connect_to_local(**ports)
                        print(f"  ✅ Connected to existing Weaviate on port {ports['port']}")
                        connected = True
                        break
                    except:
                        continue
                # If it's a different error, try next port config
                continue
        
        if not connected:
            print("  ❌ Could not start or connect to Weaviate")
            return False

        # Verify client is ready
        if not self.client.is_ready():
            print("  ❌ Weaviate client not ready")
            return False

        # Create or get collection
        try:
            if not self.client.collections.exists(COLLECTION_NAME):
                print(f"  📐 Creating collection: {COLLECTION_NAME}")
                self.client.collections.create(
                    name=COLLECTION_NAME,
                    vectorizer_config=None,
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="grantId", data_type=DataType.TEXT),
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="institution", data_type=DataType.TEXT),
                        Property(name="year", data_type=DataType.INT),
                        Property(name="isFQHC", data_type=DataType.BOOL),
                        Property(name="chunkIndex", data_type=DataType.INT),
                        Property(name="chunkType", data_type=DataType.TEXT),
                    ]
                )
                print(f"  ✅ Collection created")
            else:
                print(f"  ✅ Collection {COLLECTION_NAME} already exists")

            self.collection = self.client.collections.get(COLLECTION_NAME)
            return True

        except Exception as e:
            print(f"  ❌ Collection error: {e}")
            return False

    def is_populated(self) -> Tuple[bool, int]:
        """Check if collection has objects, return (has_objects, count)"""
        if self.collection is None:
            return False, 0
        
        try:
            # Try to get object count using multiple methods for version compatibility
            count = 0
            
            # Method 1: Try aggregate.over_all() with different attribute names
            try:
                resp = self.collection.aggregate.over_all()
                if hasattr(resp, 'total_count'):
                    count = resp.total_count
                elif hasattr(resp, 'totalResults'):
                    count = resp.totalResults
                elif hasattr(resp, 'total'):
                    count = resp.total
                else:
                    # Try to access via dictionary if it's a different structure
                    resp_dict = resp.__dict__ if hasattr(resp, '__dict__') else {}
                    count = resp_dict.get('total_count', 
                             resp_dict.get('totalResults', 
                             resp_dict.get('total', 0)))
            except:
                pass
            
            # Method 2: If count still 0, try fetching a few objects
            if count == 0:
                try:
                    objs = self.collection.query.fetch_objects(limit=5)
                    if hasattr(objs, 'objects'):
                        count = len(objs.objects)
                    elif isinstance(objs, dict):
                        count = len(objs.get('objects', []))
                except:
                    pass
            
            # Method 3: Last resort - check if any object exists
            if count == 0:
                try:
                    result = self.collection.query.fetch_objects(limit=1)
                    has_objects = len(result.objects) > 0 if hasattr(result, 'objects') else False
                    if has_objects:
                        print("  ℹ️  Collection has objects (exact count unknown)")
                        return True, -1
                except:
                    pass
            
            if count > 0:
                print(f"  ℹ️  Collection has {count} objects")
                return True, count
                
            return False, 0
            
        except Exception as e:
            print(f"  ⚠️  Could not check collection population: {e}")
            # Assume empty on error
            return False, 0

    def import_chunks(self, chunks_df: pd.DataFrame) -> int:
        """Import chunks with embeddings into Weaviate"""
        print(f"\n  📤 Importing {len(chunks_df)} chunks into Weaviate...")
        total, failed = 0, 0

        if self.collection is None:
            print("  ❌ Collection not initialized")
            return 0

        # Use a smaller batch size for stability
        batch_size = 100
        with self.collection.batch.fixed_size(batch_size=batch_size) as batch:
            for idx, row in chunks_df.iterrows():
                try:
                    # Extract embedding
                    vec = row.get("embedding")
                    if vec is None or (isinstance(vec, float) and np.isnan(vec)):
                        failed += 1
                        continue
                    
                    # Convert embedding to list
                    if isinstance(vec, str):
                        try:
                            # Handle string representation of list
                            vec = json.loads(vec.replace("'", "\""))
                        except:
                            try:
                                vec = ast.literal_eval(vec)
                            except:
                                failed += 1
                                continue
                    elif isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    
                    if not isinstance(vec, list) or len(vec) == 0:
                        failed += 1
                        continue

                    # Prepare properties with defaults for missing values
                    properties = {
                        "text": str(row.get("text", ""))[:5000],
                        "grantId": str(row.get("grant_id", "")),
                        "title": str(row.get("title", ""))[:200],
                        "institution": str(row.get("institution", ""))[:200],
                        "year": int(row.get("year", 2024) if pd.notna(row.get("year")) else 2024),
                        "isFQHC": bool(row.get("has_fqhc_terms", False)),
                        "chunkIndex": int(row.get("chunk_index", 0) if pd.notna(row.get("chunk_index")) else 0),
                        "chunkType": str(row.get("chunk_type", "abstract"))[:50],
                    }

                    batch.add_object(
                        properties=properties,
                        vector=vec
                    )
                    total += 1
                    
                    if total % 500 == 0:
                        print(f"    Imported {total}/{len(chunks_df)}...")

                except Exception as e:
                    failed += 1
                    if failed < 10:  # Only show first few errors
                        print(f"    ⚠️  Error on row {idx}: {e}")

        print(f"  ✅ Imported {total} chunks ({failed} failed)")
        return total

    def hybrid_search(self, query: str, query_vector: List[float],
                      alpha: float = DEFAULT_ALPHA,
                      top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Perform hybrid search with given alpha"""
        if self.collection is None:
            print("  ⚠️  Collection not initialized")
            return []

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
                # Handle different metadata structures across versions
                raw_score = 0
                if hasattr(obj, 'metadata'):
                    if hasattr(obj.metadata, 'score'):
                        raw_score = obj.metadata.score
                    elif isinstance(obj.metadata, dict):
                        raw_score = obj.metadata.get('score', 0)
                
                # Normalize score for display
                display_score = min(raw_score, 1.0) if alpha < 1.0 else raw_score
                
                results.append({
                    "grant_id":    str(p.get("grantId", "")),
                    "title":       str(p.get("title", "")),
                    "institution": str(p.get("institution", "")),
                    "year":        p.get("year", ""),
                    "is_fqhc":     bool(p.get("isFQHC", False)),
                    "text":        str(p.get("text", ""))[:300],
                    "score":       round(display_score, 4),
                    "raw_score":   round(raw_score, 4),
                    "source":      "hybrid_search",
                    "alpha":       alpha,
                })
            return results

        except Exception as e:
            print(f"  ⚠️  Search error: {e}")
            return []

    def close(self):
        """Close Weaviate connection"""
        if self.client:
            self.client.close()
            print("  👋 Weaviate closed")
            self.client = None
            self.collection = None


# ============ GRAPH QUERY ENGINE ============

class GraphQueryEngine:

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        # Pre-compute node types for faster access
        self.grant_nodes = {n for n, d in graph.nodes(data=True) 
                           if d.get('type') == 'grant'}
        self.hub_nodes = {n for n, d in graph.nodes(data=True) 
                         if d.get('type') in ['condition', 'intervention', 'population']}

    def expand(self, seed_grant_ids: List[str], top_k: int = 5) -> List[Dict]:
        """
        Expand seed results via graph traversal with scoring.
        Returns scored grants found through hub nodes.
        """
        if not seed_grant_ids:
            return []
            
        expanded = []
        seen = set(seed_grant_ids)
        scored = defaultdict(float)

        # Limit to top 3 seeds for performance
        for grant_id in seed_grant_ids[:3]:
            if grant_id not in self.graph:
                continue

            # Traverse from seed grant to hub nodes
            for neighbor in self.graph.neighbors(grant_id):
                if neighbor not in self.hub_nodes:
                    continue

                # Get edge weight (default 1.0)
                edge_data = self.graph.get_edge_data(grant_id, neighbor, {})
                edge_weight = float(edge_data.get('weight', 1.0))
                
                # Get hub name for source attribution
                hub_name = self.graph.nodes[neighbor].get('name', neighbor)

                # Traverse from hub to other grants
                for second_hop in self.graph.neighbors(neighbor):
                    if second_hop in seen:
                        continue
                    if second_hop not in self.grant_nodes:
                        continue

                    # Calculate score with decay
                    # Path: grant -> hub (weight) -> grant (default 1.0)
                    score = edge_weight * 0.8  # Decay factor for 2-hop path
                    scored[second_hop] = max(scored[second_hop], score)
                    
                    # Store for later use if this is the first time seeing this path
                    if second_hop not in [r['grant_id'] for r in expanded]:
                        node_data = self.graph.nodes[second_hop]
                        expanded.append({
                            "grant_id":    second_hop,
                            "title":       "",
                            "institution": node_data.get("institution", node_data.get("institute", "")),
                            "year":        node_data.get("year", ""),
                            "is_fqhc":     bool(node_data.get("is_fqhc_focused", False)),
                            "text":        "",
                            "score":       0.0,  # Will update after scoring
                            "source":      f"graph_expansion (via {hub_name})",
                            "shared_hub":  hub_name,
                            "path":        f"{grant_id} → {hub_name} → {second_hop}",
                        })

        # Update scores in expanded list and sort
        result_dict = {r['grant_id']: r for r in expanded}
        for gid, score in scored.items():
            if gid in result_dict:
                result_dict[gid]['score'] = round(score, 3)

        # Sort by score and return top_k
        sorted_results = sorted(result_dict.values(), 
                               key=lambda x: x['score'], reverse=True)
        return sorted_results[:top_k]

    def rfp_match(self, conditions: List[str] = None,
                  interventions: List[str] = None,
                  populations: List[str] = None,
                  fqhc_only: bool = False) -> List[Dict]:
        """
        Find grants via direct graph traversal — no vector search.
        Useful for structured RFP matching.
        """
        scores = defaultdict(float)

        def traverse(prefix, items, weight):
            """Traverse from concept nodes to grants"""
            for item in (items or []):
                # Try exact match first
                node_id = f"{prefix}{item}"
                matches = [node_id]
                
                # If exact match not found, try partial matching
                if node_id not in self.graph:
                    matches = [n for n in self.graph.nodes
                               if n.startswith(prefix) and item.lower() in n.lower()]
                
                for match_id in matches:
                    if match_id not in self.graph:
                        continue
                    for neighbor in self.graph.neighbors(match_id):
                        if neighbor in self.grant_nodes:
                            # Get edge weight
                            edge_data = self.graph.get_edge_data(match_id, neighbor, {})
                            edge_weight = float(edge_data.get('weight', 1.0))
                            scores[neighbor] += weight * edge_weight

        traverse("COND_", conditions, 1.5)
        traverse("INT_", interventions, 1.0)
        traverse("POP_", populations, 1.0)

        # Apply FQHC filter if requested
        if fqhc_only and "FQHC_HUB" in self.graph:
            fqhc_grants = {n for n in self.graph.neighbors("FQHC_HUB")
                          if n in self.grant_nodes}
            scores = {k: v for k, v in scores.items() if k in fqhc_grants}

        # Build results
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
        """Return graph statistics"""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)
        
        for _, d in self.graph.nodes(data=True):
            node_types[d.get("type", "unknown")] += 1
        
        for _, _, d in self.graph.edges(data=True):
            edge_types[d.get("type", "unknown")] += 1
            
        return {
            "nodes":      self.graph.number_of_nodes(),
            "edges":      self.graph.number_of_edges(),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "density":    round(nx.density(self.graph), 6),
        }


# ============ RESULT ENRICHER ============

class ResultEnricher:
    """Joins retrieval results back to abstracts CSV for full metadata"""

    def __init__(self, abstracts_df: pd.DataFrame):
        # Check for duplicate grant_ids
        if abstracts_df['grant_id'].duplicated().any():
            dup_count = abstracts_df['grant_id'].duplicated().sum()
            print(f"  ⚠️  Found {dup_count} duplicate grant_ids in abstracts")
            # Keep first occurrence for each grant_id
            self.abstracts = abstracts_df.drop_duplicates(subset=['grant_id']).set_index("grant_id")
        else:
            self.abstracts = abstracts_df.set_index("grant_id")

    def enrich(self, results: List[Dict]) -> List[Dict]:
        """Add abstract metadata to results"""
        enriched = []
        missing = 0
        
        for r in results:
            gid = r.get("grant_id", "")
            if not gid:
                enriched.append(r)
                continue
                
            if gid in self.abstracts.index:
                row = self.abstracts.loc[gid]
                # Handle potential DataFrame if duplicates remain
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                
                # Update with abstract data (don't overwrite existing values)
                r["title"] = r.get("title") or str(row.get("title", ""))
                r["institution"] = r.get("institution") or str(row.get("institution", ""))
                r["year"] = r.get("year") or int(row.get("year", 0) if pd.notna(row.get("year")) else 0)
                r["is_fqhc"] = bool(row.get("has_fqhc_terms", False)) or r.get("is_fqhc", False)
                r["abstract_snippet"] = str(row.get("abstract", ""))[:300]
            else:
                missing += 1
                
            enriched.append(r)
        
        if missing > 0:
            print(f"  ⚠️  {missing} results missing from abstracts")
            
        return enriched


# ============ GRANT QUERY PIPELINE ============

class GrantQueryPipeline:
    """
    Main pipeline class. Call setup() once, then query() as many times as needed.
    Weaviate stays alive between queries.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, top_k: int = DEFAULT_TOP_K):
        self.alpha      = alpha
        self.top_k      = top_k
        self.weaviate   = WeaviateManager()
        self.graph_engine = None
        self.enricher   = None
        self.model      = None
        self._ready     = False
        self._setup_time = 0

    def setup(self) -> bool:
        """
        Boot everything. Call once before querying.
        Returns True if setup successful.
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
        
        try:
            abstracts_df = pd.read_csv(PATHS["abstracts"])
            print(f"  ✅ {len(abstracts_df)} abstracts loaded")
            self.enricher = ResultEnricher(abstracts_df)
        except Exception as e:
            print(f"  ❌ Failed to load abstracts: {e}")
            return False

        # 2. Load embedding model
        print("\n🧠 Loading embedding model...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"  ✅ {EMBEDDING_MODEL} loaded")
        except Exception as e:
            print(f"  ❌ Model load failed: {e}")
            return False

        # 3. Start Weaviate
        if not self.weaviate.start():
            print("  ❌ Weaviate startup failed")
            return False

        # 4. Check if Weaviate is populated
        populated, count = self.weaviate.is_populated()
        
        # 5. Import chunks if needed
        if not populated or count == 0:
            print("\n  📥 Weaviate needs data — importing chunks...")
            chunks_path = PATHS["chunks"]
            if not os.path.exists(chunks_path):
                print(f"  ❌ Chunks not found: {chunks_path}")
                print("     Run phase3_document_rag.py first")
                return False

            try:
                print(f"  ⏳ Reading {chunks_path}...")
                chunks_df = pd.read_csv(chunks_path)
                print(f"  ✅ Loaded {len(chunks_df)} chunks")

                # Merge title/institution from abstracts if missing
                if "title" not in chunks_df.columns or "institution" not in chunks_df.columns:
                    meta = abstracts_df[["grant_id", "title", "institution", "has_fqhc_terms"]].copy()
                    chunks_df = chunks_df.merge(meta, on="grant_id", how="left")

                imported = self.weaviate.import_chunks(chunks_df)
                if imported == 0:
                    print("  ❌ No chunks imported")
                    return False
                    
                print(f"  ✅ Successfully imported {imported} chunks")
                
            except Exception as e:
                print(f"  ❌ Import failed: {e}")
                return False
        else:
            print(f"  ✅ Weaviate already populated with {count} objects — skipping import")

        # 6. Load knowledge graph
        print("\n🕸️  Loading knowledge graph...")
        graph = self._load_graph()
        if graph is None:
            print("  ⚠️  No graph found — run phase5_knowledge_graph.py first")
            print("       Graph expansion will be unavailable")
        else:
            self.graph_engine = GraphQueryEngine(graph)
            stats = self.graph_engine.graph_stats()
            print(f"  ✅ Graph loaded: {stats['nodes']} nodes, {stats['edges']} edges")
            print(f"     Node types: {stats['node_types']}")

        elapsed = round(time.time() - start, 1)
        self._ready = True
        self._setup_time = elapsed
        
        print(f"\n✅ Pipeline ready ({elapsed}s)")
        print(f"   Alpha: {self.alpha} (BM25/vector blend)")
        print(f"   Top-K: {self.top_k}")
        print(f"   Graph expansion: {'enabled' if self.graph_engine else 'disabled'}")

        return True

    def query(self, query_text: str,
              alpha: float = None,
              top_k: int = None,
              use_graph: bool = True,
              fqhc_only: bool = False,
              verbose: bool = True) -> List[Dict]:
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

        if not self.model:
            print("❌ No embedding model available")
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
        try:
            query_vec = self.model.encode(query_text).tolist()
        except Exception as e:
            print(f"❌ Failed to embed query: {e}")
            return []

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
            seed_ids = [r["grant_id"] for r in hybrid_results if r.get("grant_id")]
            if seed_ids:
                graph_results = self.graph_engine.expand(seed_ids, top_k=5)
                if verbose:
                    print(f"🕸️  Graph expansion: {len(graph_results)} additional results")

        # 4. Combine and deduplicate
        all_results = hybrid_results + graph_results
        seen, unique = set(), []
        for r in all_results:
            gid = r.get("grant_id")
            if gid and gid not in seen:
                seen.add(gid)
                unique.append(r)

        # 5. Filter FQHC if requested
        if fqhc_only:
            unique = [r for r in unique if r.get("is_fqhc")]
            if verbose:
                print(f"🏥 FQHC filter: {len(unique)} results")

        # 6. Enrich with abstract metadata
        if self.enricher:
            unique = self.enricher.enrich(unique)

        # 7. Print results
        if verbose:
            self._print_results(unique, query_text)

        return unique

    def rfp_query(self,
                  conditions: List[str] = None,
                  interventions: List[str] = None,
                  populations: List[str] = None,
                  free_text: str = None,
                  fqhc_only: bool = False,
                  verbose: bool = True) -> List[Dict]:
        """
        Structured RFP matching query.
        Combines graph traversal (structured) + hybrid search (free text).
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
        start = time.time()

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
        search_query = free_text
        if not search_query and any([conditions, interventions, populations]):
            # Build query from structured terms
            terms = []
            if conditions: terms.extend(conditions)
            if interventions: terms.extend(interventions)
            if populations: terms.extend(populations)
            search_query = " ".join(terms)
            
        if search_query and self.model:
            query_vec = self.model.encode(search_query).tolist()
            hybrid_results = self.weaviate.hybrid_search(
                query=search_query,
                query_vector=query_vec,
                alpha=self.alpha,
                top_k=self.top_k
            )
            if verbose:
                print(f"📊 Hybrid search: {len(hybrid_results)} results")
            all_results.extend(hybrid_results)

        # Deduplicate
        seen, unique = set(), []
        for r in all_results:
            gid = r.get("grant_id")
            if gid and gid not in seen:
                seen.add(gid)
                # Graph results come first in list, preserve their source
                unique.append(r)

        # Apply FQHC filter if requested
        if fqhc_only:
            unique = [r for r in unique if r.get("is_fqhc")]

        # Enrich with metadata
        if self.enricher:
            unique = self.enricher.enrich(unique)

        if verbose:
            print(f"\n⏱️  Total time: {time.time()-start:.2f}s")
            self._print_results(unique, search_query or "structured query")

        return unique

    def compare_approaches(self, query_text: str) -> Dict:
        """
        Show side-by-side comparison of BM25, vector, hybrid, and graph-expanded.
        Useful for demo.
        """
        if not self._ready or not self.model:
            print("❌ Pipeline not ready")
            return {}

        print(f"\n{'='*70}")
        print(f"📊 APPROACH COMPARISON: '{query_text}'")
        print(f"{'='*70}")

        query_vec = self.model.encode(query_text).tolist()
        results = {}

        # Test different alpha values
        alphas_to_test = [(0.0, "BM25 only (α=0.0)"), 
                          (0.25, "Hybrid (α=0.25)"),
                          (0.5, "Hybrid (α=0.5)"),
                          (1.0, "Vector only (α=1.0)")]

        for alpha, label in alphas_to_test:
            r = self.weaviate.hybrid_search(query_text, query_vec,
                                            alpha=alpha, top_k=5)
            results[label] = r
            print(f"\n  {label}:")
            for i, x in enumerate(r[:3], 1):
                title = x.get("title", "")[:50] or x.get("text", "")[:50]
                print(f"    {i}. [{x['score']:.4f}] {x.get('grant_id', 'N/A')} — {title}...")

        # Graph expansion on top of best hybrid
        if self.graph_engine:
            hybrid = self.weaviate.hybrid_search(query_text, query_vec,
                                                 alpha=0.25, top_k=5)
            seed_ids = [r["grant_id"] for r in hybrid if r.get("grant_id")]
            if seed_ids:
                expanded = self.graph_engine.expand(seed_ids, top_k=5)
                results["Graph expansion"] = expanded
                print(f"\n  Graph expansion (additional grants via hub nodes):")
                for i, x in enumerate(expanded[:3], 1):
                    print(f"    {i}. {x.get('grant_id', 'N/A')} — via {x.get('shared_hub', '?')} (score: {x.get('score', 0):.3f})")

        return results

    def close(self):
        """Shut down Weaviate. Call when done."""
        self.weaviate.close()
        self._ready = False
        print("  ✅ Pipeline closed")

    # ---- Helpers ----

    def _load_graph(self) -> Optional[nx.Graph]:
        """Load graph from pickle or GML with proper error handling"""
        # Try pickle first (faster)
        if os.path.exists(PATHS["graph_pkl"]):
            try:
                with open(PATHS["graph_pkl"], "rb") as f:
                    graph = pickle.load(f)
                    print(f"  ✅ Graph loaded from pickle: {graph.number_of_nodes()} nodes")
                    return graph
            except Exception as e:
                print(f"  ⚠️  Pickle load failed: {e}, trying GML...")

        # Fall back to GML
        if os.path.exists(PATHS["graph_gml"]):
            try:
                graph = nx.read_gml(PATHS["graph_gml"])
                print(f"  ✅ Graph loaded from GML: {graph.number_of_nodes()} nodes")
                return graph
            except Exception as e:
                print(f"  ⚠️  GML load failed: {e}")

        print("  ⚠️  No graph found")
        return None

    def _print_results(self, results: List[Dict], query: str):
        """Pretty print results"""
        print(f"\n{'─'*70}")
        print(f"📋 RESULTS ({len(results)} total)")
        print(f"{'─'*70}")

        for i, r in enumerate(results[:self.top_k], 1):
            # Determine source icon
            source = r.get("source", "")
            if "graph" in source.lower():
                source_icon = "🕸️"
            elif "hybrid" in source.lower():
                source_icon = "🔍"
            else:
                source_icon = "📄"

            print(f"\n{i}. {source_icon}  [score: {r.get('score', 0):.4f}]  {source}")
            print(f"   Grant ID:    {r.get('grant_id', 'N/A')}")
            
            if r.get("title"):
                print(f"   Title:       {r['title'][:80]}")
            if r.get("institution"):
                print(f"   Institution: {r['institution'][:60]}")
            if r.get("year"):
                print(f"   Year:        {r['year']}")
            if r.get("is_fqhc"):
                print(f"   ✅ FQHC-relevant")
            if r.get("abstract_snippet"):
                print(f"   Abstract:    {r['abstract_snippet']}...")


# ============ INTERACTIVE MODE ============

def interactive_mode(pipeline: GrantQueryPipeline):
    """Run interactive query session"""
    print("\n" + "="*70)
    print("💬 INTERACTIVE QUERY MODE")
    print("="*70)
    print("Commands:")
    print("  <query>              — hybrid + graph search")
    print("  /rfp <query>         — structured RFP match")
    print("  /compare <query>     — compare all approaches")
    print("  /fqhc <query>        — FQHC-only results")
    print("  /alpha <0-1> <query> — custom alpha")
    print("  /stats               — show pipeline stats")
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

        elif user_input.lower() == "/stats":
            print(f"\n📊 Pipeline Statistics:")
            print(f"   Setup time: {pipeline._setup_time}s")
            print(f"   Alpha: {pipeline.alpha}")
            print(f"   Top-K: {pipeline.top_k}")
            print(f"   Graph engine: {'Yes' if pipeline.graph_engine else 'No'}")
            continue

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
                    if 0 <= alpha <= 1:
                        q = parts[2]
                        pipeline.query(q, alpha=alpha)
                    else:
                        print("  Alpha must be between 0 and 1")
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

    # Validate alpha
    if not 0 <= args.alpha <= 1:
        print("❌ Alpha must be between 0 and 1")
        sys.exit(1)

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

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()