# ============================================================================
# query_pipeline.py - UNIFIED QUERY PIPELINE (COLAB-READY)
# ============================================================================

print("=" * 70)
print("\U0001f680 GRANT QUERY PIPELINE: Hybrid RAG + Knowledge Graph")
print("=" * 70)

import os, sys, ast, json, time, argparse, pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

GRAPH_CONFIG_PATH = "./graph_config.json"


def _load_augmentation_config() -> dict:
    """
    Load augmentation params from graph_config.json if present.
    Falls back to system defaults silently — graph augmentation is always on.
    """
    defaults = {
        "overlap_bonus":      0.10,
        "graph_only_penalty": 0.5,
        "min_graph_hits":     3,
    }
    if not os.path.exists(GRAPH_CONFIG_PATH):
        return defaults
    try:
        with open(GRAPH_CONFIG_PATH) as f:
            cfg = json.load(f)
        aug = {k: v for k, v in cfg.get("augmentation", {}).items()
               if not k.startswith("_")}
        return {**defaults, **aug}
    except Exception:
        return defaults


PATHS = {
    "abstracts": "./phase2_output/nih_research_abstracts.csv",
    "chunks":    "./phase3_results/document_chunks_with_embeddings.csv",
    "graph_pkl": "./phase5_graph_store/graph.pkl",
    "graph_gml": "./phase5_knowledge_graph.gml",
}
COLLECTION_NAME = "GrantChunk"
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
DEFAULT_ALPHA   = 0.25
DEFAULT_TOP_K   = 10
PDF_GRANT_DIR   = "./grant_documents/"


def build_source_url(grant_id: str, data_source: str = "") -> str:
    """
    Returns a local file path for PDF corpus grants,
    NIH Reporter URL for NIH abstract grants.
    Checks disk first so graph-expanded results also resolve correctly.
    """
    pdf_path = os.path.join(PDF_GRANT_DIR, f"{grant_id}.pdf")
    if data_source == "pdf_ingestion" or os.path.exists(pdf_path):
        return pdf_path
    return f"https://reporter.nih.gov/search-results?query={grant_id}"


class WeaviateManager:
    def __init__(self):
        self.client     = None
        self.collection = None

    def start(self) -> bool:
        try:
            import weaviate
            from weaviate.classes.config import Property, DataType
        except ImportError:
            print("  \u274c weaviate-client not installed")
            return False

        print("\n\U0001f50c Starting Weaviate...")

        for ports in [{"port": 8079, "grpc_port": 50050},
                      {"port": 8080, "grpc_port": 50051}]:
            try:
                self.client = weaviate.connect_to_embedded(**ports)
                print(f"  \u2705 Weaviate embedded started on port {ports['port']}")
                break
            except Exception as e:
                err = str(e).lower()
                if "already" in err or "listening" in err:
                    try:
                        self.client = weaviate.connect_to_local(**ports)
                        print(f"  \u2705 Connected to existing Weaviate on port {ports['port']}")
                        break
                    except:
                        continue
                continue
        else:
            print("  \u274c Could not connect to Weaviate")
            return False

        if not self.client.is_ready():
            print("  \u274c Weaviate not ready")
            return False

        try:
            if not self.client.collections.exists(COLLECTION_NAME):
                print(f"  \U0001f4d0 Creating collection: {COLLECTION_NAME}")
                self.client.collections.create(
                    name=COLLECTION_NAME,
                    vectorizer_config=None,
                    properties=[
                        Property(name="text",        data_type=DataType.TEXT),
                        Property(name="grantId",     data_type=DataType.TEXT),
                        Property(name="title",       data_type=DataType.TEXT),
                        Property(name="institution", data_type=DataType.TEXT),
                        Property(name="year",        data_type=DataType.INT),
                        Property(name="isFQHC",      data_type=DataType.BOOL),
                        Property(name="chunkIndex",  data_type=DataType.INT),
                        Property(name="chunkType",   data_type=DataType.TEXT),
                        Property(name="sectionType", data_type=DataType.TEXT),
                        Property(name="dataSource",  data_type=DataType.TEXT),
                    ]
                )
            else:
                print(f"  \u2705 Collection {COLLECTION_NAME} already exists")
            self.collection = self.client.collections.get(COLLECTION_NAME)
            return True
        except Exception as e:
            print(f"  \u274c Collection error: {e}")
            return False

    def is_populated(self) -> Tuple[bool, int]:
        if not self.collection:
            return False, 0
        try:
            resp  = self.collection.aggregate.over_all()
            count = getattr(resp, "total_count", 0) or 0
            if count == 0:
                res = self.collection.query.fetch_objects(limit=1)
                if hasattr(res, "objects") and res.objects:
                    print("  \u2139\ufe0f  Collection has objects (count unknown)")
                    return True, -1
            if count > 0:
                print(f"  \u2139\ufe0f  Collection has {count} objects")
                return True, count
            return False, 0
        except Exception as e:
            print(f"  \u26a0\ufe0f  Could not check population: {e}")
            return False, 0

    def import_chunks(self, chunks_df: pd.DataFrame) -> int:
        print(f"\n  \U0001f4e4 Importing {len(chunks_df)} chunks...")
        total, failed = 0, 0
        with self.collection.batch.fixed_size(batch_size=100) as batch:
            for idx, row in chunks_df.iterrows():
                try:
                    vec = row.get("embedding")
                    if vec is None or (isinstance(vec, float) and np.isnan(vec)):
                        failed += 1; continue
                    if isinstance(vec, str):
                        try: vec = json.loads(vec.replace("'", "\""))
                        except: vec = ast.literal_eval(vec)
                    elif isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    if not isinstance(vec, list) or not vec:
                        failed += 1; continue
                    section = str(row.get("section_type",
                                  row.get("chunk_type", "general")))
                    batch.add_object(
                        properties={
                            "text":        str(row.get("text", ""))[:5000],
                            "grantId":     str(row.get("grant_id", "")),
                            "title":       str(row.get("title", ""))[:200],
                            "institution": str(row.get("institution", ""))[:200],
                            "year":        int(row.get("year", 2024) if pd.notna(row.get("year")) else 2024),
                            "isFQHC":      bool(row.get("has_fqhc_terms", False)),
                            "chunkIndex":  int(row.get("chunk_index", 0) if pd.notna(row.get("chunk_index")) else 0),
                            "chunkType":   section,
                            "sectionType": section,
                            "dataSource":  str(row.get("data_source", "")),
                        },
                        vector=vec
                    )
                    total += 1
                    if total % 500 == 0: print(f"    {total}/{len(chunks_df)}...")
                except Exception as e:
                    failed += 1
        print(f"  \u2705 Imported {total} ({failed} failed)")
        return total

    SECTION_ALIASES = {
        "aims":    "specific_aims",
        "sig":     "significance",
        "innov":   "innovation",
        "bg":      "background",
        "summary": "project_summary",
        "specific_aims":   "specific_aims",
        "significance":    "significance",
        "innovation":      "innovation",
        "approach":        "approach",
        "methods":         "methods",
        "background":      "background",
        "project_summary": "project_summary",
    }

    def hybrid_search(self, query: str, query_vector: List[float],
                      alpha: float = DEFAULT_ALPHA,
                      top_k: int = DEFAULT_TOP_K,
                      section_filter: str = None,
                      fqhc_only: bool = False,
                      grant_id_filter: List[str] = None) -> List[Dict]:
        if not self.collection:
            return []
        try:
            from weaviate.classes.query import MetadataQuery, Filter

            section = self.SECTION_ALIASES.get(section_filter, section_filter)

            wv_filter = None
            if section:
                wv_filter = Filter.by_property("sectionType").equal(section)
            if fqhc_only:
                fqhc_f    = Filter.by_property("isFQHC").equal(True)
                wv_filter = (wv_filter & fqhc_f) if wv_filter else fqhc_f
            if grant_id_filter:
                id_f      = Filter.by_property("grantId").contains_any(grant_id_filter)
                wv_filter = (wv_filter & id_f) if wv_filter else id_f

            kwargs = dict(
                query=query,
                alpha=alpha,
                vector=query_vector if alpha > 0 else None,
                limit=top_k,
                return_metadata=MetadataQuery(score=True),
                return_properties=["grantId","title","institution","year",
                                   "isFQHC","text","sectionType","chunkType",
                                   "chunkIndex","dataSource"],
            )
            if wv_filter:
                kwargs["filters"] = wv_filter

            resp    = self.collection.query.hybrid(**kwargs)
            results = []
            for obj in resp.objects:
                p           = obj.properties
                score       = getattr(obj.metadata, "score", 0) or 0
                grant_id    = str(p.get("grantId", ""))
                data_source = str(p.get("dataSource", ""))

                results.append({
                    "grant_id":     grant_id,
                    "title":        str(p.get("title", "")),
                    "institution":  str(p.get("institution", "")),
                    "year":         p.get("year", ""),
                    "is_fqhc":      bool(p.get("isFQHC", False)),
                    "text":         str(p.get("text", "")),
                    "score":        round(min(score, 1.0), 4),
                    "section_type": str(p.get("sectionType", p.get("chunkType", ""))),
                    "chunk_index":  int(p.get("chunkIndex", 0)),
                    "source":       "hybrid_search",
                    "alpha":        alpha,
                    "data_source":  data_source,
                    "source_url":   build_source_url(grant_id, data_source),
                })
            return results
        except Exception as e:
            print(f"  \u26a0\ufe0f  Search error: {e}")
            return []

    def close(self):
        if self.client:
            try: self.client.close()
            except: pass
            print("  \U0001f44b Weaviate closed")
            self.client = self.collection = None


class GraphQueryEngine:
    def __init__(self, graph: nx.Graph):
        self.graph       = graph
        self.grant_nodes = {n for n, d in graph.nodes(data=True) if d.get("type") == "grant"}
        self.hub_nodes   = {n for n, d in graph.nodes(data=True)
                            if d.get("type") in ["condition", "intervention", "population"]}

    def expand(self, seed_ids: List[str], top_k: int = 5,
               section_hint: str = None) -> List[Dict]:
        seen, scored, paths = set(seed_ids), defaultdict(float), {}
        for gid in seed_ids[:3]:
            if gid not in self.graph: continue
            for hub in self.graph.neighbors(gid):
                if hub not in self.hub_nodes: continue
                hub_name = self.graph.nodes[hub].get("name", hub)
                w        = float(self.graph.get_edge_data(gid, hub, {}).get("weight", 1.0))
                for hop in self.graph.neighbors(hub):
                    if hop in seen or hop not in self.grant_nodes: continue
                    s = w * 0.8
                    if s > scored[hop]:
                        scored[hop] = s
                        paths[hop]  = hub_name
        results = []
        for g, s in sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            nd = self.graph.nodes[g]
            results.append({
                "grant_id":     g,
                "title":        "",
                "text":         "",
                "institution":  nd.get("institution", nd.get("institute", "")),
                "year":         nd.get("year", ""),
                "is_fqhc":      bool(nd.get("is_fqhc_focused", False)),
                "score":        round(s, 3),
                "source":       f"graph_expansion (via {paths[g]})",
                "shared_hub":   paths[g],
                "section_hint": section_hint or "",
                "source_url":   build_source_url(g),
            })
        return results

    def rfp_match(self, conditions=None, interventions=None,
                  populations=None, fqhc_only=False) -> List[Dict]:
        scores = defaultdict(float)
        def traverse(prefix, items, weight):
            for item in (items or []):
                nid     = f"{prefix}{item}"
                matches = [nid] if nid in self.graph else [
                    n for n in self.graph.nodes
                    if n.startswith(prefix) and item.lower() in n.lower()
                ]
                for mid in matches:
                    for nb in self.graph.neighbors(mid):
                        if nb in self.grant_nodes:
                            w = float(self.graph.get_edge_data(mid, nb, {}).get("weight", 1.0))
                            scores[nb] += weight * w
        traverse("COND_", conditions,    1.5)
        traverse("INT_",  interventions, 1.0)
        traverse("POP_",  populations,   1.0)
        if fqhc_only and "FQHC_HUB" in self.graph:
            fqhc   = {n for n in self.graph.neighbors("FQHC_HUB") if n in self.grant_nodes}
            scores = {k: v for k, v in scores.items() if k in fqhc}
        results = []
        for g, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            nd = self.graph.nodes.get(g, {})
            results.append({
                "grant_id":  g,
                "title":     "",
                "text":      "",
                "institution": nd.get("institution", nd.get("institute", "")),
                "year":      nd.get("year", ""),
                "is_fqhc":   bool(nd.get("is_fqhc_focused", False)),
                "score":     round(s, 2),
                "source":    "graph_rfp_match",
                "source_url": build_source_url(g),
            })
        return results

    _MATCH_STOPWORDS = frozenset([
        "health", "care", "program", "service", "intervention", "community",
        "patient", "disease", "disorder", "based", "management", "treatment",
        "prevention", "high", "risk", "population", "setting", "primary",
        "chronic", "mental", "behavioral",
    ])

    _LABEL_ALIASES = {
        "chw":    ["community health worker", "promotora", "lay health",
                   "outreach worker", "health promoter", "peer navigator"],
        "hiv":    ["hiv", "aids", "prep", "antiretroviral"],
        "prep":   ["prep", "pre-exposure"],
        "cbt":    ["cognitive behavioral", "counseling"],
        "sdoh":   ["social determinants", "social needs"],
        "fqhc":   ["federally qualified", "community health center", "fqhc"],
    }

    def match_nodes_to_query(self, query: str):
        """
        Dynamically match graph concept nodes to a query string.
        Works on any corpus — no hardcoded condition/intervention/population lists.
        Uses significant-word matching (ignores generic stopwords like "health",
        "care") to avoid false positives across unrelated topics.
        Returns (conditions, interventions, populations) as label string lists
        suitable for passing to rfp_match().
        """
        q = query.lower().replace("_", " ")
        q_sig = [w for w in q.split()
                 if len(w) > 2 and w not in self._MATCH_STOPWORDS]
        conditions, interventions, populations = [], [], []
        for node in self.graph.nodes:
            ntype = self.graph.nodes[node].get("type", "")
            if ntype not in ("condition", "intervention", "population"):
                continue
            label = node.split("_", 1)[-1].lower().replace("_", " ")
            label_sig = [w for w in label.split()
                         if len(w) > 2 and w not in self._MATCH_STOPWORDS]

            matched = False
            # Full label substring in query
            if label in q:
                matched = True
            # Short acronym / alias match
            elif label in self._LABEL_ALIASES:
                if any(alias in q for alias in self._LABEL_ALIASES[label]):
                    matched = True
            elif len(label) <= 4 and label in q.split():
                matched = True
            # Significant word overlap
            elif label_sig and q_sig and \
                  any(lw == qw or (len(lw) > 4 and (lw in qw or qw in lw))
                     for lw in label_sig for qw in q_sig):
                matched = True

            if matched:
                raw = node.split("_", 1)[-1]
                if ntype == "condition":      conditions.append(raw)
                elif ntype == "intervention": interventions.append(raw)
                elif ntype == "population":   populations.append(raw)
        return conditions, interventions, populations

    def graph_stats(self) -> Dict:
        counts = defaultdict(int)
        for _, d in self.graph.nodes(data=True):
            counts[d.get("type", "unknown")] += 1
        return {"nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "node_types": dict(counts)}


class ResultEnricher:
    def __init__(self, abstracts_df: pd.DataFrame):
        dups = abstracts_df["grant_id"].duplicated().sum()
        if dups:
            print(f"  \u26a0\ufe0f  Found {dups} duplicate grant_ids")
            abstracts_df = abstracts_df.drop_duplicates(subset=["grant_id"])
        self.abstracts = abstracts_df.set_index("grant_id")

    def enrich(self, results: List[Dict]) -> List[Dict]:
        out = []
        for r in results:
            gid = r.get("grant_id", "")
            if gid and gid in self.abstracts.index:
                row = self.abstracts.loc[gid]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                r["title"]            = r.get("title") or str(row.get("title", ""))
                r["institution"]      = r.get("institution") or str(row.get("institution", ""))
                r["year"]             = r.get("year") or int(row.get("year", 0) if pd.notna(row.get("year")) else 0)
                r["is_fqhc"]          = bool(row.get("has_fqhc_terms", False)) or r.get("is_fqhc", False)
                r["abstract_snippet"] = str(row.get("abstract", ""))[:300]
            out.append(r)
        return out


SYNTHETIC_PREFIXES = ("FQHC_SYNTH_", "SYNTH_")


class GrantQueryPipeline:
    def __init__(self, alpha=DEFAULT_ALPHA, top_k=DEFAULT_TOP_K,
                 exclude_synthetic=True):
        self.alpha             = alpha
        self.top_k             = top_k
        self.exclude_synthetic = exclude_synthetic
        self.weaviate          = WeaviateManager()
        self.graph_engine      = None
        self.enricher          = None
        self.model             = None
        self._ready            = False
        self._setup_time       = 0
        # Augmentation params — loaded from graph_config.json, falls back to defaults.
        # Edit graph_config.json to customise overlap_bonus, graph_only_penalty,
        # and min_graph_hits for your corpus (advanced mode).
        _aug = _load_augmentation_config()
        self._overlap_bonus      = _aug["overlap_bonus"]
        self._graph_only_penalty = _aug["graph_only_penalty"]
        self._min_graph_hits     = _aug["min_graph_hits"]

    def setup(self) -> bool:
        print("\n" + "=" * 70)
        print("\u2699\ufe0f  SETTING UP QUERY PIPELINE")
        print("=" * 70)
        t = time.time()

        print("\n\U0001f4e6 Loading abstracts...")
        if not os.path.exists(PATHS["abstracts"]):
            print(f"  \u274c Not found: {PATHS['abstracts']}"); return False
        try:
            df            = pd.read_csv(PATHS["abstracts"])
            self.enricher = ResultEnricher(df)
            print(f"  \u2705 {len(df)} abstracts loaded")
        except Exception as e:
            print(f"  \u274c {e}"); return False

        print("\n\U0001f9e0 Loading embedding model...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"  \u2705 {EMBEDDING_MODEL} loaded")
        except Exception as e:
            print(f"  \u274c {e}"); return False

        if not self.weaviate.start(): return False

        populated, count = self.weaviate.is_populated()
        if not populated:
            print("\n  \U0001f4e5 Weaviate empty — importing chunks...")
            cp = PATHS["chunks"]
            if not os.path.exists(cp):
                print(f"  \u274c Not found: {cp}"); return False
            try:
                chunks = pd.read_csv(cp)
                if "title" not in chunks.columns:
                    meta   = df[["grant_id", "title", "institution", "has_fqhc_terms"]]
                    chunks = chunks.merge(meta, on="grant_id", how="left")
                if self.weaviate.import_chunks(chunks) == 0: return False
            except Exception as e:
                print(f"  \u274c {e}"); return False
        else:
            print(f"  \u2705 Weaviate populated ({count} objects) — skipping import")

        print("\n\U0001f578\ufe0f  Loading knowledge graph...")
        graph = self._load_graph()
        if graph:
            self.graph_engine = GraphQueryEngine(graph)
            s = self.graph_engine.graph_stats()
            print(f"  \u2705 Graph loaded: {s['nodes']} nodes, {s['edges']} edges")
            print(f"     Node types: {s['node_types']}")
        else:
            print("  \u26a0\ufe0f  No graph — run phase5_knowledge_graph.py first")

        self._setup_time = round(time.time() - t, 1)
        self._ready      = True
        print(f"\n\u2705 Pipeline ready ({self._setup_time}s)")
        print(f"   Alpha: {self.alpha} | Top-K: {self.top_k} | "
              f"Graph: {'enabled' if self.graph_engine else 'disabled'}")
        return True

    def query(self, query_text, alpha=None, top_k=None,
              use_graph=True, fqhc_only=False,
              section_filter: str = None,
              verbose=True) -> List[Dict]:
        if not self._ready:
            print("\u274c Call setup() first."); return []
        alpha = alpha if alpha is not None else self.alpha
        top_k = top_k if top_k is not None else self.top_k

        section = WeaviateManager.SECTION_ALIASES.get(section_filter, section_filter)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"\U0001f50d QUERY: {query_text}")
            sec_str = f"  section={section}" if section else ""
            print(f"   \u03b1={alpha}  top_k={top_k}  graph={'on' if use_graph else 'off'}"
                  f"  fqhc_only={fqhc_only}{sec_str}")
            print(f"{'=' * 70}")
        t = time.time()

        encode_text = f"{section.replace('_',' ')}: {query_text}" if section else query_text
        query_vec   = self.model.encode(encode_text).tolist()

        hybrid = self.weaviate.hybrid_search(
            query_text, query_vec, alpha, top_k,
            section_filter=section_filter,
            fqhc_only=fqhc_only,
        )
        if verbose:
            sec_label = f" [{section}]" if section else ""
            print(f"\n\U0001f4ca Hybrid search: {len(hybrid)} results "
                  f"(\u03b1={alpha}{sec_label}, {time.time()-t:.2f}s)")

        graph_r = []
        if use_graph and self.graph_engine and hybrid:
            seeds   = [r["grant_id"] for r in hybrid if r.get("grant_id")]
            graph_r = self.graph_engine.expand(seeds, top_k=5, section_hint=section)
            if verbose: print(f"\U0001f578\ufe0f  Graph expansion: {len(graph_r)} additional results")

        # ── FIX: score-aware merge so graph results compete on merit ──────
        # Build a map of hybrid scores for overlap bonus application.
        hybrid_score_map = {r["grant_id"]: r["score"] for r in hybrid if r.get("grant_id")}

        all_r        = hybrid + graph_r
        seen, unique = set(), []
        for r in all_r:
            g = r.get("grant_id")
            if g and g not in seen:
                seen.add(g)
                # Graph result that overlaps with hybrid — apply overlap bonus
                # to the hybrid score so the combined signal is reflected.
                if r.get("source", "").startswith("graph") and g in hybrid_score_map:
                    r = dict(r)  # avoid mutating the original
                    r["score"] = round(hybrid_score_map[g] * (1 + self._overlap_bonus), 4)
                unique.append(r)

        # Re-sort merged list by score so graph-only results compete on their
        # path-weighted score rather than always trailing hybrid results.
        unique = sorted(unique, key=lambda x: x.get("score", 0), reverse=True)
        # ─────────────────────────────────────────────────────────────────

        unique = self._filter_synthetic(unique)
        if self.enricher: unique = self.enricher.enrich(unique)
        if verbose:        self._print_results(unique)
        return unique

    def discover_grants(self, topic: str, top_k: int = 10,
                        fqhc_only: bool = False,
                        parsed: Dict = None) -> List[str]:
        """
        Corpus-level grant discovery combining hybrid search + knowledge graph.

        Hybrid search captures lexical/semantic similarity.
        Graph rfp_match captures structural similarity via shared condition,
        intervention, and population nodes — complementary signal.

        Both pools are scored, normalized to [0,1], then merged into a single
        ranked pool. Best top_k grant_ids returned (score-weighted, no fixed
        per-source slot allocation). Source tag preserved for display.

        Args:
            topic:   free-text description for hybrid search encoding
            top_k:   number of grant_ids to return
            fqhc_only: restrict to FQHC-connected grants in graph
            parsed:  optional dict from ApplicationDescriptionParser with
                     conditions / interventions / populations lists for graph
        Returns:
            List[str] of grant_ids ranked by combined score (best first)
        """
        if not self._ready:
            return []

        # ── 1. Hybrid search pool ─────────────────────────────────────────
        query_vec    = self.model.encode(topic).tolist()
        hybrid_raw   = self.weaviate.hybrid_search(
            topic, query_vec,
            alpha     = self.alpha,
            top_k     = top_k * 4,
            fqhc_only = fqhc_only,
        )
        hybrid_raw = self._filter_synthetic(hybrid_raw)

        # Deduplicate to best score per grant
        hybrid_scores: Dict[str, float] = {}
        for r in hybrid_raw:
            gid = r.get("grant_id")
            if gid:
                hybrid_scores[gid] = max(hybrid_scores.get(gid, 0.0), r["score"])

        # ── 2. Graph rfp_match pool ───────────────────────────────────────
        graph_scores: Dict[str, float] = {}
        if self.graph_engine:
            conditions    = (parsed or {}).get("conditions",    [])
            interventions = (parsed or {}).get("interventions", [])
            populations   = (parsed or {}).get("populations",   [])

            # Fall back to keyword extraction from topic if no parsed dict
            if not any([conditions, interventions, populations]):
                from collections import Counter
                words = topic.lower().split()
                COND_HINTS = ["diabetes","hypertension","hiv","cancer","obesity",
                              "asthma","depression","opioid","maternal","pediatric"]
                INT_HINTS  = ["chw","telehealth","screening","education",
                              "harm reduction","care management","peer support"]
                POP_HINTS  = ["latino","hispanic","african american","black",
                              "low income","rural","elderly","lgbtq","immigrant"]
                conditions    = [h for h in COND_HINTS if h in topic.lower()]
                interventions = [h for h in INT_HINTS  if h in topic.lower()]
                populations   = [h for h in POP_HINTS  if h in topic.lower()]

            if any([conditions, interventions, populations]):
                graph_raw = self.graph_engine.rfp_match(
                    conditions    = conditions,
                    interventions = interventions,
                    populations   = populations,
                    fqhc_only     = fqhc_only,
                )
                graph_raw = self._filter_synthetic(graph_raw)
                for r in graph_raw:
                    gid = r.get("grant_id")
                    if gid:
                        graph_scores[gid] = max(
                            graph_scores.get(gid, 0.0), r["score"]
                        )

        # ── 3. Hybrid-primary, graph-augmented merge ─────────────────────
        # Hybrid scores are the primary signal — graph augments, never re-ranks.
        # Params loaded from graph_config.json (user-customisable advanced mode).
        combined: Dict[str, Dict] = {}

        for gid, score in hybrid_scores.items():
            combined[gid] = {"score": score, "source": "hybrid"}

        # Only augment if graph returned enough hits (min_graph_hits threshold)
        if len(graph_scores) >= self._min_graph_hits:
            for gid in graph_scores:
                if gid in combined:
                    # Small boost for grants confirmed by both — never reorders
                    combined[gid]["score"] *= (1 + self._overlap_bonus)
                else:
                    # Graph-only grants appended after hybrid with penalty score
                    min_h = min(hybrid_scores.values()) if hybrid_scores else 0.1
                    combined[gid] = {
                        "score": min_h * self._graph_only_penalty,
                        "source": "graph"
                    }
        else:
            if graph_scores:
                print(f"  \u2139\ufe0f  Graph returned {len(graph_scores)} hits "
                      f"(below min_graph_hits={self._min_graph_hits}) \u2014 skipping augmentation")

        # ── 4. Return top_k grant_ids by combined score ───────────────────
        ranked = sorted(combined.items(), key=lambda x: x[1]["score"], reverse=True)

        # Store source tags for display in assistant
        self._last_discovery_sources = {gid: meta["source"]
                                         for gid, meta in combined.items()}

        n_hybrid = sum(1 for m in combined.values() if m["source"] == "hybrid")
        n_graph  = sum(1 for m in combined.values() if m["source"] == "graph")
        print(f"  🧭 Discovery pool: {len(combined)} candidates "
              f"({n_hybrid} hybrid, {n_graph} graph-only)")

        return [gid for gid, _ in ranked[:top_k]]

    def rfp_query(self, conditions=None, interventions=None, populations=None,
                  free_text=None, fqhc_only=False,
                  section_filter: str = None,
                  verbose=True) -> List[Dict]:
        if not self._ready:
            print("\u274c Call setup() first."); return []
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"\U0001f3af RFP QUERY")
            if conditions:    print(f"   Conditions:    {conditions}")
            if interventions: print(f"   Interventions: {interventions}")
            if populations:   print(f"   Populations:   {populations}")
            if free_text:     print(f"   Free text:     {free_text}")
            if section_filter:print(f"   Section:       {section_filter}")
            print(f"   FQHC only: {fqhc_only}")
            print(f"{'=' * 70}")
        t, all_r = time.time(), []
        if self.graph_engine and any([conditions, interventions, populations]):
            gr = self.graph_engine.rfp_match(conditions, interventions, populations, fqhc_only)
            all_r.extend(gr)
            if verbose: print(f"\n\U0001f578\ufe0f  Graph match: {len(gr)} results")
        sq = free_text or " ".join((conditions or []) + (interventions or []) + (populations or []))
        if sq and self.model:
            section = WeaviateManager.SECTION_ALIASES.get(section_filter, section_filter)
            encode_text = f"{section.replace('_',' ')}: {sq}" if section else sq
            qv = self.model.encode(encode_text).tolist()
            hr = self.weaviate.hybrid_search(sq, qv, self.alpha, self.top_k,
                                             section_filter=section_filter,
                                             fqhc_only=fqhc_only)
            if verbose: print(f"\U0001f4ca Hybrid search: {len(hr)} results")
            all_r.extend(hr)
        seen, unique = set(), []
        for r in all_r:
            g = r.get("grant_id")
            if g and g not in seen:
                seen.add(g); unique.append(r)
        if fqhc_only: unique = [r for r in unique if r.get("is_fqhc")]
        unique = self._filter_synthetic(unique)
        if self.enricher: unique = self.enricher.enrich(unique)
        if verbose:
            print(f"\n\u23f1\ufe0f  {time.time()-t:.2f}s")
            self._print_results(unique)
        return unique

    def compare_approaches(self, query_text: str) -> Dict:
        if not self._ready:
            print("\u274c Not ready."); return {}
        print(f"\n{'=' * 70}")
        print(f"\U0001f4ca APPROACH COMPARISON: '{query_text}'")
        print(f"{'=' * 70}")
        qv, results = self.model.encode(query_text).tolist(), {}
        for alpha, label in [(0.0, "BM25 only (\u03b1=0.0)"),
                             (0.25, "Hybrid   (\u03b1=0.25)"),
                             (0.5,  "Hybrid   (\u03b1=0.50)"),
                             (1.0,  "Vector   (\u03b1=1.00)")]:
            r = self.weaviate.hybrid_search(query_text, qv, alpha=alpha, top_k=5)
            results[label] = r
            print(f"\n  {label}:")
            for i, x in enumerate(r[:3], 1):
                title = x.get("title", "")[:50] or x.get("text", "")[:50]
                print(f"    {i}. [{x['score']:.4f}] {x.get('grant_id')} — {title}...")
        if self.graph_engine:
            hr  = self.weaviate.hybrid_search(query_text, qv, alpha=0.25, top_k=5)
            exp = self.graph_engine.expand([r["grant_id"] for r in hr if r.get("grant_id")], top_k=5)
            results["Graph expansion"] = exp
            print(f"\n  Graph expansion (via hub nodes):")
            for i, x in enumerate(exp[:3], 1):
                print(f"    {i}. {x.get('grant_id')} — via {x.get('shared_hub')} (score: {x.get('score', 0):.3f})")
        return results

    def close(self):
        self.weaviate.close()
        self._ready = False

    def _load_graph(self):
        for path, loader in [(PATHS["graph_pkl"], lambda p: pickle.load(open(p, "rb"))),
                             (PATHS["graph_gml"],  lambda p: nx.read_gml(p))]:
            if os.path.exists(path):
                try:
                    g = loader(path)
                    print(f"  \u2705 Graph loaded from {os.path.basename(path)}: {g.number_of_nodes()} nodes")
                    return g
                except Exception as e:
                    print(f"  \u26a0\ufe0f  {os.path.basename(path)} failed: {e}")
        return None

    def _filter_synthetic(self, results: List[Dict]) -> List[Dict]:
        if not self.exclude_synthetic:
            return results
        filtered = [r for r in results
                    if not any(str(r.get("grant_id","")).startswith(p)
                               for p in SYNTHETIC_PREFIXES)]
        removed = len(results) - len(filtered)
        if removed > 0:
            print(f"  \U0001f52c Filtered {removed} synthetic grants (use exclude_synthetic=False to show)")
        return filtered

    def _print_results(self, results: List[Dict]):
        print(f"\n{chr(0x2500) * 70}")
        print(f"\U0001f4cb RESULTS ({len(results)} total)")
        print(f"{chr(0x2500) * 70}")
        for i, r in enumerate(results[:self.top_k], 1):
            src  = r.get("source", "")
            icon = "\U0001f578\ufe0f" if "graph" in src.lower() else "\U0001f50d"
            print(f"\n{i}. {icon}  [score: {r.get('score', 0):.4f}]  {src}")
            print(f"   Grant ID:    {r.get('grant_id', 'N/A')}")
            if r.get("title"):            print(f"   Title:       {r['title'][:80]}")
            if r.get("institution"):      print(f"   Institution: {r['institution'][:60]}")
            if r.get("year"):             print(f"   Year:        {r['year']}")
            if r.get("section_type"):     print(f"   Section:     {r['section_type']}")
            if r.get("section_hint"):     print(f"   Seeded from: {r['section_hint']} section")
            if r.get("is_fqhc"):          print(f"   \u2705 FQHC-relevant")
            if r.get("abstract_snippet"): print(f"   Abstract:    {r['abstract_snippet']}...")
            if r.get("source_url"):       print(f"   \U0001f517 Source:   {r['source_url']}")


def interactive_mode(pipeline: GrantQueryPipeline):
    print("\n" + "=" * 70)
    print("\U0001f4ac INTERACTIVE QUERY MODE")
    print("=" * 70)
    print("  <query>                    — hybrid + graph search")
    print("  /describe <description>    — parse RFP description → section-level chunks")
    print("  /aims <query>              — specific aims section only")
    print("  /methods <query>           — methods section only")
    print("  /significance <query>      — significance section only")
    print("  /approach <query>          — approach section only")
    print("  /background <query>        — background section only")
    print("  /section <n> <query>       — any section filter")
    print("  /rfp <query>               — structured RFP match")
    print("  /compare <query>           — compare all approaches")
    print("  /fqhc <query>              — FQHC-only results")
    print("  /alpha <0-1> <query>       — custom alpha")
    print("  /stats                     — pipeline stats")
    print("  /synth                     — toggle synthetic grants")
    print("  /quit                      — exit\n")
    print("  Section shorthands: aims, sig, innov, approach, methods, bg, summary\n")

    while True:
        try:
            ui = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not ui: continue
        if ui.lower() in ["/quit", "/exit", "quit", "exit"]: break

        elif ui.lower() == "/stats":
            synth = "excluded" if pipeline.exclude_synthetic else "included"
            print(f"\n\U0001f4ca Setup: {pipeline._setup_time}s | "
                  f"Alpha: {pipeline.alpha} | Top-K: {pipeline.top_k} | "
                  f"Graph: {'enabled' if pipeline.graph_engine else 'disabled'} | "
                  f"Synthetic: {synth}")

        elif ui.lower() == "/synth":
            pipeline.exclude_synthetic = not pipeline.exclude_synthetic
            state = "excluded" if pipeline.exclude_synthetic else "included"
            print(f"  Synthetic grants now {state}")

        elif ui.startswith("/describe "):
            description = ui[10:].strip()
            if description:
                try:
                    from application_assistant import ApplicationAssistant
                    assistant = ApplicationAssistant(pipeline)
                    assistant.describe_application(description, use_llm=False)
                except ImportError:
                    print("  ⚠️  application_assistant.py not found — running plain query instead")
                    pipeline.query(description)
            else:
                print("  Usage: /describe <natural language application description>")

        elif ui.startswith("/compare "):
            pipeline.compare_approaches(ui[9:].strip())
        elif ui.startswith("/fqhc "):
            pipeline.query(ui[6:].strip(), fqhc_only=True)
        elif ui.startswith("/rfp "):
            pipeline.rfp_query(free_text=ui[5:].strip())
        elif ui.startswith("/aims "):
            pipeline.query(ui[6:].strip(), section_filter="aims")
        elif ui.startswith("/methods "):
            pipeline.query(ui[9:].strip(), section_filter="methods")
        elif ui.startswith("/significance "):
            pipeline.query(ui[14:].strip(), section_filter="significance")
        elif ui.startswith("/approach "):
            pipeline.query(ui[10:].strip(), section_filter="approach")
        elif ui.startswith("/background "):
            pipeline.query(ui[12:].strip(), section_filter="background")
        elif ui.startswith("/summary "):
            pipeline.query(ui[9:].strip(), section_filter="summary")
        elif ui.startswith("/innov "):
            pipeline.query(ui[7:].strip(), section_filter="innov")
        elif ui.startswith("/section "):
            parts = ui.split(" ", 2)
            if len(parts) == 3:
                pipeline.query(parts[2], section_filter=parts[1])
            else:
                print("  Usage: /section <name> <query>")
        elif ui.startswith("/alpha "):
            parts = ui.split(" ", 2)
            if len(parts) == 3:
                try: pipeline.query(parts[2], alpha=float(parts[1]))
                except ValueError: print("  Usage: /alpha 0.5 your query")
        else:
            pipeline.query(ui)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",      type=str)
    parser.add_argument("--alpha",      type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--top-k",      type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--fqhc-only",  action="store_true")
    parser.add_argument("--no-graph",   action="store_true")
    parser.add_argument("--compare",    action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--include-synthetic", action="store_true")
    args = parser.parse_args()

    pipeline = GrantQueryPipeline(alpha=args.alpha, top_k=args.top_k,
                                  exclude_synthetic=not args.include_synthetic)
    if not pipeline.setup():
        sys.exit(1)
    if args.setup_only:
        pipeline.close(); return
    try:
        if args.query and args.compare:   pipeline.compare_approaches(args.query)
        elif args.query:                  pipeline.query(args.query, use_graph=not args.no_graph,
                                                         fqhc_only=args.fqhc_only)
        else:                             interactive_mode(pipeline)
    except KeyboardInterrupt:
        print("\n\n\u26a0\ufe0f  Interrupted")
    finally:
        try: pipeline.close()
        except: pass


if __name__ == "__main__":
    main()