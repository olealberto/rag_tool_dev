# ============================================================================
# 📁 phase5_knowledge_graph.py - KNOWLEDGE GRAPH WITH PERSISTENCE
# ============================================================================

"""
PHASE 5: KNOWLEDGE GRAPH-AUGMENTED RAG

Builds a knowledge graph from your NIH abstracts with:
    - grant nodes
    - institute nodes
    - year nodes
    - condition nodes  (diabetes, hypertension, HIV, etc.)
    - intervention nodes (CHW, telehealth, navigation, etc.)
    - population nodes  (Latino, pediatric, rural, etc.)
    - fqhc_hub node

Loads directly from CSV — no Weaviate dependency for graph building.
Graph persists between sessions via GML + pickle.

Run:
    !python phase5_knowledge_graph.py            # load cached or build
    !python phase5_knowledge_graph.py --rebuild  # force full rebuild
"""

print("="*70)
print("🎯 PHASE 5: KNOWLEDGE GRAPH (Persistent)")
print("="*70)

import sys
import os
import json
import time
import pickle
import hashlib
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

try:
    from community import community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("⚠️  python-louvain not installed — community detection skipped")
    print("   pip install python-louvain")

# ============ PATHS ============

ABSTRACTS_PATH  = "./phase2_output/nih_research_abstracts.csv"
CHUNKS_PATH     = "./phase3_results/document_chunks.csv"
GRAPH_STORE_DIR = "./phase5_graph_store"
GML_ROOT_PATH   = "./phase5_knowledge_graph.gml"  # used by run_evaluation.py


# ============ SEMANTIC KEYWORD MAPS ============

CONDITIONS = {
    "diabetes":            ["diabetes", "diabetic", "hba1c", "glycemic", "insulin",
                            "type 2 diabetes", "type ii diabetes"],
    "hypertension":        ["hypertension", "blood pressure", "cardiovascular",
                            "heart disease", "stroke", "systolic"],
    "depression":          ["depression", "depressive", "phq-9", "antidepressant",
                            "major depressive"],
    "anxiety":             ["anxiety", "gad-7", "panic disorder"],
    "HIV":                 ["hiv", "aids", "prep", "antiretroviral", "art"],
    "asthma":              ["asthma", "inhaler", "bronchial", "pulmonary"],
    "cancer":              ["cancer", "oncology", "tumor", "carcinoma",
                            "mammography", "colonoscopy"],
    "obesity":             ["obesity", "bmi", "weight loss", "overweight"],
    "substance_use":       ["substance use", "opioid", "addiction", "naloxone",
                            "buprenorphine", "alcohol use", "drug use"],
    "social_determinants": ["food insecurity", "housing instability", "sdoh",
                            "social determinants", "social needs"],
}

INTERVENTIONS = {
    "CHW":           ["community health worker", "chw", "promotora",
                      "lay health advisor", "community health educator"],
    "telehealth":    ["telehealth", "telemedicine", "virtual visit",
                      "remote monitoring", "mhealth", "mobile health"],
    "navigation":    ["patient navigation", "navigator", "care coordination",
                      "care manager"],
    "screening":     ["screening program", "preventive screening",
                      "early detection", "health screening"],
    "education":     ["health education", "health literacy",
                      "patient education", "self-management"],
    "behavioral":    ["cognitive behavioral", "cbt", "counseling",
                      "behavioral intervention", "motivational interviewing"],
    "medication":    ["medication adherence", "pharmacist",
                      "medication management"],
    "integrated_care": ["integrated care", "co-located", "collaborative care",
                        "behavioral health integration"],
}

POPULATIONS = {
    "Latino":           ["latino", "hispanic", "latinx", "spanish speaking",
                         "promotora"],
    "African_American": ["african american", "black", "african-american"],
    "pediatric":        ["pediatric", "children", "adolescent", "youth",
                         "child health", "school based"],
    "geriatric":        ["older adult", "geriatric", "elderly", "aging", "senior"],
    "rural":            ["rural", "appalachian", "frontier", "rural health"],
    "low_income":       ["low-income", "low income", "poverty", "economically",
                         "underserved"],
    "Medicaid":         ["medicaid", "uninsured", "underinsured", "safety-net"],
    "LGBTQ":            ["lgbtq", "transgender", "sexual minority", "msm"],
}

FQHC_KEYWORDS = [
    "federally qualified health center", "fqhc",
    "community health center", "safety-net clinic",
    "medically underserved", "health disparities",
]


# ============ PERSISTENCE MANAGER ============

class GraphPersistenceManager:

    def __init__(self, graph_dir: str = GRAPH_STORE_DIR):
        self.graph_dir = Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.pkl_path  = self.graph_dir / "graph.pkl"
        self.gml_path  = self.graph_dir / "graph.gml"
        self.meta_path = self.graph_dir / "graph_meta.json"

    def save(self, graph: nx.Graph, metadata: Dict = None):
        print(f"\n💾 Saving graph to {self.graph_dir}...")

        # Pickle — fast, preserves all types
        with open(self.pkl_path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✅ Pickle: {self.pkl_path.stat().st_size // 1024}KB")

        # GML — portable
        gml_graph = self._sanitize_for_gml(graph)
        nx.write_gml(gml_graph, str(self.gml_path))
        print(f"  ✅ GML:    {self.gml_path.stat().st_size // 1024}KB")

        # Also write to repo root for run_evaluation.py compatibility
        nx.write_gml(gml_graph, GML_ROOT_PATH)
        root_size = Path(GML_ROOT_PATH).stat().st_size // 1024
        print(f"  ✅ GML (root): {GML_ROOT_PATH} ({root_size}KB)")

        # Metadata
        meta = {
            "saved_at":   datetime.now().isoformat(),
            "nodes":      graph.number_of_nodes(),
            "edges":      graph.number_of_edges(),
            "node_types": self._count_types(graph),
        }
        if metadata:
            meta.update(metadata)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  ✅ Meta saved")
        print(f"  Graph: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges")

    def load(self) -> Optional[nx.Graph]:
        if self.pkl_path.exists():
            print(f"📦 Loading from pickle...")
            try:
                with open(self.pkl_path, "rb") as f:
                    g = pickle.load(f)
                print(f"  ✅ {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
                return g
            except Exception as e:
                print(f"  ⚠️  Pickle failed: {e}, trying GML...")

        if self.gml_path.exists():
            print(f"📦 Loading from GML...")
            try:
                g = nx.read_gml(str(self.gml_path))
                print(f"  ✅ {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
                return g
            except Exception as e:
                print(f"  ❌ GML failed: {e}")

        return None

    def exists(self) -> bool:
        return self.pkl_path.exists() or self.gml_path.exists()

    def get_fingerprint(self) -> Optional[str]:
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                return json.load(f).get("corpus_fingerprint")
        return None

    def _sanitize_for_gml(self, graph: nx.Graph) -> nx.Graph:
        g = graph.copy()
        for node, data in g.nodes(data=True):
            for k, v in list(data.items()):
                if isinstance(v, bool):
                    g.nodes[node][k] = int(v)
                elif isinstance(v, (list, dict, set)):
                    g.nodes[node][k] = json.dumps(v)
                elif v is None:
                    g.nodes[node][k] = ""
        for u, v, data in g.edges(data=True):
            for k, val in list(data.items()):
                if isinstance(val, bool):
                    g.edges[u, v][k] = int(val)
                elif isinstance(val, (list, dict, set)):
                    g.edges[u, v][k] = json.dumps(val)
                elif val is None:
                    g.edges[u, v][k] = ""
        return g

    def _count_types(self, graph: nx.Graph) -> Dict:
        counts = defaultdict(int)
        for _, d in graph.nodes(data=True):
            counts[d.get("type", "unknown")] += 1
        return dict(counts)


# ============ DATA LOADER ============

class DataLoader:

    def load_abstracts(self) -> pd.DataFrame:
        if not os.path.exists(ABSTRACTS_PATH):
            raise FileNotFoundError(
                f"Abstracts not found: {ABSTRACTS_PATH}\nRun phase2_api.py first"
            )
        df = pd.read_csv(ABSTRACTS_PATH)
        print(f"  ✅ Abstracts: {len(df)} rows")
        return df

    def load_chunks(self) -> pd.DataFrame:
        if not os.path.exists(CHUNKS_PATH):
            print(f"  ⚠️  Chunks not found: {CHUNKS_PATH}")
            return pd.DataFrame()
        df = pd.read_csv(CHUNKS_PATH)
        if "embedding" in df.columns:
            df = df.drop(columns=["embedding"])
        print(f"  ✅ Chunks:    {len(df)} rows")
        return df


# ============ KNOWLEDGE GRAPH BUILDER ============

class KnowledgeGraphBuilder:
    """
    Builds knowledge graph from NIH abstracts.
    No Weaviate required — reads directly from CSV.

    Node types: grant, institute, year, condition, intervention, population, fqhc_hub
    Edge types: funded_by, published_in, treats, uses, targets, is_fqhc, similar_study
    """

    def __init__(self):
        self.graph = nx.Graph()

    def build(self, abstracts_df: pd.DataFrame,
              chunks_df: pd.DataFrame = None) -> nx.Graph:

        print(f"\n🏗️  Building graph from {len(abstracts_df)} abstracts...")

        # Identify columns
        self.id_col   = self._find_col(abstracts_df,
                                       ["grant_id", "grantId", "project_num",
                                        "application_id", "id"])
        self.text_col = self._find_col(abstracts_df,
                                       ["abstract", "text", "Abstract"])
        self.inst_col = self._find_col(abstracts_df,
                                       ["institute", "org_name", "ic_name",
                                        "administering_ic"], required=False)
        self.year_col = self._find_col(abstracts_df,
                                       ["year", "fiscal_year",
                                        "award_fiscal_year"], required=False)

        print(f"  ID col:   {self.id_col}")
        print(f"  Text col: {self.text_col}")
        print(f"  Inst col: {self.inst_col}")
        print(f"  Year col: {self.year_col}")

        # Lowercase text for matching
        df = abstracts_df.copy()
        df["_text"] = df[self.text_col].fillna("").str.lower()

        # FQHC scoring
        df["_fqhc_score"] = df["_text"].apply(
            lambda t: sum(1.0 for kw in FQHC_KEYWORDS if kw in t)
        )
        df["_is_fqhc"] = df["_fqhc_score"] > 0
        print(f"  FQHC-relevant: {df['_is_fqhc'].sum()}")

        # Build all node types
        self._add_grant_nodes(df)
        self._add_institute_nodes(df)
        self._add_year_nodes(df)
        self._add_condition_nodes(df)
        self._add_intervention_nodes(df)
        self._add_population_nodes(df)
        self._add_fqhc_hub(df)
        self._add_similarity_edges(df)

        # Summary
        type_counts = defaultdict(int)
        for _, d in self.graph.nodes(data=True):
            type_counts[d.get("type", "unknown")] += 1

        print(f"\n  ✅ Graph complete:")
        print(f"     Total nodes: {self.graph.number_of_nodes()}")
        print(f"     Total edges: {self.graph.number_of_edges()}")
        for ntype, count in sorted(type_counts.items()):
            print(f"     {ntype}: {count}")

        return self.graph

    def _find_col(self, df: pd.DataFrame, candidates: List[str],
                  required: bool = True) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        if required:
            raise ValueError(
                f"Could not find any of {candidates} in columns: "
                f"{list(df.columns)}"
            )
        return None

    def _add_grant_nodes(self, df: pd.DataFrame):
        print("  Adding grant nodes...")
        for _, row in df.iterrows():
            gid = str(row[self.id_col])
            year = 2024
            if self.year_col and pd.notna(row.get(self.year_col)):
                try:
                    year = int(row[self.year_col])
                except (ValueError, TypeError):
                    pass
            inst = "Unknown"
            if self.inst_col and pd.notna(row.get(self.inst_col)):
                inst = str(row[self.inst_col])

            self.graph.add_node(
                gid,
                type="grant",
                year=year,
                institute=inst,
                is_fqhc_focused=bool(row.get("_is_fqhc", False)),
                fqhc_score=float(row.get("_fqhc_score", 0.0)),
                data_source="nih_api",
            )

    def _add_institute_nodes(self, df: pd.DataFrame):
        if not self.inst_col:
            return
        print("  Adding institute nodes...")
        for inst, group in df.groupby(self.inst_col):
            if not inst or str(inst) in ["Unknown", "nan", ""]:
                continue
            node_id = f"INST_{inst}"
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, type="institute", name=str(inst))
            for _, row in group.iterrows():
                gid = str(row[self.id_col])
                if not self.graph.has_edge(gid, node_id):
                    self.graph.add_edge(gid, node_id,
                                        type="funded_by", weight=1.0)

    def _add_year_nodes(self, df: pd.DataFrame):
        if not self.year_col:
            return
        print("  Adding year nodes...")
        years = sorted(df[self.year_col].dropna().unique().tolist())
        for year in years:
            node_id = f"YEAR_{int(year)}"
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, type="year", year=int(year))

        for i in range(len(years) - 1):
            y1 = f"YEAR_{int(years[i])}"
            y2 = f"YEAR_{int(years[i+1])}"
            if not self.graph.has_edge(y1, y2):
                self.graph.add_edge(y1, y2, type="consecutive_year", weight=0.3)

        for _, row in df.iterrows():
            val = row.get(self.year_col)
            if val and not pd.isna(val):
                node_id = f"YEAR_{int(val)}"
                gid = str(row[self.id_col])
                if self.graph.has_node(node_id) and not self.graph.has_edge(gid, node_id):
                    self.graph.add_edge(gid, node_id,
                                        type="published_in", weight=0.6)

    def _add_condition_nodes(self, df: pd.DataFrame):
        print("  Adding condition nodes...")
        edges = 0
        for condition, keywords in CONDITIONS.items():
            node_id = f"COND_{condition}"
            mask = df["_text"].apply(lambda t: any(kw in t for kw in keywords))
            matching = df[mask]
            if matching.empty:
                continue
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, type="condition", name=condition,
                                    grant_count=len(matching))
            for _, row in matching.iterrows():
                gid = str(row[self.id_col])
                if not self.graph.has_edge(gid, node_id):
                    self.graph.add_edge(gid, node_id, type="treats", weight=1.2)
                    edges += 1
        print(f"    {edges} condition edges added")

    def _add_intervention_nodes(self, df: pd.DataFrame):
        print("  Adding intervention nodes...")
        edges = 0
        for intervention, keywords in INTERVENTIONS.items():
            node_id = f"INT_{intervention}"
            mask = df["_text"].apply(lambda t: any(kw in t for kw in keywords))
            matching = df[mask]
            if matching.empty:
                continue
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, type="intervention",
                                    name=intervention,
                                    grant_count=len(matching))
            for _, row in matching.iterrows():
                gid = str(row[self.id_col])
                if not self.graph.has_edge(gid, node_id):
                    self.graph.add_edge(gid, node_id, type="uses", weight=1.1)
                    edges += 1
        print(f"    {edges} intervention edges added")

    def _add_population_nodes(self, df: pd.DataFrame):
        print("  Adding population nodes...")
        edges = 0
        for population, keywords in POPULATIONS.items():
            node_id = f"POP_{population}"
            mask = df["_text"].apply(lambda t: any(kw in t for kw in keywords))
            matching = df[mask]
            if matching.empty:
                continue
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, type="population",
                                    name=population,
                                    grant_count=len(matching))
            for _, row in matching.iterrows():
                gid = str(row[self.id_col])
                if not self.graph.has_edge(gid, node_id):
                    self.graph.add_edge(gid, node_id, type="targets", weight=1.1)
                    edges += 1
        print(f"    {edges} population edges added")

    def _add_fqhc_hub(self, df: pd.DataFrame):
        print("  Adding FQHC hub...")
        if not self.graph.has_node("FQHC_HUB"):
            self.graph.add_node("FQHC_HUB", type="fqhc_hub",
                                description="Central FQHC relevance hub")
        fqhc = df[df["_is_fqhc"]]
        for _, row in fqhc.iterrows():
            gid   = str(row[self.id_col])
            score = float(row.get("_fqhc_score", 0.5))
            if not self.graph.has_edge(gid, "FQHC_HUB"):
                self.graph.add_edge(gid, "FQHC_HUB",
                                    type="is_fqhc",
                                    weight=min(score / 5, 1.0))
        print(f"    {len(fqhc)} FQHC grants connected to hub")

    def _add_similarity_edges(self, df: pd.DataFrame):
        print("  Adding similarity edges...")
        edges = 0
        for condition, keywords in CONDITIONS.items():
            mask = df["_text"].apply(lambda t: any(kw in t for kw in keywords))
            group_ids = df[mask][self.id_col].astype(str).tolist()
            for i in range(len(group_ids)):
                for j in range(i + 1, min(i + 6, len(group_ids))):
                    g1, g2 = group_ids[i], group_ids[j]
                    if not self.graph.has_edge(g1, g2):
                        self.graph.add_edge(g1, g2, type="similar_study",
                                            shared_condition=condition,
                                            weight=0.7)
                        edges += 1
        print(f"    {edges} similarity edges added")


# ============ GRAPH ANALYZER ============

class GraphAnalyzer:

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def summary(self) -> Dict:
        node_types = defaultdict(int)
        for _, d in self.graph.nodes(data=True):
            node_types[d.get("type", "unknown")] += 1

        edge_types = defaultdict(int)
        for _, _, d in self.graph.edges(data=True):
            edge_types[d.get("type", "unknown")] += 1

        grant_nodes = [n for n, d in self.graph.nodes(data=True)
                       if d.get("type") == "grant"]

        result = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "density":     round(nx.density(self.graph), 6),
            "node_types":  dict(node_types),
            "edge_types":  dict(edge_types),
            "grant_count": len(grant_nodes),
        }

        if len(grant_nodes) > 2:
            sub   = self.graph.subgraph(grant_nodes)
            deg_c = nx.degree_centrality(sub)
            result["top_grants_by_centrality"] = [
                {"grant_id": n, "centrality": round(v, 4)}
                for n, v in sorted(deg_c.items(),
                                   key=lambda x: x[1], reverse=True)[:10]
            ]

        if LOUVAIN_AVAILABLE and len(grant_nodes) > 5:
            try:
                partition = community_louvain.best_partition(self.graph)
                result["communities"] = {
                    "count":      len(set(partition.values())),
                    "modularity": round(
                        community_louvain.modularity(partition, self.graph), 4
                    ),
                }
            except Exception:
                pass

        return result

    def find_grants_by_condition(self, condition: str) -> List[str]:
        node_id = f"COND_{condition}"
        if node_id not in self.graph:
            matches = [n for n in self.graph.nodes
                       if n.startswith("COND_") and condition.lower() in n.lower()]
            if not matches:
                return []
            node_id = matches[0]
        return [n for n in self.graph.neighbors(node_id)
                if self.graph.nodes[n].get("type") == "grant"]

    def find_grants_by_intervention(self, intervention: str) -> List[str]:
        node_id = f"INT_{intervention}"
        if node_id not in self.graph:
            matches = [n for n in self.graph.nodes
                       if n.startswith("INT_") and intervention.lower() in n.lower()]
            if not matches:
                return []
            node_id = matches[0]
        return [n for n in self.graph.neighbors(node_id)
                if self.graph.nodes[n].get("type") == "grant"]

    def find_grants_by_population(self, population: str) -> List[str]:
        node_id = f"POP_{population}"
        if node_id not in self.graph:
            matches = [n for n in self.graph.nodes
                       if n.startswith("POP_") and population.lower() in n.lower()]
            if not matches:
                return []
            node_id = matches[0]
        return [n for n in self.graph.neighbors(node_id)
                if self.graph.nodes[n].get("type") == "grant"]

    def find_related_grants(self, grant_id: str,
                            max_depth: int = 2,
                            top_k: int = 10) -> List[Dict]:
        if grant_id not in self.graph:
            return []

        related = []
        visited = {grant_id}
        queue   = [(grant_id, 0, 1.0)]

        while queue and len(related) < top_k * 3:
            current, depth, strength = queue.pop(0)
            if depth >= max_depth:
                continue
            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                edge  = self.graph.get_edge_data(current, neighbor, {})
                ndata = dict(self.graph.nodes[neighbor])
                w     = edge.get("weight", 0.5)

                if ndata.get("type") == "grant":
                    related.append({
                        "grant_id":  neighbor,
                        "depth":     depth + 1,
                        "strength":  round(strength * w, 3),
                        "edge_type": edge.get("type", ""),
                        "shared_condition": edge.get("shared_condition", ""),
                    })
                else:
                    queue.append((neighbor, depth + 1, strength * w))

        related.sort(key=lambda x: x["strength"], reverse=True)
        seen, unique = set(), []
        for r in related:
            if r["grant_id"] not in seen:
                seen.add(r["grant_id"])
                unique.append(r)
        return unique[:top_k]

    def rfp_graph_query(self,
                        conditions: List[str]    = None,
                        interventions: List[str] = None,
                        populations: List[str]   = None,
                        fqhc_only: bool          = True) -> List[Dict]:
        scores = defaultdict(float)

        if conditions:
            for c in conditions:
                for gid in self.find_grants_by_condition(c):
                    scores[gid] += 1.5

        if interventions:
            for i in interventions:
                for gid in self.find_grants_by_intervention(i):
                    scores[gid] += 1.0

        if populations:
            for p in populations:
                for gid in self.find_grants_by_population(p):
                    scores[gid] += 1.0

        if fqhc_only and "FQHC_HUB" in self.graph:
            fqhc_grants = {
                n for n in self.graph.neighbors("FQHC_HUB")
                if self.graph.nodes[n].get("type") == "grant"
            }
            scores = {k: v for k, v in scores.items() if k in fqhc_grants}

        results = []
        for gid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            ndata = dict(self.graph.nodes.get(gid, {}))
            results.append({
                "grant_id":        gid,
                "match_score":     round(score, 2),
                "institute":       ndata.get("institute", "Unknown"),
                "year":            ndata.get("year", "Unknown"),
                "is_fqhc_focused": bool(ndata.get("is_fqhc_focused", False)),
            })
        return results


# ============ VISUALIZATION ============

def visualize_graph(graph: nx.Graph, analysis: Dict,
                    save_path: str = "phase5_knowledge_graph.png"):
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle("Phase 5: Knowledge Graph Analysis",
                 fontsize=16, fontweight="bold")

    node_types = analysis.get("node_types", {})
    edge_types = analysis.get("edge_types", {})
    TYPE_COLORS = {
        "grant": "#4C72B0", "institute": "#DD8452", "year": "#55A868",
        "condition": "#C44E52", "intervention": "#8172B2",
        "population": "#937860", "fqhc_hub": "#DA8BC3",
    }

    # 1. Node types
    ax = axes[0, 0]
    if node_types:
        colors = [TYPE_COLORS.get(t, "#999999") for t in node_types.keys()]
        ax.bar(node_types.keys(), node_types.values(), color=colors)
        ax.set_title("Nodes by Type")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    # 2. Edge types
    ax = axes[0, 1]
    top_edges = dict(sorted(edge_types.items(), key=lambda x: x[1], reverse=True)[:8])
    if top_edges:
        ax.barh(list(top_edges.keys()), list(top_edges.values()), color="#4C72B0")
        ax.set_title("Edge Types")
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.3)

    # 3. Top grants by centrality
    ax = axes[0, 2]
    centrality = analysis.get("top_grants_by_centrality", [])
    if centrality:
        labels = [d["grant_id"][:15] for d in centrality[:8]]
        values = [d["centrality"] for d in centrality[:8]]
        ax.barh(labels[::-1], values[::-1], color="#55A868")
        ax.set_title("Top Grants by Centrality")
        ax.set_xlabel("Degree Centrality")
        ax.grid(axis="x", alpha=0.3)

    # 4. Condition coverage
    ax = axes[1, 0]
    cond_nodes = [
        (n.replace("COND_", ""), len([
            nb for nb in graph.neighbors(n)
            if graph.nodes[nb].get("type") == "grant"
        ]))
        for n, d in graph.nodes(data=True) if d.get("type") == "condition"
    ]
    cond_nodes.sort(key=lambda x: x[1], reverse=True)
    if cond_nodes:
        names, counts = zip(*cond_nodes)
        ax.bar(names, counts, color="#C44E52", alpha=0.85)
        ax.set_title("Grants per Condition")
        ax.set_ylabel("Grant Count")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3)

    # 5. Intervention coverage
    ax = axes[1, 1]
    int_nodes = [
        (n.replace("INT_", ""), len([
            nb for nb in graph.neighbors(n)
            if graph.nodes[nb].get("type") == "grant"
        ]))
        for n, d in graph.nodes(data=True) if d.get("type") == "intervention"
    ]
    int_nodes.sort(key=lambda x: x[1], reverse=True)
    if int_nodes:
        names, counts = zip(*int_nodes)
        ax.bar(names, counts, color="#8172B2", alpha=0.85)
        ax.set_title("Grants per Intervention")
        ax.set_ylabel("Grant Count")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3)

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")
    communities = analysis.get("communities", {})
    rows = [
        ["Total Nodes",       str(analysis.get("total_nodes", 0))],
        ["Total Edges",       str(analysis.get("total_edges", 0))],
        ["Grant Nodes",       str(node_types.get("grant", 0))],
        ["Condition Nodes",   str(node_types.get("condition", 0))],
        ["Intervention Nodes",str(node_types.get("intervention", 0))],
        ["Population Nodes",  str(node_types.get("population", 0))],
        ["Graph Density",     f"{analysis.get('density', 0):.5f}"],
        ["Communities",       str(communities.get("count", "N/A"))],
        ["Modularity",        str(communities.get("modularity", "N/A"))],
    ]
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    ax.set_title("Graph Summary", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"💾 Visualization saved to {save_path}")
    plt.close()


# ============ MAIN ============

def main(force_rebuild: bool = False):
    print("\n" + "="*70)
    print("🚀 PHASE 5: KNOWLEDGE GRAPH")
    print("="*70)

    persistence = GraphPersistenceManager()

    # Load data
    print("\n📦 STEP 1: LOADING DATA")
    print("-"*50)
    loader       = DataLoader()
    abstracts_df = loader.load_abstracts()
    chunks_df    = loader.load_chunks()

    # Corpus fingerprint
    id_col = next(
        (c for c in ["grant_id", "grantId", "project_num", "id"]
         if c in abstracts_df.columns),
        abstracts_df.columns[0]
    )
    fingerprint = hashlib.md5(
        "|".join(sorted(abstracts_df[id_col].astype(str).tolist())).encode()
    ).hexdigest()[:16]

    stored_fp      = persistence.get_fingerprint()
    corpus_changed = fingerprint != stored_fp

    # Load or build
    print("\n🏗️  STEP 2: LOADING OR BUILDING GRAPH")
    print("-"*50)

    graph = None

    if not force_rebuild and persistence.exists() and not corpus_changed:
        print("✅ Graph up to date — loading from disk")
        graph = persistence.load()

        # Verify condition nodes exist — if not, old graph, force rebuild
        if graph is not None:
            has_conditions = any(
                d.get("type") == "condition"
                for _, d in graph.nodes(data=True)
            )
            if not has_conditions:
                print("⚠️  Graph missing condition nodes — rebuilding")
                graph = None

    if graph is None:
        reason = "forced rebuild" if force_rebuild else (
            "corpus changed" if corpus_changed else "no cached graph"
        )
        print(f"🏗️  Building from scratch ({reason})...")
        builder = KnowledgeGraphBuilder()
        graph   = builder.build(abstracts_df, chunks_df)
        persistence.save(graph, metadata={"corpus_fingerprint": fingerprint})

    # Analyze
    print("\n📈 STEP 3: ANALYZING GRAPH")
    print("-"*50)
    analyzer = GraphAnalyzer(graph)
    analysis = analyzer.summary()

    print(f"  Nodes:      {analysis['total_nodes']}")
    print(f"  Edges:      {analysis['total_edges']}")
    print(f"  Density:    {analysis['density']}")
    print(f"  Node types: {analysis['node_types']}")
    if "communities" in analysis:
        print(f"  Communities: {analysis['communities']['count']}")
        print(f"  Modularity:  {analysis['communities']['modularity']}")

    # Demo queries
    print("\n🔍 STEP 4: DEMO QUERIES")
    print("-"*50)
    diabetes_grants = analyzer.find_grants_by_condition("diabetes")
    chw_grants      = analyzer.find_grants_by_intervention("CHW")
    latino_grants   = analyzer.find_grants_by_population("Latino")

    print(f"  Diabetes grants:    {len(diabetes_grants)}")
    print(f"  CHW grants:         {len(chw_grants)}")
    print(f"  Latino grants:      {len(latino_grants)}")

    print("\n  🎯 RFP: diabetes + CHW + Latino + FQHC")
    rfp_results = analyzer.rfp_graph_query(
        conditions=["diabetes"], interventions=["CHW"],
        populations=["Latino"], fqhc_only=True
    )
    if rfp_results:
        for r in rfp_results[:5]:
            print(f"    • {r['grant_id']} — score: {r['match_score']} "
                  f"| {r['institute']}")
    else:
        print("    No FQHC matches — retrying without fqhc_only filter")
        rfp_results = analyzer.rfp_graph_query(
            conditions=["diabetes"], interventions=["CHW"],
            populations=["Latino"], fqhc_only=False
        )
        for r in rfp_results[:3]:
            print(f"    • {r['grant_id']} — score: {r['match_score']}")

    # Related grants example
    grant_nodes = [n for n, d in graph.nodes(data=True)
                   if d.get("type") == "grant"]
    if grant_nodes:
        sample  = grant_nodes[0]
        related = analyzer.find_related_grants(sample, max_depth=2, top_k=3)
        print(f"\n  🔗 Related to {sample}:")
        for r in related:
            print(f"    • {r['grant_id']} (strength={r['strength']}, "
                  f"via {r['edge_type']})")

    # Visualize
    print("\n📊 STEP 5: VISUALIZATION")
    print("-"*50)
    visualize_graph(graph, analysis)

    # Save results
    print("\n💾 STEP 6: SAVING RESULTS")
    print("-"*50)
    results = {
        "phase":              "phase5",
        "timestamp":          datetime.now().isoformat(),
        "corpus_fingerprint": fingerprint,
        "graph_stats": {
            "nodes":      analysis["total_nodes"],
            "edges":      analysis["total_edges"],
            "density":    analysis["density"],
            "node_types": analysis["node_types"],
            "edge_types": analysis["edge_types"],
        },
        "communities": analysis.get("communities", {}),
        "demo_queries": {
            "diabetes_grants": len(diabetes_grants),
            "chw_grants":      len(chw_grants),
            "latino_grants":   len(latino_grants),
            "rfp_results":     len(rfp_results),
        },
    }
    with open("phase5_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("✅ PHASE 5 COMPLETE")
    print("="*70)
    print(f"\n📁 Saved files:")
    print(f"  • phase5_results.json")
    print(f"  • phase5_knowledge_graph.png")
    print(f"  • phase5_knowledge_graph.gml   ← used by run_evaluation.py")
    print(f"  • {GRAPH_STORE_DIR}/graph.pkl  ← fast reload next run")
    print(f"\n🔄 Next run loads from disk — no rebuild unless corpus changes")
    print(f"   Force rebuild: python phase5_knowledge_graph.py --rebuild")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Force full graph rebuild")
    args = parser.parse_args()
    main(force_rebuild=args.rebuild)