# ============================================================================
# 📁 phase5_knowledge_graph.py - KNOWLEDGE GRAPH WITH PERSISTENCE
# ============================================================================

"""
PHASE 5: KNOWLEDGE GRAPH-AUGMENTED RAG

Builds a knowledge graph from your NIH abstracts + full-text PDF grants with:
    - grant nodes      (NIH RePORTER abstracts + PDF/OCR ingested grants)
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
CHUNKS_EMB_PATH = "./phase3_results/document_chunks_with_embeddings.csv"
GRAPH_STORE_DIR = "./phase5_graph_store"
GML_ROOT_PATH   = "./phase5_knowledge_graph.gml"


# ============ SEMANTIC KEYWORD MAPS ============

CONDITIONS = {
    "diabetes":            ["diabetes", "diabetic", "hba1c", "glycemic", "insulin",
                            "type 2 diabetes", "type ii diabetes", "prediabetes",
                            "diabetes prevention", "glucose"],
    "hypertension":        ["hypertension", "blood pressure", "cardiovascular",
                            "heart disease", "stroke", "systolic", "antihypertensive",
                            "hypertensive", "cardiac"],
    "depression":          ["depression", "depressive", "phq-9", "antidepressant",
                            "major depressive", "mood disorder"],
    "anxiety":             ["anxiety", "gad-7", "panic disorder", "anxious"],
    "HIV":                 ["hiv", "aids", "prep", "antiretroviral", "art",
                            "hiv prevention", "viral load", "cd4"],
    "asthma":              ["asthma", "inhaler", "bronchial", "pulmonary", "copd",
                            "respiratory", "lung disease"],
    "cancer":              ["cancer", "oncology", "tumor", "carcinoma",
                            "mammography", "colonoscopy", "cervical", "colorectal"],
    "obesity":             ["obesity", "bmi", "weight loss", "overweight",
                            "weight management", "bariatric"],
    "substance_use":       ["substance use", "opioid", "addiction", "naloxone",
                            "buprenorphine", "alcohol use", "drug use",
                            "substance abuse", "suds", "overdose", "methadone",
                            "harm reduction", "recovery", "withdrawal",
                            "stimulant", "fentanyl", "heroin", "cocaine"],
    "social_determinants": ["food insecurity", "housing instability", "sdoh",
                            "social determinants", "social needs",
                            "food access", "transportation barrier",
                            "social risk", "unmet social", "housing insecurity",
                            "economic hardship", "utility", "interpersonal violence"],
    "behavioral_health":   ["behavioral health", "mental health", "psychiatric",
                            "psychosocial", "behavioral disorder", "mental illness",
                            "mental disorder", "co-occurring", "dual diagnosis",
                            "trauma", "ptsd", "crisis intervention"],
    "chronic_disease":     ["chronic disease", "chronic condition", "comorbidity",
                            "multiple chronic", "disease management", "chronic care"],
    "maternal_health":     ["maternal", "prenatal", "pregnancy", "postpartum",
                            "birth outcomes", "infant mortality", "obstetric"],
    "infectious_disease":  ["infectious disease", "sexually transmitted", "sti",
                            "tuberculosis", "hepatitis", "vaccine", "immunization"],
}

INTERVENTIONS = {
    "CHW":              ["community health worker", "chw", "promotora",
                         "lay health advisor", "community health educator",
                         "outreach worker", "peer navigator", "health promoter"],
    "telehealth":       ["telehealth", "telemedicine", "virtual visit",
                         "remote monitoring", "mhealth", "mobile health",
                         "digital health", "text message", "sms", "mobile app",
                         "video visit", "remote care", "electronic health"],
    "navigation":       ["patient navigation", "navigator", "care coordination",
                         "care manager", "case management", "care coordinator"],
    "screening":        ["screening program", "preventive screening",
                         "early detection", "health screening", "cancer screening",
                         "diabetes screening", "blood pressure screening"],
    "education":        ["health education", "health literacy",
                         "patient education", "self-management",
                         "community education", "wellness education"],
    "behavioral":       ["cognitive behavioral", "cbt", "counseling",
                         "behavioral intervention", "motivational interviewing",
                         "behavior change", "psychotherapy"],
    "medication":       ["medication adherence", "pharmacist",
                         "medication management", "drug therapy",
                         "prescription", "pharmacotherapy"],
    "integrated_care":  ["integrated care", "co-located", "collaborative care",
                         "behavioral health integration", "primary care integration",
                         "co-occurring treatment", "whole person care",
                         "team-based care", "interdisciplinary"],
    "peer_support":     ["peer support", "peer specialist", "peer counselor",
                         "lived experience", "recovery coach", "peer recovery"],
    "harm_reduction":   ["harm reduction", "syringe services", "needle exchange",
                         "overdose prevention", "naloxone distribution",
                         "safer use"],
    "care_management":  ["care management", "disease management", "chronic care",
                         "population health management", "registry"],
}

POPULATIONS = {
    "Latino":           ["latino", "hispanic", "latinx", "spanish speaking",
                         "promotora", "spanish-speaking", "mexican", "puerto rican",
                         "central american", "spanish language"],
    "African_American": ["african american", "black", "african-american",
                         "black community", "african american community"],
    "pediatric":        ["pediatric", "children", "adolescent", "youth",
                         "child health", "school based", "school-based",
                         "teenager", "juvenile", "childhood"],
    "geriatric":        ["older adult", "geriatric", "elderly", "aging", "senior",
                         "older patient", "aged", "gerontology"],
    "rural":            ["rural", "appalachian", "frontier", "rural health",
                         "rural community", "remote area", "underserved rural"],
    "low_income":       ["low-income", "low income", "poverty", "economically",
                         "underserved", "disadvantaged", "below poverty",
                         "200% fpl", "safety net", "uninsured"],
    "Medicaid":         ["medicaid", "uninsured", "underinsured", "safety-net",
                         "chip", "dual eligible", "medicaid beneficiary"],
    "LGBTQ":            ["lgbtq", "transgender", "sexual minority", "msm",
                         "gender nonconforming", "queer", "bisexual", "gay",
                         "lesbian", "gender identity"],
    "immigrant":        ["immigrant", "refugee", "undocumented", "limited english",
                         "lep", "foreign born", "newcomer", "asylum"],
    "homeless":         ["homeless", "unhoused", "housing instability",
                         "shelter", "transitional housing"],
}

FQHC_KEYWORDS = [
    "federally qualified health center", "fqhc",
    "community health center", "safety-net clinic",
    "medically underserved", "health disparities",
]

GRAPH_CONFIG_PATH = "./graph_config.json"


# ============ GRAPH CONFIG LOADER ============

def load_graph_config() -> dict:
    """
    Load graph_config.json if present, merging user overrides onto system defaults.
    If not found, writes a default config file and returns defaults.
    Any key missing from the user file falls back to the system default silently.
    """
    defaults = {
        "augmentation": {
            "overlap_bonus":      0.10,
            "graph_only_penalty": 0.5,
            "min_graph_hits":     3,
        },
        "edge_weights": {
            "treats":       1.2,
            "uses":         1.1,
            "targets":      1.1,
            "published_in": 0.6,
            "similar_to":   0.8,
            "fqhc_hub":     1.0,
        },
        "conditions":    CONDITIONS,
        "interventions": INTERVENTIONS,
        "populations":   POPULATIONS,
    }

    if not os.path.exists(GRAPH_CONFIG_PATH):
        export = {"_comment": "Knowledge graph augmentation config. Edit to customize for your corpus.",
                  "_docs": "Graph augmentation is always active. These settings control how the "
                           "graph is built and how it augments hybrid search results."}
        export.update(defaults)
        with open(GRAPH_CONFIG_PATH, "w") as f:
            json.dump(export, f, indent=2)
        print(f"  \u2139\ufe0f  graph_config.json not found \u2014 wrote defaults to {GRAPH_CONFIG_PATH}")
        print(f"      Edit that file to customize vocabulary and augmentation params.")
        return defaults

    with open(GRAPH_CONFIG_PATH) as f:
        user = json.load(f)

    config = {}
    for key, default_val in defaults.items():
        if key not in user:
            config[key] = default_val
        elif isinstance(default_val, dict) and isinstance(user[key], dict):
            merged = dict(default_val)
            merged.update({k: v for k, v in user[key].items()
                           if not k.startswith("_")})
            config[key] = merged
        else:
            config[key] = user[key]

    aug = config["augmentation"]
    print(f"  \u2705 graph_config.json loaded: "
          f"{len(config['conditions'])} conditions, "
          f"{len(config['interventions'])} interventions, "
          f"{len(config['populations'])} populations")
    print(f"     augmentation: overlap_bonus={aug['overlap_bonus']}, "
          f"min_hits={aug['min_graph_hits']}, penalty={aug['graph_only_penalty']}")
    return config


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

        with open(self.pkl_path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✅ Pickle: {self.pkl_path.stat().st_size // 1024}KB")

        gml_graph = self._sanitize_for_gml(graph)
        nx.write_gml(gml_graph, str(self.gml_path))
        print(f"  ✅ GML:    {self.gml_path.stat().st_size // 1024}KB")

        nx.write_gml(gml_graph, GML_ROOT_PATH)
        root_size = Path(GML_ROOT_PATH).stat().st_size // 1024
        print(f"  ✅ GML (root): {GML_ROOT_PATH} ({root_size}KB)")

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
        """
        Load document chunks.
        Prefer document_chunks_with_embeddings.csv (has data_source, section_type).
        Fall back to document_chunks.csv only if embeddings file is absent.
        """
        for path in [CHUNKS_EMB_PATH, CHUNKS_PATH]:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if "embedding" in df.columns:
                    df = df.drop(columns=["embedding"])
                has_src = "data_source" in df.columns
                pdf_n   = df[df["data_source"].str.lower()
                             .isin(["pdf","ocr","pdf_ingestion","synthetic_fqhc"])]["grant_id"].nunique()                           if has_src else 0
                print(f"  ✅ Chunks: {len(df)} rows  ({path})")
                if has_src:
                    print(f"     {pdf_n} PDF/OCR grants detected")
                else:
                    print(f"  ⚠️  data_source column missing — PDF grants skipped")
                return df
        print(f"  ⚠️  Chunks not found — PDF grants will be skipped")
        return pd.DataFrame()

    def extract_pdf_grants(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate chunk text per PDF grant into a pseudo-abstract DataFrame
        that matches the format expected by KnowledgeGraphBuilder.

        Returns a DataFrame with columns:
            grant_id, abstract, institution, year, is_fqhc_focused, data_source
        """
        if chunks_df.empty:
            return pd.DataFrame()

        # PDF and OCR sourced chunks
        pdf_mask = chunks_df["data_source"].str.lower().isin(["pdf", "ocr", "pdf_ingestion", "synthetic_fqhc"]) \
                   if "data_source" in chunks_df.columns \
                   else pd.Series(False, index=chunks_df.index)

        pdf_chunks = chunks_df[pdf_mask].copy()
        if pdf_chunks.empty:
            print(f"  ⚠️  No PDF/OCR chunks found in chunks CSV")
            return pd.DataFrame()

        print(f"  ✅ PDF/OCR chunks: {len(pdf_chunks)} rows "
              f"({pdf_chunks['grant_id'].nunique()} unique grants)")

        # Concatenate all chunk text per grant to form a pseudo-abstract
        # Prioritise high-value sections for keyword matching
        HIGH_VALUE = {"specific_aims", "significance", "innovation",
                      "approach", "methods", "background", "project_summary"}

        section_col = "section_type" if "section_type" in pdf_chunks.columns \
                      else "chunk_type" if "chunk_type" in pdf_chunks.columns \
                      else None

        records = []
        for grant_id, group in pdf_chunks.groupby("grant_id"):
            # Sort: high-value sections first
            if section_col:
                group = group.copy()
                group["_rank"] = group[section_col].apply(
                    lambda s: 0 if str(s) in HIGH_VALUE else 1
                )
                group = group.sort_values("_rank")

            # Concatenate up to 8000 chars of text
            text = " ".join(group["text"].fillna("").astype(str).tolist())[:8000]

            # Pull metadata from first row
            row0 = group.iloc[0]
            institution = str(row0.get("institution", "")) \
                          if "institution" in group.columns else ""
            year = int(row0.get("year", 2024)) \
                   if "year" in group.columns and pd.notna(row0.get("year")) \
                   else 2024
            is_fqhc = bool(row0.get("is_fqhc_focused",
                           row0.get("has_fqhc_terms", False))) \
                      if any(c in group.columns
                             for c in ["is_fqhc_focused", "has_fqhc_terms"]) \
                      else False

            records.append({
                "grant_id":        grant_id,
                "abstract":        text,
                "institution":     institution,
                "year":            year,
                "is_fqhc_focused": is_fqhc,
                "data_source":     str(row0.get("data_source", "pdf_ingestion")),
            })

        df = pd.DataFrame(records)
        print(f"  ✅ PDF pseudo-abstracts built: {len(df)} grants")
        return df


# ============ KNOWLEDGE GRAPH BUILDER ============

class KnowledgeGraphBuilder:
    """
    Builds knowledge graph from NIH abstracts + PDF grants.

    Node types: grant, institute, year, condition, intervention, population, fqhc_hub
    Edge types: funded_by, published_in, treats, uses, targets, is_fqhc, similar_study
    """

    def __init__(self, config: dict = None):
        self.graph  = nx.Graph()
        cfg         = config or {}
        self.conditions    = cfg.get("conditions",    CONDITIONS)
        self.interventions = cfg.get("interventions", INTERVENTIONS)
        self.populations   = cfg.get("populations",   POPULATIONS)
        ew = cfg.get("edge_weights", {})
        self.ew_treats      = ew.get("treats",       1.2)
        self.ew_uses        = ew.get("uses",          1.1)
        self.ew_targets     = ew.get("targets",       1.1)
        self.ew_published   = ew.get("published_in",  0.6)
        self.ew_similar     = ew.get("similar_to",    0.8)
        self.ew_fqhc        = ew.get("fqhc_hub",      1.0)

    def build(self, abstracts_df: pd.DataFrame,
              chunks_df: pd.DataFrame = None) -> nx.Graph:
        """
        Build knowledge graph.
        abstracts_df — PRIMARY corpus (PDF pseudo-abstracts from extract_pdf_grants)
        chunks_df    — SUPPLEMENTAL NIH abstracts (optional). Used only to enrich
                       grants not already present from PDFs.
        """
        print(f"\n\U0001f3d7\ufe0f  Building graph from {len(abstracts_df)} PDF grants (primary corpus)...")


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

        # Build from NIH abstracts
        df = abstracts_df.copy()
        df["_text"]      = df[self.text_col].fillna("").str.lower()
        df["_fqhc_score"] = df["_text"].apply(
            lambda t: sum(1.0 for kw in FQHC_KEYWORDS if kw in t)
        )
        df["_is_fqhc"] = df["_fqhc_score"] > 0
        print(f"  FQHC-relevant (abstracts): {df['_is_fqhc'].sum()}")

        self._add_grant_nodes(df)
        self._add_institute_nodes(df)
        self._add_year_nodes(df)
        self._add_condition_nodes(df)
        self._add_intervention_nodes(df)
        self._add_population_nodes(df)
        self._add_fqhc_hub(df)
        self._add_similarity_edges(df)

        # ── Supplemental NIH enrichment disabled ──────────────────────────
        # NIH abstracts (chunks_df) use a different schema than pdf_grants_df
        # and don't add value since all 92 grants are already in the graph
        # from full PDF text. Re-enable here if broader NIH corpus needed.
        # ───────────────────────────────────────────────────────────────────

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

    # ── NEW: PDF grant ingestion ────────────────────────────────────────────

    def _add_pdf_grants(self, pdf_df: pd.DataFrame):
        """
        Add PDF-sourced grants to the graph.
        Skips grants already present (ingested from NIH abstracts).
        Runs the same condition / intervention / population / FQHC matching
        against the concatenated chunk text.
        """
        print(f"\n  📄 Adding PDF grants to graph...")

        pdf_df = pdf_df.copy()
        pdf_df["_text"] = pdf_df["abstract"].fillna("").str.lower()
        pdf_df["_fqhc_score"] = pdf_df["_text"].apply(
            lambda t: sum(1.0 for kw in FQHC_KEYWORDS if kw in t)
        )
        pdf_df["_is_fqhc"] = (pdf_df["_fqhc_score"] > 0) | \
                              pdf_df["is_fqhc_focused"].astype(bool)

        new_grants  = added = skipped = 0
        cond_edges  = int_edges = pop_edges = fqhc_edges = 0

        for _, row in pdf_df.iterrows():
            gid  = str(row["grant_id"])
            text = row["_text"]

            # Skip if already in graph from abstracts
            if self.graph.has_node(gid):
                skipped += 1
                continue

            # Add grant node
            self.graph.add_node(
                gid,
                type="grant",
                year=int(row.get("year", 2024)),
                institute=str(row.get("institution", "")),
                is_fqhc_focused=bool(row["_is_fqhc"]),
                fqhc_score=float(row["_fqhc_score"]),
                data_source=str(row.get("data_source", "pdf_ingestion")),
            )
            new_grants += 1

            # Connect to condition hubs
            for condition, keywords in CONDITIONS.items():
                if any(kw in text for kw in keywords):
                    node_id = f"COND_{condition}"
                    if self.graph.has_node(node_id) and \
                       not self.graph.has_edge(gid, node_id):
                        self.graph.add_edge(gid, node_id,
                                            type="treats", weight=1.2)
                        cond_edges += 1

            # Connect to intervention hubs
            for intervention, keywords in INTERVENTIONS.items():
                if any(kw in text for kw in keywords):
                    node_id = f"INT_{intervention}"
                    if self.graph.has_node(node_id) and \
                       not self.graph.has_edge(gid, node_id):
                        self.graph.add_edge(gid, node_id,
                                            type="uses", weight=1.1)
                        int_edges += 1

            # Connect to population hubs
            for population, keywords in POPULATIONS.items():
                if any(kw in text for kw in keywords):
                    node_id = f"POP_{population}"
                    if self.graph.has_node(node_id) and \
                       not self.graph.has_edge(gid, node_id):
                        self.graph.add_edge(gid, node_id,
                                            type="targets", weight=1.1)
                        pop_edges += 1

            # Connect to FQHC hub
            if row["_is_fqhc"] and self.graph.has_node("FQHC_HUB"):
                score = float(row["_fqhc_score"])
                if not self.graph.has_edge(gid, "FQHC_HUB"):
                    self.graph.add_edge(gid, "FQHC_HUB",
                                        type="is_fqhc",
                                        weight=min(score / 5, 1.0))
                    fqhc_edges += 1

        # Similarity edges among new PDF grants (same condition co-occurrence)
        pdf_ids = [str(r["grant_id"]) for _, r in pdf_df.iterrows()
                   if self.graph.has_node(str(r["grant_id"])) and
                      self.graph.nodes[str(r["grant_id"])].get("data_source") == "pdf"]

        sim_edges = 0
        for condition, keywords in CONDITIONS.items():
            matching = [
                gid for gid in pdf_ids
                if any(kw in self.graph.nodes[gid].get("_text_cache", "")
                       for kw in keywords)
            ]
            # Simple: use text from pdf_df directly
            cond_mask = pdf_df["_text"].apply(
                lambda t: any(kw in t for kw in keywords)
            )
            cond_pdf_ids = [str(r["grant_id"]) for _, r in pdf_df[cond_mask].iterrows()
                            if self.graph.has_node(str(r["grant_id"]))]
            for i in range(len(cond_pdf_ids)):
                for j in range(i + 1, min(i + 6, len(cond_pdf_ids))):
                    g1, g2 = cond_pdf_ids[i], cond_pdf_ids[j]
                    if not self.graph.has_edge(g1, g2):
                        self.graph.add_edge(g1, g2, type="similar_study",
                                            shared_condition=condition,
                                            weight=0.7)
                        sim_edges += 1

        print(f"    New PDF grants added: {new_grants} "
              f"({skipped} already in graph from abstracts)")
        print(f"    Condition edges:   {cond_edges}")
        print(f"    Intervention edges:{int_edges}")
        print(f"    Population edges:  {pop_edges}")
        print(f"    FQHC hub edges:    {fqhc_edges}")
        print(f"    Similarity edges:  {sim_edges}")

    # ───────────────────────────────────────────────────────────────────────

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
        # Detect whether this DataFrame came from PDFs or NIH API
        is_pdf_source = "data_source" in df.columns and                         df["data_source"].str.lower().isin(
                            ["pdf", "ocr", "pdf_ingestion", "synthetic_fqhc"]
                        ).any()
        source_label = "pdf_ingestion" if is_pdf_source else "nih_api"
        print(f"  Adding grant nodes ({source_label}, {len(df)} grants)...")
        for _, row in df.iterrows():
            gid  = str(row[self.id_col])
            year = 2024
            if self.year_col and pd.notna(row.get(self.year_col)):
                try:    year = int(row[self.year_col])
                except: pass
            inst = "Unknown"
            if self.inst_col and pd.notna(row.get(self.inst_col)):
                inst = str(row[self.inst_col])

            # Use row-level data_source if available, else fall back to inferred label
            row_source = str(row.get("data_source", source_label)).lower()
            if row_source not in ("pdf", "ocr", "pdf_ingestion", "synthetic_fqhc"):
                row_source = source_label

            self.graph.add_node(
                gid,
                type="grant",
                year=year,
                institute=inst,
                is_fqhc_focused=bool(row.get("_is_fqhc", False)),
                fqhc_score=float(row.get("_fqhc_score", 0.0)),
                data_source=row_source,
            )

    def _add_institute_nodes(self, df: pd.DataFrame):
        if not self.inst_col: return
        print("  Adding institute nodes...")
        for inst, group in df.groupby(self.inst_col):
            if not inst or str(inst) in ["Unknown", "nan", ""]: continue
            node_id = f"INST_{inst}"
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, type="institute", name=str(inst))
            for _, row in group.iterrows():
                gid = str(row[self.id_col])
                if not self.graph.has_edge(gid, node_id):
                    self.graph.add_edge(gid, node_id, type="funded_by", weight=1.0)

    def _add_year_nodes(self, df: pd.DataFrame):
        if not self.year_col: return
        print("  Adding year nodes...")
        years = sorted(df[self.year_col].dropna().unique().tolist())
        for year in years:
            node_id = f"YEAR_{int(year)}"
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, type="year", year=int(year))
        for i in range(len(years) - 1):
            y1, y2 = f"YEAR_{int(years[i])}", f"YEAR_{int(years[i+1])}"
            if not self.graph.has_edge(y1, y2):
                self.graph.add_edge(y1, y2, type="consecutive_year", weight=0.3)
        for _, row in df.iterrows():
            val = row.get(self.year_col)
            if val and not pd.isna(val):
                node_id = f"YEAR_{int(val)}"
                gid     = str(row[self.id_col])
                if self.graph.has_node(node_id) and not self.graph.has_edge(gid, node_id):
                    self.graph.add_edge(gid, node_id, type="published_in", weight=0.6)

    def _add_condition_nodes(self, df: pd.DataFrame):
        print("  Adding condition nodes...")
        edges = 0
        for condition, keywords in self.conditions.items():
            node_id = f"COND_{condition}"
            mask    = df["_text"].apply(lambda t: any(kw in t for kw in keywords))
            matching = df[mask]
            if matching.empty: continue
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, type="condition", name=condition,
                                    grant_count=len(matching))
            for _, row in matching.iterrows():
                gid = str(row[self.id_col])
                if not self.graph.has_edge(gid, node_id):
                    self.graph.add_edge(gid, node_id, type="treats", weight=self.ew_treats)
                    edges += 1
        print(f"    {edges} condition edges added")

    def _add_intervention_nodes(self, df: pd.DataFrame):
        print("  Adding intervention nodes...")
        edges = 0
        for intervention, keywords in self.interventions.items():
            node_id = f"INT_{intervention}"
            mask    = df["_text"].apply(lambda t: any(kw in t for kw in keywords))
            matching = df[mask]
            if matching.empty: continue
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, type="intervention",
                                    name=intervention, grant_count=len(matching))
            for _, row in matching.iterrows():
                gid = str(row[self.id_col])
                if not self.graph.has_edge(gid, node_id):
                    self.graph.add_edge(gid, node_id, type="uses", weight=self.ew_uses)
                    edges += 1
        print(f"    {edges} intervention edges added")

    def _add_population_nodes(self, df: pd.DataFrame):
        print("  Adding population nodes...")
        edges = 0
        for population, keywords in self.populations.items():
            node_id = f"POP_{population}"
            mask    = df["_text"].apply(lambda t: any(kw in t for kw in keywords))
            matching = df[mask]
            if matching.empty: continue
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, type="population",
                                    name=population, grant_count=len(matching))
            for _, row in matching.iterrows():
                gid = str(row[self.id_col])
                if not self.graph.has_edge(gid, node_id):
                    self.graph.add_edge(gid, node_id, type="targets", weight=self.ew_targets)
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
                self.graph.add_edge(gid, "FQHC_HUB", type="is_fqhc",
                                    weight=min(score / 5, 1.0))
        print(f"    {len(fqhc)} FQHC grants connected to hub")

    def _add_similarity_edges(self, df: pd.DataFrame):
        print("  Adding similarity edges (NIH abstracts)...")
        edges = 0
        for condition, keywords in CONDITIONS.items():
            mask      = df["_text"].apply(lambda t: any(kw in t for kw in keywords))
            group_ids = df[mask][self.id_col].astype(str).tolist()
            for i in range(len(group_ids)):
                for j in range(i + 1, min(i + 6, len(group_ids))):
                    g1, g2 = group_ids[i], group_ids[j]
                    if not self.graph.has_edge(g1, g2):
                        self.graph.add_edge(g1, g2, type="similar_study",
                                            shared_condition=condition, weight=0.7)
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

        # Split by source for reporting
        nih_grants = [n for n, d in self.graph.nodes(data=True)
                      if d.get("type") == "grant" and
                         d.get("data_source", "nih_api") == "nih_api"]
        pdf_grants = [n for n, d in self.graph.nodes(data=True)
                      if d.get("type") == "grant" and
                         d.get("data_source", "") in ("pdf", "pdf_ingestion", "synthetic_fqhc")]

        result = {
            "total_nodes":  self.graph.number_of_nodes(),
            "total_edges":  self.graph.number_of_edges(),
            "density":      round(nx.density(self.graph), 6),
            "node_types":   dict(node_types),
            "edge_types":   dict(edge_types),
            "grant_count":  len(grant_nodes),
            "nih_grants":   len(nih_grants),
            "pdf_grants":   len(pdf_grants),
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
            if not matches: return []
            node_id = matches[0]
        return [n for n in self.graph.neighbors(node_id)
                if self.graph.nodes[n].get("type") == "grant"]

    def find_grants_by_intervention(self, intervention: str) -> List[str]:
        node_id = f"INT_{intervention}"
        if node_id not in self.graph:
            matches = [n for n in self.graph.nodes
                       if n.startswith("INT_") and intervention.lower() in n.lower()]
            if not matches: return []
            node_id = matches[0]
        return [n for n in self.graph.neighbors(node_id)
                if self.graph.nodes[n].get("type") == "grant"]

    def find_grants_by_population(self, population: str) -> List[str]:
        node_id = f"POP_{population}"
        if node_id not in self.graph:
            matches = [n for n in self.graph.nodes
                       if n.startswith("POP_") and population.lower() in n.lower()]
            if not matches: return []
            node_id = matches[0]
        return [n for n in self.graph.neighbors(node_id)
                if self.graph.nodes[n].get("type") == "grant"]

    def find_related_grants(self, grant_id: str,
                            max_depth: int = 2, top_k: int = 10) -> List[Dict]:
        if grant_id not in self.graph: return []
        related  = []
        visited  = {grant_id}
        queue    = [(grant_id, 0, 1.0)]
        while queue and len(related) < top_k * 3:
            current, depth, strength = queue.pop(0)
            if depth >= max_depth: continue
            for neighbor in self.graph.neighbors(current):
                if neighbor in visited: continue
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
                        "source":    ndata.get("data_source", "nih_api"),
                        "shared_condition": edge.get("shared_condition", ""),
                    })
                else:
                    queue.append((neighbor, depth + 1, strength * w))
        related.sort(key=lambda x: x["strength"], reverse=True)
        seen, unique = set(), []
        for r in related:
            if r["grant_id"] not in seen:
                seen.add(r["grant_id"]); unique.append(r)
        return unique[:top_k]

    def rfp_graph_query(self, conditions=None, interventions=None,
                        populations=None, fqhc_only=True) -> List[Dict]:
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
            fqhc_grants = {n for n in self.graph.neighbors("FQHC_HUB")
                           if self.graph.nodes[n].get("type") == "grant"}
            scores = {k: v for k, v in scores.items() if k in fqhc_grants}
        results = []
        for gid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            nd = dict(self.graph.nodes.get(gid, {}))
            results.append({
                "grant_id":        gid,
                "match_score":     round(score, 2),
                "institute":       nd.get("institute", "Unknown"),
                "year":            nd.get("year", "Unknown"),
                "is_fqhc_focused": bool(nd.get("is_fqhc_focused", False)),
                "data_source":     nd.get("data_source", "nih_api"),
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
        ax.set_title("Nodes by Type"); ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.3)

    # 2. Edge types
    ax = axes[0, 1]
    top_edges = dict(sorted(edge_types.items(), key=lambda x: x[1], reverse=True)[:8])
    if top_edges:
        ax.barh(list(top_edges.keys()), list(top_edges.values()), color="#4C72B0")
        ax.set_title("Edge Types"); ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.3)

    # 3. Top grants by centrality
    ax = axes[0, 2]
    centrality = analysis.get("top_grants_by_centrality", [])
    if centrality:
        labels = [d["grant_id"][:15] for d in centrality[:8]]
        values = [d["centrality"] for d in centrality[:8]]
        ax.barh(labels[::-1], values[::-1], color="#55A868")
        ax.set_title("Top Grants by Centrality"); ax.set_xlabel("Degree Centrality")
        ax.grid(axis="x", alpha=0.3)

    # 4. Condition coverage
    ax = axes[1, 0]
    cond_nodes = [
        (n.replace("COND_", ""), len([nb for nb in graph.neighbors(n)
                                       if graph.nodes[nb].get("type") == "grant"]))
        for n, d in graph.nodes(data=True) if d.get("type") == "condition"
    ]
    cond_nodes.sort(key=lambda x: x[1], reverse=True)
    if cond_nodes:
        names, counts = zip(*cond_nodes)
        ax.bar(names, counts, color="#C44E52", alpha=0.85)
        ax.set_title("Grants per Condition"); ax.set_ylabel("Grant Count")
        ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=0.3)

    # 5. NIH vs PDF grant breakdown
    ax = axes[1, 1]
    sources = {"NIH API": analysis.get("nih_grants", 0),
               "PDF/OCR": analysis.get("pdf_grants", 0)}
    ax.bar(sources.keys(), sources.values(),
           color=["#4C72B0", "#DD8452"], alpha=0.85)
    ax.set_title("Grant Nodes by Source"); ax.set_ylabel("Count")
    for i, (k, v) in enumerate(sources.items()):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")
    communities = analysis.get("communities", {})
    rows = [
        ["Total Nodes",        str(analysis.get("total_nodes", 0))],
        ["Total Edges",        str(analysis.get("total_edges", 0))],
        ["NIH API Grants",     str(analysis.get("nih_grants", 0))],
        ["PDF/OCR Grants",     str(analysis.get("pdf_grants", 0))],
        ["Condition Nodes",    str(node_types.get("condition", 0))],
        ["Intervention Nodes", str(node_types.get("intervention", 0))],
        ["Population Nodes",   str(node_types.get("population", 0))],
        ["Graph Density",      f"{analysis.get('density', 0):.5f}"],
        ["Communities",        str(communities.get("count", "N/A"))],
    ]
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.6)
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

    print("\n📦 STEP 1: LOADING DATA")
    print("-"*50)
    loader    = DataLoader()
    chunks_df = loader.load_chunks()

    # ── PDFs are the primary corpus ────────────────────────────────────────
    # Build the graph from the 42 PDF grants (full application text), not
    # the NIH Reporter abstracts CSV which only has 200-300 word summaries.
    if chunks_df.empty:
        raise RuntimeError(
            "No PDF chunks found. Run phase3_document_rag.py first to produce "
            f"{CHUNKS_EMB_PATH}"
        )

    pdf_grants_df = loader.extract_pdf_grants(chunks_df)
    if pdf_grants_df.empty:
        raise RuntimeError(
            "extract_pdf_grants returned empty — check data_source column "
            "in chunks CSV (expected 'pdf_ingestion', 'pdf', or 'ocr')."
        )

    # Optionally load NIH abstracts for supplemental metadata enrichment.
    # If the abstracts CSV is missing we proceed with PDFs only.
    abstracts_df = pd.DataFrame()
    if os.path.exists(ABSTRACTS_PATH):
        try:
            abstracts_df = loader.load_abstracts()
            print(f"  \u2139\ufe0f  NIH abstracts loaded for supplemental enrichment "
                  f"({len(abstracts_df)} rows)")
        except Exception as e:
            print(f"  \u26a0\ufe0f  Could not load NIH abstracts ({e}) — PDF-only graph")
    else:
        print(f"  \u2139\ufe0f  NIH abstracts CSV not found — building from PDFs only")

    # Corpus fingerprint based on PDF grants (primary corpus)
    pdf_count   = pdf_grants_df["grant_id"].nunique()
    fingerprint = hashlib.md5(
        ("|".join(sorted(pdf_grants_df["grant_id"].astype(str).tolist()))
         + f"|pdf={pdf_count}").encode()
    ).hexdigest()[:16]

    stored_fp      = persistence.get_fingerprint()
    corpus_changed = fingerprint != stored_fp

    print("\n🏗️  STEP 2: LOADING OR BUILDING GRAPH")
    print("-"*50)

    graph = None
    if not force_rebuild and persistence.exists() and not corpus_changed:
        print("✅ Graph up to date — loading from disk")
        graph = persistence.load()
        if graph is not None:
            has_conditions = any(d.get("type") == "condition"
                                 for _, d in graph.nodes(data=True))
            has_pdf = any(d.get("data_source", "") in ("pdf", "pdf_ingestion", "synthetic_fqhc")
                          for _, d in graph.nodes(data=True))
            if not has_conditions:
                print("⚠️  Graph missing condition nodes — rebuilding")
                graph = None
            elif pdf_count > 0 and not has_pdf:
                print("⚠️  Graph missing PDF grants — rebuilding")
                graph = None

    if graph is None:
        reason = "forced rebuild" if force_rebuild else (
            "corpus changed" if corpus_changed else "no cached graph"
        )
        print(f"🏗️  Building from scratch ({reason})...")
        graph_cfg = load_graph_config()
        builder = KnowledgeGraphBuilder(config=graph_cfg)
        # PDFs primary — pdf_grants_df is the main corpus; NIH abstracts supplemental
        graph   = builder.build(pdf_grants_df,
                               abstracts_df if not abstracts_df.empty else None)
        persistence.save(graph, metadata={"corpus_fingerprint": fingerprint})

    print("\n📈 STEP 3: ANALYZING GRAPH")
    print("-"*50)
    analyzer = GraphAnalyzer(graph)
    analysis = analyzer.summary()

    print(f"  Nodes:       {analysis['total_nodes']}")
    print(f"  Edges:       {analysis['total_edges']}")
    print(f"  NIH grants:  {analysis['nih_grants']}")
    print(f"  PDF grants:  {analysis['pdf_grants']}")
    print(f"  Density:     {analysis['density']}")
    if "communities" in analysis:
        print(f"  Communities: {analysis['communities']['count']}")

    print("\n🔍 STEP 4: DEMO QUERIES")
    print("-"*50)
    diabetes_grants = analyzer.find_grants_by_condition("diabetes")
    chw_grants      = analyzer.find_grants_by_intervention("CHW")
    latino_grants   = analyzer.find_grants_by_population("Latino")
    print(f"  Diabetes grants: {len(diabetes_grants)}")
    print(f"  CHW grants:      {len(chw_grants)}")
    print(f"  Latino grants:   {len(latino_grants)}")

    # Show PDF grants found via graph
    pdf_diabetes = [g for g in diabetes_grants
                    if graph.nodes[g].get("data_source") == "pdf"]
    if pdf_diabetes:
        print(f"\n  📄 PDF grants matched to diabetes: {pdf_diabetes}")

    print("\n  🎯 RFP: diabetes + CHW + Latino + FQHC")
    rfp_results = analyzer.rfp_graph_query(
        conditions=["diabetes"], interventions=["CHW"],
        populations=["Latino"], fqhc_only=True
    )
    if rfp_results:
        for r in rfp_results[:5]:
            src = "📄" if r.get("data_source", "") in ("pdf", "pdf_ingestion", "synthetic_fqhc") else "🔬"
            print(f"    {src} {r['grant_id']} — score: {r['match_score']} "
                  f"| {r['institute']}")
    else:
        print("    No FQHC matches — retrying without filter")
        rfp_results = analyzer.rfp_graph_query(
            conditions=["diabetes"], interventions=["CHW"],
            populations=["Latino"], fqhc_only=False
        )
        for r in rfp_results[:3]:
            src = "📄" if r.get("data_source", "") in ("pdf", "pdf_ingestion", "synthetic_fqhc") else "🔬"
            print(f"    {src} {r['grant_id']} — score: {r['match_score']}")

    print("\n📊 STEP 5: VISUALIZATION")
    print("-"*50)
    visualize_graph(graph, analysis)

    print("\n💾 STEP 6: SAVING RESULTS")
    print("-"*50)
    results = {
        "phase":     "phase5",
        "timestamp": datetime.now().isoformat(),
        "corpus_fingerprint": fingerprint,
        "graph_stats": {
            "nodes":      analysis["total_nodes"],
            "edges":      analysis["total_edges"],
            "density":    analysis["density"],
            "node_types": analysis["node_types"],
            "edge_types": analysis["edge_types"],
            "nih_grants": analysis["nih_grants"],
            "pdf_grants": analysis["pdf_grants"],
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
    print(f"  • phase5_knowledge_graph.gml")
    print(f"  • {GRAPH_STORE_DIR}/graph.pkl")
    print(f"\n  NIH API grants: {analysis['nih_grants']}")
    print(f"  PDF/OCR grants: {analysis['pdf_grants']}")
    print(f"\n🔄 Force rebuild: python phase5_knowledge_graph.py --rebuild")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    main(force_rebuild=args.rebuild)