# ============================================================================
# 📁 phase3_document_rag.py - SECTION-AWARE RAG WITH DOCUMENT CHUNKING
# ============================================================================

"""
PHASE 3: SECTION-AWARE RAG WITH DOCUMENT CHUNKING

Key upgrade: Section-aware embeddings
  - Chunks are prefixed with their section type before embedding
    e.g. "specific aims: <text>", "significance: <text>"
  - Separate FAISS indexes per section for targeted retrieval
  - Output CSV includes `section_type` column for Phase 4 Weaviate filtering

Output files (Phase 4 / Phase 5 compatible):
  ./phase3_results/document_chunks_with_embeddings.csv
    Columns: grant_id, text, embedding, section_type, chunk_type,
             title, institution, year, is_fqhc_focused, ...
"""

print("=" * 70)
print("🎯 PHASE 3: SECTION-AWARE RAG WITH DOCUMENT CHUNKING")
print("=" * 70)

import sys
sys.path.append('.')

import os, re, json, time
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️  Run: pip install sentence-transformers")

try:
    from config import RAG_CONFIG
except ImportError:
    RAG_CONFIG = {}

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL  = (RAG_CONFIG.get("phase3", {})
                    .get("embedding_model", "pritamdeka/S-PubMedBert-MS-MARCO"))
CHUNKS_CSV       = "./phase3_results/document_chunks_with_embeddings.csv"
FAISS_DIR        = "./phase3_results/faiss"
RESULTS_DIR      = "./phase3_results"

# Sections considered most valuable for grant-writing retrieval
HIGH_VALUE_SECTIONS = {
    "specific_aims", "significance", "innovation",
    "approach", "methods", "background", "project_summary",
}

# Prefix prepended to chunk text before embedding — teaches the model
# that "specific aims: ..." is semantically different from "methods: ..."
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
    "full_text":        "grant document:",
    "semantic_chunk":   "grant section:",
    "general":          "grant text:",
}

SECTION_PATTERNS = {
    "project_summary":   r"(?:PROJECT\s+SUMMARY|ABSTRACT)\s*[:\-]?\s*\n",
    "project_narrative": r"PROJECT\s+NARRATIVE\s*[:\-]?\s*\n",
    "specific_aims":     r"SPECIFIC\s+AIMS?\s*[:\-]?\s*\n",
    "background":        r"(?:BACKGROUND|SIGNIFICANCE\s+AND\s+INNOVATION)\s*[:\-]?\s*\n",
    "significance":      r"(?:A\.|1\.)\s*SIGNIFICANCE\s*[:\-]?\s*\n|^SIGNIFICANCE\s*[:\-]?\s*\n",
    "innovation":        r"(?:B\.|2\.)\s*INNOVATION\s*[:\-]?\s*\n|^INNOVATION\s*[:\-]?\s*\n",
    "approach":          r"(?:C\.|3\.)\s*APPROACH\s*[:\-]?\s*\n|^APPROACH\s*[:\-]?\s*\n",
    "methods":           r"(?:RESEARCH\s+)?(?:DESIGN\s+AND\s+)?METHODS?\s*[:\-]?\s*\n",
    "human_subjects":    r"(?:PROTECTION\s+OF\s+)?HUMAN\s+SUBJECTS?\s*[:\-]?\s*\n",
    "bibliography":      r"(?:BIBLIOGRAPHY|REFERENCES?\s+CITED)\s*[:\-]?\s*\n",
}


# ── Document chunker ──────────────────────────────────────────────────────────

class DocumentChunker:
    """
    Splits grant documents into section-aware chunks.
    Each chunk carries a `section_type` label used for:
      - Section-prefixed embedding
      - Per-section FAISS indexes
      - Weaviate section_type property (Phase 4)
    """

    def __init__(self, chunk_size: int = 300, overlap_sentences: int = 2):
        self.chunk_size        = chunk_size
        self.overlap_sentences = overlap_sentences

    def chunk_documents(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n🔪 Chunking {len(df)} documents (section-aware)...")
        all_chunks = []

        for idx, row in df.iterrows():
            doc       = row.to_dict()
            grant_id  = doc.get("grant_id", f"doc_{idx}")
            chunks    = self._chunk_document(doc)

            for i, chunk in enumerate(chunks):
                chunk["grant_id"]           = grant_id
                chunk["title"]              = doc.get("title", "")
                chunk["institution"]        = doc.get("institution",
                                               doc.get("institute", "Unknown"))
                chunk["year"]               = doc.get("year", 2024)
                chunk["institute"]          = doc.get("institute",
                                               doc.get("institution", "Unknown"))
                chunk["is_fqhc_focused"]    = bool(doc.get("is_fqhc_focused", False)
                                               or doc.get("has_fqhc_terms", False))
                chunk["has_fqhc_terms"]     = chunk["is_fqhc_focused"]
                chunk["fqhc_score"]         = float(doc.get("fqhc_score", 0.0))
                chunk["data_source"]        = doc.get("data_source", "nih_reporter")
                chunk["primary_condition"]  = doc.get("primary_condition", "general")
                chunk["chunk_index"]        = i
                chunk["chunk_id"]           = f"{grant_id}_chunk{len(all_chunks)}"
                all_chunks.append(chunk)

            if (idx + 1) % 200 == 0:
                print(f"  {idx+1}/{len(df)} documents chunked…")

        chunks_df = pd.DataFrame(all_chunks)
        print(f"✅ {len(chunks_df)} chunks from {len(df)} documents "
              f"(avg {len(chunks_df)/max(len(df),1):.1f}/doc)")

        if "section_type" in chunks_df.columns:
            print("   Section distribution:")
            for s, c in chunks_df["section_type"].value_counts().head(10).items():
                hv = "⭐" if s in HIGH_VALUE_SECTIONS else "  "
                print(f"   {hv} {s}: {c}")

        return chunks_df

    def _chunk_document(self, doc: dict) -> List[Dict]:
        # Prefer pre-parsed sections (from pdf_ingestion.py)
        sections = doc.get("sections")
        if isinstance(sections, str):
            try:    sections = json.loads(sections)
            except: sections = {}

        if sections and len(sections) > 1:
            return self._chunk_from_sections(sections)

        # Fall back to regex section detection on raw text
        text = self._get_text(doc)
        if not text:
            return []

        sections = self._detect_sections(text)
        if sections and len(sections) > 1:
            return self._chunk_from_sections(sections)

        # Final fallback: semantic paragraph chunking
        return self._semantic_chunks(text, "semantic_chunk")

    def _get_text(self, doc: dict) -> str:
        for field in ["full_text", "text", "abstract", "content"]:
            v = doc.get(field)
            if isinstance(v, str) and v.strip():
                return v.strip()
        parts = [str(doc.get(f, "")) for f in ["title", "abstract"] if doc.get(f)]
        return "\n\n".join(parts)

    def _detect_sections(self, text: str) -> Dict[str, str]:
        boundaries = []
        for name, pat in SECTION_PATTERNS.items():
            for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
                boundaries.append({"pos": m.start(), "end": m.end(), "name": name})
        boundaries.sort(key=lambda x: x["pos"])
        boundaries = [b for i, b in enumerate(boundaries)
                      if i == 0 or b["pos"] - boundaries[i-1]["pos"] > 50]

        if not boundaries:
            return {}

        sections = {}
        for i, b in enumerate(boundaries):
            end = boundaries[i+1]["pos"] if i+1 < len(boundaries) else len(text)
            s   = text[b["end"]:end].strip()
            if s and len(s.split()) >= 20 and b["name"] not in sections:
                sections[b["name"]] = s
        return sections

    def _chunk_from_sections(self, sections: Dict[str, str]) -> List[Dict]:
        chunks = []
        for name, text in sections.items():
            if name == "full_text" or not text or len(text.split()) < 20:
                continue
            section_type = name if name in SECTION_PATTERNS or name in HIGH_VALUE_SECTIONS \
                           else "general"
            chunks.extend(self._split_section(text, section_type))
        if not chunks:
            full = sections.get("full_text", "")
            if full:
                chunks = self._semantic_chunks(full, "semantic_chunk")
        return chunks

    def _split_section(self, text: str, section_type: str) -> List[Dict]:
        words = text.split()
        base  = {"section_type": section_type,
                 "chunk_type":   section_type,
                 "is_high_value": section_type in HIGH_VALUE_SECTIONS,
                 "word_count":   0}

        if len(words) <= self.chunk_size:
            return [{**base, "text": text, "word_count": len(words)}]

        # Split at sentence boundaries with overlap
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, cur, cw = [], [], 0

        for sent in sentences:
            sw = len(sent.split())
            if cw + sw > self.chunk_size and cur:
                chunks.append({**base,
                                "text":       " ".join(cur),
                                "word_count": cw})
                cur = cur[-self.overlap_sentences:] + [sent]
                cw  = sum(len(s.split()) for s in cur)
            else:
                cur.append(sent)
                cw += sw

        if cur:
            chunks.append({**base, "text": " ".join(cur), "word_count": cw})
        return chunks

    def _semantic_chunks(self, text: str, section_type: str) -> List[Dict]:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        base  = {"section_type": section_type,
                 "chunk_type":   "semantic_chunk",
                 "is_high_value": False,
                 "word_count":   0}
        chunks, cur, cw = [], [], 0

        for para in paras:
            pw = len(para.split())
            if cw + pw > self.chunk_size and cur:
                chunks.append({**base,
                                "text":       "\n\n".join(cur),
                                "word_count": cw,
                                "section_type": self._classify(cur)})
                cur, cw = [para], pw
            else:
                cur.append(para)
                cw += pw

        if cur:
            chunks.append({**base,
                            "text":       "\n\n".join(cur),
                            "word_count": cw,
                            "section_type": self._classify(cur)})
        return chunks

    def _classify(self, text_parts: List[str]) -> str:
        t = " ".join(text_parts).lower()
        if any(w in t for w in ["method", "approach", "design", "procedure"]):
            return "methods"
        if any(w in t for w in ["background", "significance", "rationale"]):
            return "background"
        if any(w in t for w in ["aim", "objective", "goal"]):
            return "specific_aims"
        if any(w in t for w in ["innovative", "novel", "unique"]):
            return "innovation"
        return "general"


# ── Section-aware embedder ────────────────────────────────────────────────────

class SectionAwareEmbedder:
    """
    Embeds chunks with section-type prefix so the model encodes
    section context into the vector.

    e.g. "specific aims: The goal of this project is to..."
         "methods: We will recruit 200 participants..."

    Also builds separate per-section FAISS indexes for targeted search.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"\n🧠 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"  ✅ Model loaded")
        self.section_indexes: Dict[str, faiss.Index] = {}
        self.section_chunk_ids: Dict[str, List[int]] = {}  # section → row indices in chunks_df

    def embed_chunks(self, chunks_df: pd.DataFrame,
                     batch_size: int = 64) -> pd.DataFrame:
        """
        Embed all chunks with section prefix.
        Adds `embedding` column to chunks_df.
        Returns updated DataFrame.
        """
        print(f"\n📐 Embedding {len(chunks_df)} chunks (section-prefixed)…")

        texts = []
        for _, row in chunks_df.iterrows():
            section = row.get("section_type", "general")
            prefix  = SECTION_PREFIXES.get(section, "grant text:")
            texts.append(f"{prefix} {row['text']}")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-normalize for cosine via dot product
        )

        chunks_df = chunks_df.copy()
        chunks_df["embedding"] = embeddings.tolist()
        print(f"  ✅ Embeddings shape: {embeddings.shape}")
        return chunks_df, embeddings

    def build_indexes(self, chunks_df: pd.DataFrame,
                      embeddings: np.ndarray) -> Dict[str, faiss.Index]:
        """
        Build one global FAISS index + per-section indexes.
        Returns dict of {section_type: faiss.Index}.
        """
        print("\n🗂️  Building FAISS indexes…")
        dim = embeddings.shape[1]
        os.makedirs(FAISS_DIR, exist_ok=True)

        # Global index
        global_idx = faiss.IndexFlatIP(dim)
        global_idx.add(embeddings.astype(np.float32))
        faiss.write_index(global_idx, f"{FAISS_DIR}/global.index")
        print(f"  ✅ Global index: {global_idx.ntotal} vectors")
        self.section_indexes["global"] = global_idx

        # Per-section indexes
        section_types = chunks_df["section_type"].unique() \
                        if "section_type" in chunks_df.columns else []

        for section in section_types:
            mask     = chunks_df["section_type"] == section
            row_idxs = np.where(mask)[0]
            if len(row_idxs) == 0:
                continue
            sec_embs = embeddings[row_idxs].astype(np.float32)
            sec_idx  = faiss.IndexFlatIP(dim)
            sec_idx.add(sec_embs)
            faiss.write_index(sec_idx, f"{FAISS_DIR}/{section}.index")
            self.section_indexes[section]   = sec_idx
            self.section_chunk_ids[section] = row_idxs.tolist()
            hv = "⭐" if section in HIGH_VALUE_SECTIONS else "  "
            print(f"  {hv} {section}: {sec_idx.ntotal} vectors")

        # Save section→row mapping
        with open(f"{FAISS_DIR}/section_chunk_ids.json", "w") as f:
            json.dump(self.section_chunk_ids, f)

        print(f"  ✅ {len(self.section_indexes)} indexes built "
              f"(1 global + {len(self.section_indexes)-1} section)")
        return self.section_indexes

    def load_indexes(self, chunks_df: pd.DataFrame) -> bool:
        """Load pre-built indexes from disk."""
        global_path = f"{FAISS_DIR}/global.index"
        if not os.path.exists(global_path):
            return False

        print("\n📦 Loading FAISS indexes from disk…")
        self.section_indexes["global"] = faiss.read_index(global_path)
        print(f"  ✅ Global: {self.section_indexes['global'].ntotal} vectors")

        map_path = f"{FAISS_DIR}/section_chunk_ids.json"
        if os.path.exists(map_path):
            with open(map_path) as f:
                self.section_chunk_ids = json.load(f)

        for section in self.section_chunk_ids:
            p = f"{FAISS_DIR}/{section}.index"
            if os.path.exists(p):
                self.section_indexes[section] = faiss.read_index(p)
                print(f"  ✅ {section}: {self.section_indexes[section].ntotal} vectors")

        return True

    def search(self, query: str, section_filter: Optional[str] = None,
               top_k: int = 10,
               chunks_df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Search with optional section filter.

        Args:
            query:          Natural language query
            section_filter: If set, search only that section's index
                           e.g. "specific_aims", "significance", "methods"
                           Pass None to search all chunks (global index)
            top_k:          Number of results
            chunks_df:      DataFrame to look up metadata
        """
        # Embed query with matching section prefix
        if section_filter:
            prefix      = SECTION_PREFIXES.get(section_filter, "grant text:")
            query_text  = f"{prefix} {query}"
            index       = self.section_indexes.get(section_filter,
                          self.section_indexes.get("global"))
            row_map     = self.section_chunk_ids.get(section_filter)
        else:
            query_text  = query
            index       = self.section_indexes.get("global")
            row_map     = None

        if index is None:
            print("⚠️  No index available")
            return []

        qvec = self.model.encode([query_text], normalize_embeddings=True)
        qvec = qvec.astype(np.float32)

        scores, local_ids = index.search(qvec, min(top_k, index.ntotal))

        results = []
        for score, local_id in zip(scores[0], local_ids[0]):
            if local_id < 0:
                continue
            # Map local index position back to global chunks_df row
            global_id = row_map[local_id] if row_map else local_id

            result = {"score": round(float(score), 4),
                      "local_id": int(local_id),
                      "global_id": int(global_id),
                      "section_filter": section_filter or "global"}

            if chunks_df is not None and global_id < len(chunks_df):
                row = chunks_df.iloc[global_id]
                result.update({
                    "grant_id":       row.get("grant_id", ""),
                    "title":          row.get("title", ""),
                    "institution":    row.get("institution", ""),
                    "year":           row.get("year", ""),
                    "section_type":   row.get("section_type", ""),
                    "is_fqhc_focused":bool(row.get("is_fqhc_focused", False)),
                    "text":           str(row.get("text", ""))[:300],
                    "word_count":     row.get("word_count", 0),
                    "is_high_value":  row.get("section_type", "") in HIGH_VALUE_SECTIONS,
                    "source":         "section_aware_faiss",
                })
            results.append(result)

        return results


# ── Enhanced FQHC dataset ────────────────────────────────────────────────────
# Loads data in priority order:
#   1. pdf_ingestion_output/combined_grants.csv  (PDFs + Phase 2 merged)
#   2. pdf_ingestion_output/ingested_grants.csv  (PDFs only)
#   3. phase2_output/nih_research_abstracts.csv  (abstracts only, fallback)
# Then appends 50 synthetic FQHC examples.

class EnhancedFQHCDataset:

    COMBINED_CSV = "./pdf_ingestion_output/combined_grants.csv"
    PDF_ONLY_CSV = "./pdf_ingestion_output/ingested_grants.csv"

    def __init__(self, phase2_path="./phase2_output/nih_research_abstracts.csv"):
        self.phase2_path = phase2_path

    def load(self, force_rebuild=False) -> pd.DataFrame:
        cache = "./phase3_data/enhanced_fqhc_dataset.csv"
        if not force_rebuild and os.path.exists(cache):
            df = pd.read_csv(cache)
            print(f"📂 Loaded cached dataset: {len(df)} docs")
            return df

        print("📝 Building enhanced dataset…")
        parts = []

        # Priority 1: combined CSV (PDFs + abstracts already merged by pdf_ingestion.py)
        if os.path.exists(self.COMBINED_CSV):
            df = pd.read_csv(self.COMBINED_CSV)
            if "is_fqhc_focused" not in df.columns:
                text = df["abstract"].fillna("") if "abstract" in df.columns else pd.Series([""] * len(df))
                df["is_fqhc_focused"] = text.apply(self._fqhc)
            if "fqhc_score" not in df.columns:
                text = df["abstract"].fillna("") if "abstract" in df.columns else pd.Series([""] * len(df))
                df["fqhc_score"] = text.apply(self._fqhc_score)
            pdf_n = (df["data_source"] == "pdf_ingestion").sum() if "data_source" in df.columns else 0
            abs_n = len(df) - pdf_n
            print(f"  ✅ Combined CSV: {len(df)} docs ({pdf_n} PDFs + {abs_n} abstracts)")
            parts.append(df)

        # Priority 2: PDFs only (combined CSV not yet generated)
        elif os.path.exists(self.PDF_ONLY_CSV):
            pdf_df = pd.read_csv(self.PDF_ONLY_CSV)
            if "data_source" not in pdf_df.columns:
                pdf_df["data_source"] = "pdf_ingestion"
            parts.append(pdf_df)
            print(f"  ✅ PDFs: {len(pdf_df)} docs")
            # Also load Phase 2 abstracts
            if os.path.exists(self.phase2_path):
                abs_df = pd.read_csv(self.phase2_path)
                abs_df["is_fqhc_focused"] = abs_df["abstract"].apply(self._fqhc)
                abs_df["fqhc_score"]      = abs_df["abstract"].apply(self._fqhc_score)
                abs_df["data_source"]     = "phase2_nih"
                parts.append(abs_df)
                print(f"  ✅ Phase 2 abstracts: {len(abs_df)} docs")

        # Priority 3: abstracts only
        elif os.path.exists(self.phase2_path):
            abs_df = pd.read_csv(self.phase2_path)
            abs_df["is_fqhc_focused"] = abs_df["abstract"].apply(self._fqhc)
            abs_df["fqhc_score"]      = abs_df["abstract"].apply(self._fqhc_score)
            abs_df["data_source"]     = "phase2_nih"
            parts.append(abs_df)
            print(f"  ✅ Abstracts only: {len(abs_df)} docs")
            print(f"  ⚠️  No PDF data found — run pdf_ingestion.py for richer results")

        else:
            print("  ❌ No data sources found. Check your paths.")
            return pd.DataFrame()

        # Always add synthetic FQHC examples
        parts.append(self._synthetic(50))

        combined = pd.concat(parts, ignore_index=True)
        combined = combined.drop_duplicates(subset=["grant_id"], keep="last")

        os.makedirs("./phase3_data", exist_ok=True)
        combined.to_csv(cache, index=False)

        pdf_n   = (combined["data_source"] == "pdf_ingestion").sum()  if "data_source" in combined.columns else 0
        abs_n   = (combined["data_source"] == "phase2_nih").sum()     if "data_source" in combined.columns else 0
        synth_n = (combined["data_source"] == "synthetic_fqhc").sum() if "data_source" in combined.columns else 0
        fqhc_n  = combined["is_fqhc_focused"].sum()                   if "is_fqhc_focused" in combined.columns else 0

        print(f"\n✅ Enhanced dataset: {len(combined)} docs total")
        print(f"   📄 PDF full-text:      {pdf_n}")
        print(f"   📝 NIH abstracts:      {abs_n}")
        print(f"   🧪 Synthetic FQHC:     {synth_n}")
        print(f"   🏥 FQHC-focused total: {fqhc_n}")
        return combined

    def _fqhc(self, text):
        t = str(text).lower()
        return any(k in t for k in
                   ["federally qualified health center", "fqhc",
                    "community health center", "safety-net"])

    def _fqhc_score(self, text):
        t = str(text).lower()
        w = {"federally qualified health center": 3.0, "fqhc": 3.0,
             "community health center": 2.5, "safety-net": 2.0,
             "medically underserved": 2.0, "health disparities": 1.5,
             "low-income": 1.0, "uninsured": 1.0, "medicaid": 1.0}
        return round(min(sum(v for k, v in w.items() if k in t) / sum(w.values()), 1.0), 3)

    def _synthetic(self, n=50):
        conditions   = ["diabetes", "hypertension", "depression", "HIV", "cancer"]
        populations  = ["Latino", "African American", "low-income", "Medicaid", "rural"]
        settings     = ["Federally Qualified Health Centers", "community health centers"]
        study_types  = ["randomized controlled trial", "implementation study"]
        rows = []
        for i in range(n):
            cond = np.random.choice(conditions)
            pop  = np.random.choice(populations)
            rows.append({
                "grant_id":        f"FQHC_SYNTH_{i:04d}",
                "title":           f"CHW Program for {cond} in {pop} {np.random.choice(settings)}",
                "abstract":        (f"This {np.random.choice(study_types)} evaluates a community "
                                    f"health worker-led {cond} management program for {pop} patients "
                                    f"at {np.random.choice(settings)}."),
                "year":            int(np.random.choice([2022, 2023, 2024])),
                "institute":       "NIMHD",
                "institution":     "SYNTHETIC_FQHC",
                "is_fqhc_focused": True,
                "fqhc_score":      round(0.8 + np.random.random() * 0.2, 3),
                "data_source":     "synthetic_fqhc",
                "has_fqhc_terms":  True,
            })
        return pd.DataFrame(rows)


# ── Main RAG class ────────────────────────────────────────────────────────────

class ChunkBasedRAG:
    """
    Section-aware chunk-based RAG.

    Usage:
        # First run (builds embeddings ~90 min)
        rag = ChunkBasedRAG()
        rag.setup()

        # Subsequent runs (loads from disk, ~5s)
        rag = ChunkBasedRAG(load_existing=True)
        rag.setup()

        # Search all sections
        results = rag.search("diabetes CHW intervention")

        # Search specific section only
        results = rag.search("diabetes CHW", section_filter="specific_aims")
        results = rag.search("health disparities innovation", section_filter="significance")
    """

    def __init__(self, load_existing: bool = None,
                 phase2_path: str = None,
                 force_rebuild_dataset: bool = False):
        self.load_existing = load_existing
        self.phase2_path   = phase2_path or "./phase2_output/nih_research_abstracts.csv"
        self.force_rebuild = force_rebuild_dataset

        self.chunks_df  = None
        self.embedder   = None
        self._ready     = False

    def setup(self) -> bool:
        print("\n" + "=" * 70)
        print("⚙️  SETTING UP SECTION-AWARE RAG")
        print("=" * 70)
        t0 = time.time()

        # 1. Load dataset
        dataset = EnhancedFQHCDataset(self.phase2_path)
        data    = dataset.load(force_rebuild=self.force_rebuild)
        if data.empty:
            print("❌ No data"); return False

        # 2. Chunk documents
        chunker       = DocumentChunker(chunk_size=300, overlap_sentences=2)
        self.chunks_df = chunker.chunk_documents(data)

        # 3. Embedder
        self.embedder = SectionAwareEmbedder(EMBEDDING_MODEL)

        # 4. Auto-detect whether to load or build
        global_exists = os.path.exists(f"{FAISS_DIR}/global.index")
        csv_exists    = os.path.exists(CHUNKS_CSV)

        load = self.load_existing
        if load is None:
            load = global_exists and csv_exists

        if load and global_exists and csv_exists:
            print("\n📦 Loading pre-built embeddings from disk…")
            saved = pd.read_csv(CHUNKS_CSV)
            # Restore embedding column as list
            if "embedding" in saved.columns:
                saved["embedding"] = saved["embedding"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
            # Merge saved embeddings into current chunks_df on chunk_id
            if "embedding" in saved.columns and "chunk_id" in saved.columns:
                self.chunks_df = self.chunks_df.merge(
                    saved[["chunk_id", "embedding"]], on="chunk_id", how="left"
                )
            self.embedder.load_indexes(self.chunks_df)
            print(f"✅ Loaded in {round(time.time()-t0, 1)}s")
        else:
            print("\n⚡ Building new embeddings (this takes ~90 min first time)…")
            self.chunks_df, embeddings = self.embedder.embed_chunks(self.chunks_df)
            self.embedder.build_indexes(self.chunks_df, embeddings)
            self._save(embeddings)
            print(f"✅ Built in {round(time.time()-t0, 1)}s")

        self._ready = True
        print(f"\n✅ RAG ready — {len(self.chunks_df)} chunks | "
              f"{len(self.embedder.section_indexes)} indexes")
        return True

    def _save(self, embeddings: np.ndarray):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # Save numpy embeddings
        np.save(f"{RESULTS_DIR}/chunk_embeddings.npy", embeddings)

        # Save chunks CSV with embeddings as JSON strings
        out = self.chunks_df.copy()
        if "embedding" in out.columns:
            out["embedding"] = out["embedding"].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
        else:
            out["embedding"] = [json.dumps(e.tolist()) for e in embeddings]

        out.to_csv(CHUNKS_CSV, index=False)
        print(f"💾 Saved {len(out)} chunks to {CHUNKS_CSV}")
        print(f"   Columns: {list(out.columns)}")

    def search(self, query: str,
               section_filter: Optional[str] = None,
               top_k: int = 10,
               fqhc_only: bool = False) -> List[Dict]:
        """
        Search for relevant grant chunks.

        Args:
            query:          Natural language query
            section_filter: Restrict to section type e.g. "specific_aims"
            top_k:          Number of results
            fqhc_only:      Filter to FQHC-focused grants only
        """
        if not self._ready:
            print("❌ Call setup() first"); return []

        results = self.embedder.search(
            query, section_filter=section_filter,
            top_k=top_k * 2, chunks_df=self.chunks_df
        )

        if fqhc_only:
            results = [r for r in results if r.get("is_fqhc_focused")]

        return results[:top_k]

    def compare_sections(self, query: str, top_k: int = 3) -> Dict[str, List[Dict]]:
        """
        Search query across each section type and return side-by-side results.
        Useful for demo — shows how the same query retrieves differently
        depending on which section you're looking at.
        """
        if not self._ready:
            print("❌ Call setup() first"); return {}

        print(f"\n{'='*70}")
        print(f"📊 SECTION COMPARISON: '{query}'")
        print(f"{'='*70}")

        sections_to_compare = [
            None,               # global
            "specific_aims",
            "significance",
            "methods",
            "background",
        ]

        all_results = {}
        for section in sections_to_compare:
            label   = section or "global (all sections)"
            results = self.embedder.search(
                query, section_filter=section,
                top_k=top_k, chunks_df=self.chunks_df
            )
            all_results[label] = results
            print(f"\n  {label}:")
            for i, r in enumerate(results[:top_k], 1):
                print(f"    {i}. [{r['score']:.4f}] {r.get('grant_id','')} — "
                      f"{r.get('title','')[:50]}...")

        return all_results

    def evaluate(self, test_queries: List[Dict] = None) -> Dict:
        """Evaluate retrieval performance."""
        if not self._ready:
            print("❌ Not ready"); return {}

        if test_queries is None:
            eval_path = "./phase2_output/evaluation_set.json"
            if os.path.exists(eval_path):
                with open(eval_path) as f:
                    test_queries = json.load(f)
                print(f"📋 Loaded {len(test_queries)} eval queries")
            else:
                test_queries = self._default_queries()

        metrics = {"p@1": [], "p@3": [], "p@5": [],
                   "fqhc_alignment": [], "avg_score": []}

        for qd in test_queries:
            query       = qd.get("query", "")
            relevant    = set(qd.get("relevant_grant_ids", []))
            results     = self.search(query, top_k=5)
            retrieved   = [r.get("grant_id", "") for r in results]

            for k in [1, 3, 5]:
                hits = sum(1 for r in retrieved[:k] if r in relevant)
                metrics[f"p@{k}"].append(hits / k)

            if results:
                metrics["fqhc_alignment"].append(
                    np.mean([1 if r.get("is_fqhc_focused") else 0 for r in results[:3]])
                )
                metrics["avg_score"].append(
                    np.mean([r.get("score", 0) for r in results[:3]])
                )

        avg = {k: round(float(np.mean(v)), 4) for k, v in metrics.items() if v}

        print(f"\n📊 Section-Aware RAG Evaluation:")
        print(f"   P@1: {avg.get('p@1', 0):.3f}")
        print(f"   P@3: {avg.get('p@3', 0):.3f}")
        print(f"   P@5: {avg.get('p@5', 0):.3f}")
        print(f"   FQHC alignment: {avg.get('fqhc_alignment', 0):.3f}")
        print(f"   Avg score: {avg.get('avg_score', 0):.3f}")

        return avg

    def _default_queries(self) -> List[Dict]:
        return [
            {"query": "diabetes prevention community health workers",
             "relevant_grant_ids": []},
            {"query": "behavioral health integration FQHC",
             "relevant_grant_ids": []},
            {"query": "cancer screening underserved populations",
             "relevant_grant_ids": []},
            {"query": "telehealth rural community health centers",
             "relevant_grant_ids": []},
            {"query": "CHW interventions Latino patients",
             "relevant_grant_ids": []},
        ]

    def interactive_demo(self):
        print("\n" + "=" * 70)
        print("💬 SECTION-AWARE RAG DEMO")
        print("=" * 70)
        print("Commands:")
        print("  <query>                  — search all sections")
        print("  /aims <query>            — search specific aims only")
        print("  /significance <query>    — search significance sections")
        print("  /methods <query>         — search methods sections")
        print("  /compare <query>         — compare across all sections")
        print("  /fqhc <query>            — FQHC grants only")
        print("  /quit                    — exit\n")

        while True:
            try:
                ui = input("Query > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not ui: continue
            if ui.lower() in ["/quit", "quit", "exit"]: break

            elif ui.startswith("/aims "):
                self._print_results(self.search(ui[6:], section_filter="specific_aims"))
            elif ui.startswith("/significance "):
                self._print_results(self.search(ui[14:], section_filter="significance"))
            elif ui.startswith("/methods "):
                self._print_results(self.search(ui[9:], section_filter="methods"))
            elif ui.startswith("/compare "):
                self.compare_sections(ui[9:])
            elif ui.startswith("/fqhc "):
                self._print_results(self.search(ui[6:], fqhc_only=True))
            else:
                self._print_results(self.search(ui))

    def _print_results(self, results: List[Dict]):
        print(f"\n{'─'*70}")
        print(f"📋 {len(results)} results")
        print(f"{'─'*70}")
        for i, r in enumerate(results, 1):
            hv   = "⭐" if r.get("is_high_value") else "  "
            fqhc = "🏥" if r.get("is_fqhc_focused") else "  "
            print(f"\n{i}. {hv}{fqhc} [{r['score']:.4f}] "
                  f"section:{r.get('section_type','?')}")
            print(f"   Grant:   {r.get('grant_id','N/A')}")
            if r.get("title"):
                print(f"   Title:   {r['title'][:70]}")
            if r.get("text"):
                print(f"   Text:    {r['text'][:200]}…")


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_phase3_results(rag_system, evaluation_metrics):
    """Visualize Phase 3 results."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Phase 3: Section-Aware RAG Results", fontsize=16, fontweight="bold")

        data      = getattr(rag_system, "data", pd.DataFrame())
        chunks_df = getattr(rag_system, "chunks_df", pd.DataFrame())

        # 1. Dataset composition by source
        ax = axes[0, 0]
        if not data.empty and "data_source" in data.columns:
            sc = data["data_source"].value_counts()
            ax.bar(sc.index, sc.values, color=["#4CAF50","#2196F3","#FF9800","#9C27B0"][:len(sc)])
            ax.set_title("Dataset Composition by Source")
            ax.set_ylabel("Documents")
            ax.tick_params(axis="x", rotation=30)
            for i, v in enumerate(sc.values):
                ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)

        # 2. Section distribution
        ax = axes[0, 1]
        if not chunks_df.empty and "section_type" in chunks_df.columns:
            sc2 = chunks_df["section_type"].value_counts().head(10)
            colors = ["gold" if s in HIGH_VALUE_SECTIONS else "skyblue" for s in sc2.index]
            ax.barh(sc2.index[::-1], sc2.values[::-1], color=colors[::-1])
            ax.set_title("Section Distribution (gold=high-value)")
            ax.set_xlabel("Chunks")

        # 3. Evaluation metrics
        ax = axes[0, 2]
        if evaluation_metrics:
            labels = ["P@1", "P@3", "P@5", "FQHC Align"]
            values = [
                evaluation_metrics.get("p@1", evaluation_metrics.get("precision_at_1", 0)),
                evaluation_metrics.get("p@3", evaluation_metrics.get("precision_at_3", 0)),
                evaluation_metrics.get("p@5", evaluation_metrics.get("precision_at_5", 0)),
                evaluation_metrics.get("fqhc_alignment", 0),
            ]
            bars = ax.bar(labels, values, color=["skyblue","lightgreen","salmon","gold"])
            ax.set_ylim(0, 1.1)
            ax.set_title("Evaluation Metrics")
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        # 4. Chunk size distribution
        ax = axes[1, 0]
        if not chunks_df.empty and "word_count" in chunks_df.columns:
            ax.hist(chunks_df["word_count"].dropna(), bins=40, color="lightgreen", alpha=0.8)
            mean = chunks_df["word_count"].mean()
            ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.0f}w")
            ax.set_title("Chunk Size Distribution")
            ax.set_xlabel("Words per chunk")
            ax.legend()

        # 5. FQHC vs non-FQHC pie
        ax = axes[1, 1]
        if not data.empty and "is_fqhc_focused" in data.columns:
            fq = int(data["is_fqhc_focused"].sum())
            nf = len(data) - fq
            ax.pie([fq, nf], labels=[f"FQHC ({fq})", f"Non-FQHC ({nf})"],
                   colors=["#4CAF50","#90CAF9"], autopct="%1.1f%%", startangle=90)
            ax.set_title("FQHC vs Non-FQHC")

        # 6. System summary table
        ax = axes[1, 2]
        ax.axis("off")
        ds = data.get("data_source", pd.Series([])) if not data.empty else pd.Series([])
        info = [
            ["Total docs",      str(len(data))],
            ["Total chunks",    str(len(chunks_df))],
            ["FQHC docs",       str(int(data["is_fqhc_focused"].sum())) if "is_fqhc_focused" in data.columns else "N/A"],
            ["PDF full-text",   str(int((ds=="pdf_ingestion").sum()))],
            ["NIH abstracts",   str(int((ds=="phase2_nih").sum()))],
            ["Synthetic",       str(int((ds=="synthetic_fqhc").sum()))],
            ["Section indexes", str(len(getattr(getattr(rag_system,"embedder",None),"section_indexes",{})))],
        ]
        tbl = ax.table(cellText=info, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.6)
        ax.set_title("System Summary")

        plt.tight_layout()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = f"{RESULTS_DIR}/phase3_results.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"📊 Chart saved to {out}")

    except ImportError:
        print("Matplotlib not available — skipping visualization")


# Backward-compatible alias used in some notebooks
def run_phase3_chunking(load_existing_embeddings: bool = False,
                        phase2_data_path: str = None,
                        force_rebuild_dataset: bool = False):
    """Alias for run_phase3() matching original function name."""
    return run_phase3(
        load_existing=load_existing_embeddings,
        phase2_path=phase2_data_path,
        force_rebuild_dataset=force_rebuild_dataset,
    )


# ── Main execution ────────────────────────────────────────────────────────────

def run_phase3(load_existing: bool = None,
               phase2_path: str = None,
               force_rebuild_dataset: bool = False):

    print("\n" + "=" * 70)
    print("🚀 STARTING PHASE 3: SECTION-AWARE RAG")
    print("=" * 70)

    rag = ChunkBasedRAG(
        load_existing=load_existing,
        phase2_path=phase2_path,
        force_rebuild_dataset=force_rebuild_dataset
    )
    ok = rag.setup()
    if not ok:
        return None, None

    print("\n🧪 EVALUATING…")
    metrics = rag.evaluate()

    results = {
        "phase":      "phase3_section_aware_rag",
        "timestamp":  datetime.now().isoformat(),
        "dataset":    {"total_chunks": len(rag.chunks_df),
                       "section_types": rag.chunks_df["section_type"]
                           .value_counts().to_dict()
                           if "section_type" in rag.chunks_df.columns else {}},
        "indexes":    list(rag.embedder.section_indexes.keys()),
        "metrics":    metrics,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/phase3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ PHASE 3 COMPLETE")
    print("=" * 70)
    print(f"\n📁 Outputs:")
    print(f"  • {CHUNKS_CSV}")
    print(f"     ↳ Columns include `section_type` — used by Phase 4 Weaviate filter")
    print(f"  • {FAISS_DIR}/ — per-section FAISS indexes")
    print(f"  • {RESULTS_DIR}/phase3_results.json")
    print(f"\n🔗 Phase 4 change needed:")
    print(f"   Add Property(name='sectionType', data_type=DataType.TEXT) to Weaviate schema")
    print(f"   Map chunks_df['section_type'] → Weaviate object property 'sectionType'")

    return rag, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild dataset + embeddings (use after adding new PDFs)")
    parser.add_argument("--rebuild-dataset-only", action="store_true",
                        help="Rebuild dataset cache but reuse embeddings if chunk count matches")
    args, _ = parser.parse_known_args()

    embeddings_exist = os.path.exists(f"{FAISS_DIR}/global.index")
    csv_exists       = os.path.exists(CHUNKS_CSV)
    cache_exists     = os.path.exists("./phase3_data/enhanced_fqhc_dataset.csv")

    # Auto-detect if PDF data is newer than the dataset cache
    pdf_combined     = "./pdf_ingestion_output/combined_grants.csv"
    pdf_newer        = False
    if os.path.exists(pdf_combined) and cache_exists:
        pdf_mtime   = os.path.getmtime(pdf_combined)
        cache_mtime = os.path.getmtime("./phase3_data/enhanced_fqhc_dataset.csv")
        pdf_newer   = pdf_mtime > cache_mtime
        if pdf_newer:
            print("🆕 PDF data is newer than dataset cache — will rebuild dataset")

    force_rebuild = args.rebuild or pdf_newer

    if args.rebuild or not embeddings_exist or not csv_exists:
        print("⚠️  Building embeddings from scratch (~90 min)…")
        rag, results = run_phase3(load_existing=False, force_rebuild_dataset=force_rebuild)
    else:
        print("✅ Found existing embeddings — loading from disk (fast)…")
        rag, results = run_phase3(load_existing=True, force_rebuild_dataset=force_rebuild)

    if rag:
        if results:
            visualize_phase3_results(rag, results.get("metrics", {}))
        rag.interactive_demo()