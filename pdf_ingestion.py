# ============================================================================
# pdf_ingestion.py - NIH GRANT PDF INGESTION PIPELINE
# ============================================================================

"""
Three-tier PDF extraction:
  1. pdfplumber  — primary, best for multi-column layouts
  2. pymupdf     — fallback for problematic text-based PDFs
  3. pytesseract — OCR fallback for scanned/image-based PDFs

REQUIREMENTS:
    pip install pdfplumber pymupdf pytesseract pillow pandas numpy
    apt-get install tesseract-ocr   (Linux/Colab — auto-installed if missing)
"""

import os, re, json, time, hashlib, logging
import pdfplumber
import fitz
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Section patterns ──────────────────────────────────────────────────────────
NIH_SECTION_PATTERNS = {
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
    "specific_aim_1":    r"(?:Specific\s+)?Aim\s+1[\.:]\s*",
    "specific_aim_2":    r"(?:Specific\s+)?Aim\s+2[\.:]\s*",
    "specific_aim_3":    r"(?:Specific\s+)?Aim\s+3[\.:]\s*",
}

GRANT_NUMBER_PATTERN = re.compile(r"\b([A-Z]\d{2}[A-Z]{2}\d{6}(?:-\d{2}[A-Z]?)?)\b")
YEAR_PATTERN         = re.compile(r"\b(20\d{2})\b")
INSTITUTE_MAP = {
    "CA":"NCI",  "MD":"NIMHD","MH":"NIMH", "HL":"NHLBI","AG":"NIA",
    "DK":"NIDDK","HD":"NICHD","NR":"NINR", "NS":"NINDS","EY":"NEI",
    "DC":"NIDCD","AA":"NIAAA","DA":"NIDA", "ES":"NIEHS","GM":"NIGMS",
    "LM":"NLM",  "AI":"NIAID","AR":"NIAMS","HG":"NHGRI","RR":"NCRR",
}


# ── OCR setup ─────────────────────────────────────────────────────────────────
def _setup_ocr() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        pass
    print("  📦 Installing tesseract for OCR...")
    os.system("apt-get install -qq tesseract-ocr 2>/dev/null")
    os.system("pip install pytesseract pillow -q")
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("  ✅ Tesseract ready")
        return True
    except Exception as e:
        print(f"  ⚠️  Tesseract install failed: {e}")
        return False


# ── Corpus validator ──────────────────────────────────────────────────────────
class CorpusValidator:
    """
    Validates that a document belongs in the corpus before it is chunked
    and embedded. Uses a broad, unified domain vocabulary covering:

      - Healthcare and clinical terms
      - Community services and nonprofit programming
      - Social determinants and population health
      - Grant structural signals

    A document passes if it has sufficient combined domain density — it does
    NOT need to match both health AND community terms. A youth arts grant with
    strong community services language passes just as a clinical NIH grant does.
    Only truly off-domain documents (energy, engineering, real estate, etc.)
    are rejected.

    Use force_ingest=True on NIHGrantPDFIngester to bypass validation entirely.
    """

    # Unified domain vocabulary — health + community services + nonprofit + SDOH.
    # OR logic: any term from any category contributes to the domain score.
    # This allows purely community-focused grants (arts, youth, housing) to pass
    # alongside clinical health grants, since both can be adapted for FQHC use.
    DOMAIN_TERMS = [
        # ── Clinical health ───────────────────────────────────────────────
        "diabetes", "hypertension", "cancer", "hiv", "depression", "anxiety",
        "asthma", "obesity", "opioid", "substance use", "mental health",
        "cardiovascular", "stroke", "maternal", "pediatric", "chronic disease",
        "infectious", "tuberculosis", "hepatitis", "behavioral health",
        "health center", "clinic", "hospital", "primary care", "fqhc",
        "community health", "public health", "patient", "clinical", "medical",
        "healthcare", "health care", "provider", "physician", "nurse",
        "pharmacist", "care coordinator", "telehealth", "telemedicine",
        "health disparity", "health equity", "mortality", "morbidity",
        "intervention", "treatment", "prevention", "screening", "diagnosis",
        "medication", "therapy", "counseling", "health promotion",
        "community health worker", "chw", "care management",
        # ── Community services and nonprofit programming ───────────────────
        "nonprofit", "non-profit", "community organization", "community-based",
        "community services", "direct services", "social services",
        "program participants", "program activities", "program outcomes",
        "youth development", "after school", "out of school", "mentoring",
        "workforce development", "job training", "employment services",
        "food access", "food insecurity", "nutrition", "food pantry",
        "housing stability", "affordable housing", "shelter", "transitional",
        "domestic violence", "survivor", "trauma-informed", "crisis services",
        "arts program", "arts education", "creative arts", "art therapy",
        "literacy", "tutoring", "education program", "academic support",
        "childcare", "early childhood", "head start", "family support",
        "refugee", "immigrant", "resettlement", "language access",
        "reentry", "formerly incarcerated", "justice-involved",
        "community engagement", "civic engagement", "volunteer",
        "capacity building", "technical assistance", "coalition",
        # ── Social determinants and population health ─────────────────────
        "underserved", "low-income", "low income", "medicaid", "uninsured",
        "minority", "vulnerable population", "at-risk", "marginalized",
        "racial equity", "social justice", "health equity", "disparity",
        "social determinants", "sdoh", "food insecurity", "housing instability",
        "poverty", "economic insecurity", "transportation barrier",
        "digital divide", "language barrier", "cultural competency",
        "lived experience", "trusted messenger", "peer support",
        # ── Grant and proposal structure ──────────────────────────────────
        "specific aims", "significance", "innovation", "approach",
        "nih", "hrsa", "samhsa", "cdc", "grant", "funding", "award",
        "principal investigator", "project director", "program officer",
        "abstract", "project summary", "project narrative",
        "study design", "research design", "objective", "goal",
        "deliverable", "milestone", "evaluation plan", "work plan",
        "logic model", "theory of change", "scope of work",
        "target population", "service area", "eligible population",
        "budget narrative", "budget justification", "sustainability",
        "organizational capacity", "letters of support", "partnership",
        "memorandum of understanding", "subcontractor", "fiscal sponsor",
    ]

    # Grant structural signals — documents missing all of these are probably
    # not grant applications (e.g. a research paper or news article)
    GRANT_SIGNALS = [
        "grant", "funding", "award", "application", "proposal",
        "budget", "objective", "goal", "target population",
        "evaluation", "work plan", "deliverable", "milestone",
        "project summary", "project description", "scope of work",
        "logic model", "theory of change", "organizational capacity",
        "specific aims", "significance", "approach", "methods",
        "principal investigator", "project director",
    ]

    # Off-domain exclusion clusters — if a document scores 3+ hits in any
    # single cluster AND has low domain density, it is rejected as off-domain.
    # This catches documents like energy policy reports that pass grant signal
    # checks because they use generic funding/budget/objective language.
    EXCLUSION_CLUSTERS = {
        "energy": [
            "wind energy", "solar energy", "fossil fuel", "carbon emission",
            "kilowatt", "megawatt", "turbine", "wind farm", "renewable energy",
            "energy efficiency", "utility company", "electricity generation",
            "natural gas", "coal plant", "energy grid", "power plant",
        ],
        "engineering": [
            "structural engineering", "civil engineering", "load bearing",
            "tensile strength", "hydraulic system", "mechanical design",
            "construction materials", "bridge design", "seismic", "geotechnical",
        ],
        "real_estate": [
            "property value", "square footage", "zoning ordinance", "mortgage",
            "real estate", "landlord", "lease agreement", "property tax",
            "commercial property", "residential development", "appraisal",
        ],
        "agriculture": [
            "crop yield", "livestock management", "irrigation system", "pesticide",
            "soil composition", "harvest season", "fertilizer", "crop rotation",
            "agricultural production", "farming operation", "agroforestry",
        ],
        "finance": [
            "stock market", "equity portfolio", "hedge fund", "derivatives",
            "securities trading", "asset management", "investment banking",
            "bond yield", "mutual fund", "financial instrument",
        ],
    }

    # Exclusion threshold — hits in a single cluster to trigger rejection
    EXCLUSION_CLUSTER_THRESHOLD = 3

    # Domain density below which exclusion clusters are checked
    EXCLUSION_DOMAIN_CEILING    = 0.5   # hits per 100 words

    # Minimum thresholds — loosened to accommodate non-clinical grants
    MIN_WORDS          = 300     # below this it's probably a cover page or TOC
    MIN_DOMAIN_DENSITY = 0.003   # domain term hits per word — ~3 per 1000 words
    MIN_GRANT_SIGNALS  = 1       # at least 1 grant structural signal

    def validate(self, text: str, filename: str) -> Dict:
        """
        Returns a validation result dict with:
          - validation_passed: bool
          - domain_score: float (domain term hits per 100 words)
          - grant_signal_count: int
          - word_count: int
          - warnings: list of warning strings
          - rejection_reason: str or None
        """
        result = {
            "validation_passed":  True,
            "domain_score":       0.0,
            "grant_signal_count": 0,
            "word_count":         0,
            "warnings":           [],
            "rejection_reason":   None,
        }

        if not text or not text.strip():
            result["validation_passed"] = False
            result["rejection_reason"]  = "empty text — extraction may have failed"
            return result

        tl         = text.lower()
        words      = tl.split()
        word_count = len(words)
        result["word_count"] = word_count

        # ── Check 1: minimum word count ───────────────────────────────────
        if word_count < self.MIN_WORDS:
            result["validation_passed"] = False
            result["rejection_reason"]  = (
                f"document too short ({word_count} words, "
                f"minimum {self.MIN_WORDS}) — may be a cover page or scanned image"
            )
            return result

        # ── Check 2: unified domain density (health OR community) ─────────
        domain_hits  = sum(1 for term in self.DOMAIN_TERMS if term in tl)
        domain_score = domain_hits / max(word_count / 100, 1)  # hits per 100 words
        result["domain_score"] = round(domain_score, 4)

        density = domain_hits / max(word_count, 1)
        if density < self.MIN_DOMAIN_DENSITY:
            result["validation_passed"] = False
            result["rejection_reason"]  = (
                f"low domain score ({domain_score:.2f} hits/100 words, "
                f"minimum {self.MIN_DOMAIN_DENSITY * 100:.1f}) — "
                f"document does not appear to be a health or community services grant"
            )
            result["warnings"].append(
                f"Only {domain_hits} domain terms found in {word_count} words"
            )
            return result

        # ── Check 3: at least one grant structural signal ─────────────────
        grant_hits = sum(1 for sig in self.GRANT_SIGNALS if sig in tl)
        result["grant_signal_count"] = grant_hits

        if grant_hits < self.MIN_GRANT_SIGNALS:
            result["validation_passed"] = False
            result["rejection_reason"]  = (
                f"no grant structure signals found — "
                f"document may be a research paper, report, or news article "
                f"rather than a grant application"
            )
            result["warnings"].append(
                "Missing typical grant language (budget, objectives, work plan, etc.)"
            )
            return result

        # ── Check 4: off-domain exclusion clusters ────────────────────────
        # Only run this check when domain density is low — documents with
        # strong health/community signals are allowed through regardless.
        if domain_score < self.EXCLUSION_DOMAIN_CEILING:
            for cluster_name, cluster_terms in self.EXCLUSION_CLUSTERS.items():
                cluster_hits = sum(1 for term in cluster_terms if term in tl)
                if cluster_hits >= self.EXCLUSION_CLUSTER_THRESHOLD:
                    result["validation_passed"] = False
                    result["rejection_reason"]  = (
                        f"off-domain content detected: '{cluster_name}' cluster "
                        f"({cluster_hits} hits) with low health/community domain score "
                        f"({domain_score:.2f}/100w) — document does not belong in corpus"
                    )
                    result["warnings"].append(
                        f"Strong '{cluster_name}' signal with weak health/community signal"
                    )
                    return result

        # ── Soft warnings (pass but flag) ─────────────────────────────────
        if domain_score < 0.5:
            result["warnings"].append(
                f"Low domain density ({domain_score:.2f} hits/100 words) — "
                f"verify this document is relevant to health or community services"
            )
        if grant_hits < 3:
            result["warnings"].append(
                f"Few grant structure signals ({grant_hits}) — "
                f"may be an abstract or summary rather than a full application"
            )

        return result

    def report(self, filename: str, result: Dict) -> str:
        """Returns a human-readable one-line validation summary."""
        status   = "✅ PASSED" if result["validation_passed"] else "❌ FAILED"
        reason   = f" — {result['rejection_reason']}" if result["rejection_reason"] else ""
        warnings = f" [{len(result['warnings'])} warning(s)]" if result["warnings"] else ""
        return (
            f"  {status}  {filename}  "
            f"[domain_score={result['domain_score']:.2f}/100w, "
            f"grant_signals={result['grant_signal_count']}]"
            f"{reason}{warnings}"
        )


# ── Text extractor ────────────────────────────────────────────────────────────
class PDFTextExtractor:
    """Three-tier: pdfplumber → pymupdf → pytesseract OCR"""

    def __init__(self, min_chars_per_page: int = 100, use_ocr: bool = True):
        self.min_chars   = min_chars_per_page
        self.use_ocr     = use_ocr
        self._ocr_ready  = None   # lazy init

    def extract(self, pdf_path: str) -> Tuple[str, Dict]:
        path = Path(pdf_path)
        meta = {"filename": path.name,
                "file_size_kb": round(path.stat().st_size / 1024, 1),
                "extraction_method": None, "page_count": 0,
                "extraction_quality": None, "warnings": []}

        # Tier 1 — pdfplumber
        text, m = self._pdfplumber(pdf_path)
        if not self._poor(text, m.get("page_count", 0)):
            meta.update(m)
            meta["extraction_quality"] = self._quality(text, m.get("page_count", 0))
            return text, meta

        # Tier 2 — pymupdf
        logger.warning(f"pdfplumber poor for {path.name}, trying pymupdf…")
        meta["warnings"].append("pdfplumber poor")
        text, m = self._pymupdf(pdf_path)
        if not self._poor(text, m.get("page_count", 0)):
            meta.update(m)
            meta["extraction_quality"] = self._quality(text, m.get("page_count", 0))
            return text, meta

        # Tier 3 — OCR
        if self.use_ocr:
            logger.warning(f"pymupdf poor for {path.name}, trying OCR…")
            meta["warnings"].append("pymupdf poor — using OCR")
            text, m = self._ocr(pdf_path)
            meta.update(m)
            meta["extraction_quality"] = self._quality(text, m.get("page_count", 0))
            return text, meta

        logger.warning(f"Extraction failed for {path.name}")
        meta["extraction_quality"] = "failed"
        meta.update(m)
        return "", meta

    # ── Tier 1 ──────────────────────────────────────────────────────────────
    def _pdfplumber(self, path):
        pages, n = [], 0
        try:
            with pdfplumber.open(path) as pdf:
                n = len(pdf.pages)
                for pg in pdf.pages:
                    t = pg.extract_text(x_tolerance=2, y_tolerance=2,
                                        layout=True, x_density=7.25, y_density=13)
                    pages.append(self._clean(t) if t else "")
        except Exception as e:
            logger.error(f"pdfplumber: {e}")
            return "", {"page_count": 0, "extraction_method": "pdfplumber_failed"}
        return "\n\n".join(p for p in pages if p), {
            "page_count": n, "extraction_method": "pdfplumber",
            "pages_with_text": sum(1 for p in pages if p)}

    # ── Tier 2 ──────────────────────────────────────────────────────────────
    def _pymupdf(self, path):
        pages, n = [], 0
        try:
            doc = fitz.open(path)
            n   = len(doc)
            for pg in doc:
                blks = pg.get_text("blocks", sort=True)
                t    = "\n".join(b[4].strip() for b in blks if b[6] == 0 and b[4].strip())
                pages.append(self._clean(t))
            doc.close()
        except Exception as e:
            logger.error(f"pymupdf: {e}")
            return "", {"page_count": 0, "extraction_method": "pymupdf_failed"}
        return "\n\n".join(p for p in pages if p), {
            "page_count": n, "extraction_method": "pymupdf",
            "pages_with_text": sum(1 for p in pages if p)}

    # ── Tier 3: OCR ─────────────────────────────────────────────────────────
    def _ocr(self, path):
        if self._ocr_ready is None:
            self._ocr_ready = _setup_ocr()
        if not self._ocr_ready:
            return "", {"page_count": 0, "extraction_method": "ocr_unavailable"}

        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return "", {"page_count": 0, "extraction_method": "ocr_import_failed"}

        pages, n = [], 0
        try:
            doc = fitz.open(path)
            n   = len(doc)
            print(f"    🔍 OCR: processing {n} pages in {Path(path).name}…")
            for i, pg in enumerate(doc):
                mat = fitz.Matrix(300 / 72, 300 / 72)   # 300 DPI
                pix = pg.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                t   = pytesseract.image_to_string(img, config="--oem 3 --psm 3 -l eng")
                pages.append(self._clean(t))
                if (i + 1) % 5 == 0:
                    print(f"      {i+1}/{n} pages done…")
            doc.close()
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", {"page_count": n, "extraction_method": "ocr_failed"}

        full = "\n\n".join(p for p in pages if p)
        print(f"    ✅ OCR complete: {len(full):,} chars")
        return full, {"page_count": n, "extraction_method": "ocr_tesseract",
                      "pages_with_text": sum(1 for p in pages if p)}

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _clean(self, text: str) -> str:
        if not text: return ""
        lines, out = text.split("\n"), []
        for line in lines:
            s = line.strip()
            if re.match(r"^\d+$", s): continue
            if len(s) < 3 and s not in [".", "-", "–"]: continue
            if out and out[-1].endswith("-"):
                out[-1] = out[-1][:-1] + s; continue
            out.append(line)
        text = "\n".join(out)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\f", "\n\n", text)
        for pat in [r"PHS\s+\d{3,}\s*\(Rev\.\s*[\d/]+\)",
                    r"OMB\s+No\.\s*[\d\-]+",
                    r"Continuation\s+Format\s+Page",
                    r"Page\s+\d+\s+of\s+\d+"]:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        return text.strip()

    def _poor(self, text: str, pages: int) -> bool:
        if not text or pages == 0: return True
        if len(text) / max(pages, 1) < self.min_chars: return True
        if sum(1 for c in text if ord(c) > 127) / max(len(text), 1) > 0.15: return True
        return False

    def _quality(self, text: str, pages: int) -> str:
        if not text or pages == 0: return "failed"
        cpp = len(text) / max(pages, 1)
        if cpp >= 800: return "high"
        if cpp >= 300: return "medium"
        if cpp >= 100: return "low"
        return "failed"


# ── Section parser ────────────────────────────────────────────────────────────
class NIHSectionParser:
    def parse_sections(self, text: str) -> Dict[str, str]:
        if not text: return {"full_text": ""}
        boundaries = []
        for name, pat in NIH_SECTION_PATTERNS.items():
            for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
                boundaries.append({"pos": m.start(), "end": m.end(), "name": name})
        boundaries.sort(key=lambda x: x["pos"])
        boundaries = [b for i, b in enumerate(boundaries)
                      if i == 0 or b["pos"] - boundaries[i-1]["pos"] > 50]

        if not boundaries: return {"full_text": text}

        sections = {}
        if boundaries[0]["pos"] > 100:
            pre = text[:boundaries[0]["pos"]].strip()
            if pre: sections["preamble"] = pre

        for i, b in enumerate(boundaries):
            end = boundaries[i+1]["pos"] if i+1 < len(boundaries) else len(text)
            s   = text[b["end"]:end].strip()
            if s and len(s.split()) >= 10 and b["name"] not in sections:
                sections[b["name"]] = s

        sections["full_text"] = text
        return sections


# ── Metadata extractor ────────────────────────────────────────────────────────
class NIHMetadataExtractor:
    def extract(self, filename: str, text: str, sections: Dict) -> Dict:
        meta = {"grant_id": None, "title": None, "year": None, "institute": None,
                "is_fqhc_focused": False, "fqhc_score": 0.0,
                "primary_condition": None, "study_type": None, "population": None}
        self._from_filename(filename, meta)
        if text:     self._from_text(text, meta)
        if sections: self._from_sections(sections, meta)
        tl = text.lower() if text else ""
        meta.update({
            "is_fqhc_focused":  self._fqhc(tl),
            "fqhc_score":       self._fqhc_score(tl),
            "primary_condition":self._condition(tl),
            "study_type":       self._study_type(tl),
            "population":       self._population(tl),
        })
        if not meta["grant_id"]:
            meta["grant_id"] = re.sub(r"[^A-Za-z0-9_\-]", "_", Path(filename).stem)
        return meta

    def _from_filename(self, fn, meta):
        m = GRANT_NUMBER_PATTERN.search(fn)
        if m:
            meta["grant_id"] = m.group(1)
            ic = re.match(r"[A-Z]\d{2}([A-Z]{2})", m.group(1))
            if ic: meta["institute"] = INSTITUTE_MAP.get(ic.group(1), ic.group(1))
        ym = YEAR_PATTERN.search(fn)
        if ym: meta["year"] = int(ym.group(1))

    def _from_text(self, text, meta):
        pre = text[:2000]
        if not meta["grant_id"]:
            m = GRANT_NUMBER_PATTERN.search(pre)
            if m:
                meta["grant_id"] = m.group(1)
                ic = re.match(r"[A-Z]\d{2}([A-Z]{2})", m.group(1))
                if ic: meta["institute"] = INSTITUTE_MAP.get(ic.group(1), ic.group(1))
        if not meta["title"]:
            for pat in [r"(?:Project\s+Title|TITLE)[:\s]+([^\n]{10,150})"]:
                m = re.search(pat, pre, re.IGNORECASE)
                if m: meta["title"] = m.group(1).strip(); break
        if not meta["year"]:
            ym = YEAR_PATTERN.search(pre)
            if ym:
                y = int(ym.group(1))
                if 2000 <= y <= 2030: meta["year"] = y

    def _from_sections(self, sections, meta):
        if not meta["title"] and "project_summary" in sections:
            line = sections["project_summary"].split("\n")[0].strip()
            if 10 < len(line) < 200: meta["title"] = line

    def _fqhc(self, tl):
        return any(t in tl for t in
                   ["federally qualified health center", "fqhc", "community health center"])

    def _fqhc_score(self, tl):
        w = {"federally qualified health center": 3.0, "fqhc": 3.0,
             "community health center": 2.5, "safety-net": 2.0,
             "medically underserved": 2.0, "health disparities": 1.5,
             "low-income": 1.0, "uninsured": 1.0, "medicaid": 1.0,
             "underserved": 1.0, "primary care access": 1.5}
        return round(min(sum(v for k, v in w.items() if k in tl) / sum(w.values()), 1.0), 3)

    def _condition(self, tl):
        for c, terms in {"diabetes": ["diabetes","hba1c","glycemic"],
                         "hypertension": ["hypertension","blood pressure"],
                         "depression": ["depression","phq-9"],
                         "HIV": ["hiv","aids","antiretroviral"],
                         "cancer": ["cancer","oncology","tumor"],
                         "obesity": ["obesity","bmi","weight loss"],
                         "substance_use": ["substance use","opioid","addiction"]}.items():
            if any(t in tl for t in terms): return c
        return "general"

    def _study_type(self, tl):
        if "randomized controlled trial" in tl or " rct " in tl: return "RCT"
        if "stepped-wedge" in tl:   return "stepped_wedge"
        if "cluster randomized" in tl: return "cluster_RCT"
        if "pragmatic" in tl and "trial" in tl: return "pragmatic_trial"
        if "implementation" in tl and "science" in tl: return "implementation_science"
        if "qualitative" in tl: return "qualitative"
        if "mixed method" in tl: return "mixed_methods"
        return "unspecified"

    def _population(self, tl):
        found = []
        for p, terms in {"Latino": ["latino","hispanic","latinx"],
                         "African_American": ["african american","black"],
                         "pediatric": ["pediatric","children","adolescent"],
                         "geriatric": ["older adult","geriatric","elderly"],
                         "rural": ["rural","appalachian","frontier"],
                         "low_income": ["low-income","low income","poverty"],
                         "Medicaid": ["medicaid","uninsured"]}.items():
            if any(t in tl for t in terms): found.append(p)
        return ", ".join(found) if found else "general"


# ── Main ingester ─────────────────────────────────────────────────────────────
class NIHGrantPDFIngester:

    def __init__(self, pdf_dir="./grant_documents",
                 output_dir="./pdf_ingestion_output",
                 min_quality="low", use_ocr=True,
                 force_ingest=False):
        self.pdf_dir     = Path(pdf_dir)
        self.output_dir  = Path(output_dir)
        self.min_quality = min_quality
        self.force_ingest = force_ingest   # bypass corpus validation if True
        self.extractor   = PDFTextExtractor(use_ocr=use_ocr)
        self.parser      = NIHSectionParser()
        self.meta_ex     = NIHMetadataExtractor()
        self.validator   = CorpusValidator()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def ingest_all(self, force_reprocess=False) -> pd.DataFrame:
        print(f"\n{'='*70}\n📄 NIH GRANT PDF INGESTION PIPELINE\n{'='*70}")
        print(f"   Source: {self.pdf_dir}\n   Output: {self.output_dir}")
        if self.force_ingest:
            print(f"   ⚠️  force_ingest=True — corpus validation bypassed")

        pdfs = list(self.pdf_dir.glob("**/*.pdf"))
        if not pdfs:
            print(f"❌ No PDFs in {self.pdf_dir}"); return pd.DataFrame()
        print(f"   Found: {len(pdfs)} PDF files\n")

        cache_path = self.output_dir / "ingestion_cache.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() and not force_reprocess else {}
        if cache: print(f"📦 Cache: {len(cache)} previously processed files")

        docs, stats = [], {
            "success": 0, "skipped": 0, "failed": 0,
            "low_quality": 0, "validation_failed": 0, "validation_warned": 0,
        }
        qord = {"high": 3, "medium": 2, "low": 1, "failed": 0}

        # Collect validation failures for summary report
        validation_failures = []

        for i, pdf in enumerate(pdfs):
            print(f"[{i+1}/{len(pdfs)}] {pdf.name}")
            fh = self._hash(pdf)

            if fh in cache and not force_reprocess:
                print("  ✓ Cached")
                stats["skipped"] += 1
                cp = self.output_dir / f"{cache[fh]}.json"
                if cp.exists():
                    docs.append(json.loads(cp.read_text()))
                continue

            doc = self._process(pdf)
            if doc is None:
                stats["failed"] += 1; continue

            q = doc.get("extraction_quality", "failed")
            if qord.get(q, 0) < qord.get(self.min_quality, 1):
                print(f"  ⚠️  Quality too low ({q})"); stats["low_quality"] += 1; continue

            # ── Corpus validation ─────────────────────────────────────────
            # Runs after extraction so we have clean text to assess.
            # Rejected documents are logged but not added to the corpus unless
            # force_ingest=True is set on the ingester.
            if not self.force_ingest:
                vresult = self.validator.validate(doc.get("full_text", ""), pdf.name)
                doc["validation_passed"]      = vresult["validation_passed"]
                doc["validation_domain_score"]= vresult["domain_score"]
                doc["validation_grant_signals"]= vresult["grant_signal_count"]
                doc["validation_warnings"]    = vresult["warnings"]

                print(self.validator.report(pdf.name, vresult))

                if not vresult["validation_passed"]:
                    stats["validation_failed"] += 1
                    validation_failures.append({
                        "filename":         pdf.name,
                        "rejection_reason": vresult["rejection_reason"],
                        "domain_score":     vresult["domain_score"],
                        "grant_signals":    vresult["grant_signal_count"],
                        "word_count":       vresult["word_count"],
                    })
                    print(f"  🚫 Rejected: {vresult['rejection_reason']}")
                    print(f"     To override: set force_ingest=True on NIHGrantPDFIngester")
                    continue

                if vresult["warnings"]:
                    stats["validation_warned"] += 1
                    for w in vresult["warnings"]:
                        print(f"  ⚠️  {w}")
            else:
                doc["validation_passed"]       = None   # bypassed
                doc["validation_domain_score"] = None
                doc["validation_grant_signals"]= None
                doc["validation_warnings"]     = ["validation bypassed (force_ingest=True)"]

            (self.output_dir / f"{doc['grant_id']}.json").write_text(json.dumps(doc, indent=2))
            cache[fh] = doc["grant_id"]
            docs.append(doc)
            stats["success"] += 1
            print(f"  ✅ {q} | {doc.get('extraction_method','?')} | "
                  f"FQHC:{doc['is_fqhc_focused']} | "
                  f"sections:{len(doc.get('sections',{}))} | words:{doc.get('word_count',0)}")

        cache_path.write_text(json.dumps(cache, indent=2))

        # ── Validation failure report ─────────────────────────────────────
        if validation_failures:
            vf_path = self.output_dir / "validation_failures.json"
            vf_path.write_text(json.dumps(validation_failures, indent=2))
            print(f"\n  ⚠️  {len(validation_failures)} document(s) failed corpus validation")
            print(f"     See: {vf_path}")
            print(f"     Review these files and either remove them from {self.pdf_dir}/")
            print(f"     or re-run with force_ingest=True to include them anyway.")

        if not docs:
            print("❌ No documents ingested"); return pd.DataFrame()

        df = self._to_df(docs)
        df.to_csv(self.output_dir / "ingested_grants.csv", index=False)
        self._report(df, stats)

        print(f"\n{'='*70}\n✅ INGESTION COMPLETE\n{'='*70}")
        print(f"   Success:{stats['success']}  Skipped:{stats['skipped']}  "
              f"Failed:{stats['failed']}  LowQuality:{stats['low_quality']}  "
              f"ValidationFailed:{stats['validation_failed']}  "
              f"ValidationWarned:{stats['validation_warned']}")
        print(f"   Total:{len(df)}  FQHC:{df['is_fqhc_focused'].sum()}")
        if "extraction_method" in df.columns:
            print("   Methods used:")
            for m, c in df["extraction_method"].value_counts().items():
                print(f"     {m}: {c}")
        return df

    def _process(self, pdf: Path) -> Optional[Dict]:
        t0 = time.time()
        try:
            text, emeta = self.extractor.extract(str(pdf))
            if emeta["extraction_quality"] == "failed": return None
            sections = self.parser.parse_sections(text)
            meta     = self.meta_ex.extract(pdf.name, text, sections)
            abstract = next(
                (" ".join(sections[k].split()[:500])
                 for k in ["project_summary","abstract","specific_aims","preamble"]
                 if k in sections and len(sections[k].split()) >= 50),
                " ".join(text.split()[:500])
            )
            return {
                "grant_id": meta["grant_id"],
                "title":    meta["title"] or f"Grant {meta['grant_id']}",
                "abstract": abstract, "full_text": text,
                "year":     meta["year"] or 2024,
                "institute": meta["institute"] or "Unknown",
                "is_fqhc_focused":   meta["is_fqhc_focused"],
                "fqhc_score":        meta["fqhc_score"],
                "primary_condition": meta["primary_condition"],
                "study_type":        meta["study_type"],
                "population":        meta["population"],
                "data_source":       "pdf_ingestion",
                "sections":          sections,
                "section_names":     list(sections.keys()),
                "source_pdf":        pdf.name,
                "extraction_method": emeta["extraction_method"],
                "extraction_quality":emeta["extraction_quality"],
                "page_count":        emeta["page_count"],
                "word_count":        len(text.split()),
                "char_count":        len(text),
                "ingested_at":       datetime.now().isoformat(),
                "processing_time_s": round(time.time() - t0, 2),
            }
        except Exception as e:
            logger.error(f"Failed {pdf.name}: {e}"); return None

    def _to_df(self, docs):
        rows = []
        for d in docs:
            r = {k: v for k, v in d.items() if k != "sections"}
            r["section_names"]     = ", ".join(d.get("section_names", []))
            r["has_specific_aims"] = "specific_aims" in d.get("sections", {})
            r["has_methods"]       = any(k in d.get("sections", {}) for k in ["methods","approach"])
            r["has_background"]    = "background" in d.get("sections", {})
            r["section_count"]     = len([k for k in d.get("sections", {}) if k != "full_text"])
            rows.append(r)
        df = pd.DataFrame(rows)
        for c in ["grant_id","title","abstract","year","institute",
                  "is_fqhc_focused","fqhc_score","data_source","word_count"]:
            if c not in df.columns: df[c] = None
        return df

    def _hash(self, path: Path) -> str:
        h = hashlib.md5()
        h.update(str(path.stat().st_size).encode())
        h.update(path.name.encode())
        return h.hexdigest()

    def _report(self, df, stats):
        r = {"ingestion_timestamp": datetime.now().isoformat(), "statistics": stats,
             "total_documents": len(df), "fqhc_focused": int(df["is_fqhc_focused"].sum()),
             "quality_distribution": df["extraction_quality"].value_counts().to_dict()
                 if "extraction_quality" in df.columns else {},
             "method_distribution": df["extraction_method"].value_counts().to_dict()
                 if "extraction_method" in df.columns else {},
             "avg_word_count": float(df["word_count"].mean())
                 if "word_count" in df.columns else 0,
             "validation_failures": stats.get("validation_failed", 0),
             "validation_warnings": stats.get("validation_warned", 0)}
        (self.output_dir / "ingestion_report.json").write_text(json.dumps(r, indent=2))
        print(f"\n📋 Report: {self.output_dir}/ingestion_report.json")


# ── Phase 3 helpers ───────────────────────────────────────────────────────────
class PDFAwareChunkBasedRAG:
    @staticmethod
    def load_ingested_pdfs(ingestion_dir="./pdf_ingestion_output") -> pd.DataFrame:
        p = Path(ingestion_dir) / "ingested_grants.csv"
        if not p.exists():
            raise FileNotFoundError(f"Run NIHGrantPDFIngester().ingest_all() first.")
        df = pd.read_csv(p)
        print(f"✅ Loaded {len(df)} PDFs  (FQHC: {df['is_fqhc_focused'].sum()})")
        return df

    @staticmethod
    def merge_with_existing(pdf_df, existing_df) -> pd.DataFrame:
        combined = pd.concat([existing_df, pdf_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["grant_id"], keep="last")
        print(f"✅ Merged: {len(combined)} docs  "
              f"(PDFs:{len(pdf_df)} + existing:{len(existing_df)}, "
              f"dupes removed:{len(existing_df)+len(pdf_df)-len(combined)})")
        return combined


# ── Section-aware chunker ─────────────────────────────────────────────────────
class SectionAwareChunker:
    HIGH_VALUE = ["specific_aims","significance","innovation","approach",
                  "methods","background","project_summary"]

    def __init__(self, max_words=400, overlap_sentences=2):
        self.max_words = max_words
        self.overlap   = overlap_sentences

    def chunk_document(self, doc: Dict) -> List[Dict]:
        sections = doc.get("sections", {})
        if sections and len(sections) > 1:
            chunks = []
            for name, text in sections.items():
                if name == "full_text" or not text or len(text.split()) < 20:
                    continue
                chunks.extend(self._chunk_section(text, name, doc))
            return chunks
        return self._paragraphs(doc.get("full_text", doc.get("abstract", "")), doc)

    def _chunk_section(self, text, name, doc):
        base = self._base(doc, name)
        base["is_high_value"] = name in self.HIGH_VALUE
        words = text.split()
        if len(words) <= self.max_words:
            return [{**base, "text": text, "chunk_type": name,
                     "section_name": name, "word_count": len(words),
                     "chunk_index": 0, "total_chunks_in_section": 1}]
        return self._split(text, name, base)

    def _split(self, text, name, base):
        sents  = re.split(r"(?<=[.!?])\s+", text)
        chunks, cur, cw = [], [], 0
        for s in sents:
            sw = len(s.split())
            if cw + sw > self.max_words and cur:
                chunks.append({**base, "text": " ".join(cur), "chunk_type": name,
                                "section_name": name, "word_count": cw,
                                "chunk_index": len(chunks)})
                cur = cur[-self.overlap:] + [s]
                cw  = sum(len(x.split()) for x in cur)
            else:
                cur.append(s); cw += sw
        if cur:
            chunks.append({**base, "text": " ".join(cur), "chunk_type": name,
                            "section_name": name, "word_count": cw,
                            "chunk_index": len(chunks)})
        for c in chunks: c["total_chunks_in_section"] = len(chunks)
        return chunks

    def _paragraphs(self, text, doc):
        paras  = [p.strip() for p in text.split("\n\n") if p.strip()]
        base   = self._base(doc, "full_text")
        chunks, cur, cw = [], [], 0
        for p in paras:
            pw = len(p.split())
            if cw + pw > self.max_words and cur:
                chunks.append({**base, "text": "\n\n".join(cur),
                                "chunk_type": "paragraph_block",
                                "section_name": "unstructured", "word_count": cw,
                                "chunk_index": len(chunks), "is_high_value": False})
                cur, cw = [p], pw
            else:
                cur.append(p); cw += pw
        if cur:
            chunks.append({**base, "text": "\n\n".join(cur),
                            "chunk_type": "paragraph_block",
                            "section_name": "unstructured", "word_count": cw,
                            "chunk_index": len(chunks), "is_high_value": False})
        return chunks

    def _base(self, doc, section_name):
        gid = doc.get("grant_id", "unknown")
        return {
            "chunk_id":           f"{gid}_{section_name}_{int(time.time()*1000)%10000}",
            "grant_id":           gid,
            "document_title":     doc.get("title", "Untitled"),
            "year":               doc.get("year", 2024),
            "institute":          doc.get("institute", "Unknown"),
            "is_fqhc_focused":    doc.get("is_fqhc_focused", False),
            "fqhc_score":         doc.get("fqhc_score", 0.0),
            "data_source":        doc.get("data_source", "pdf_ingestion"),
            "source_pdf":         doc.get("source_pdf", ""),
            "primary_condition":  doc.get("primary_condition", "general"),
            "study_type":         doc.get("study_type", "unspecified"),
            "population":         doc.get("population", "general"),
            "extraction_quality": doc.get("extraction_quality", "unknown"),
            "extraction_method":  doc.get("extraction_method", "unknown"),
        }

    def chunk_all_documents(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n🔪 Section-aware chunking: {len(df)} documents…")
        all_chunks = []
        for _, row in df.iterrows():
            doc = row.to_dict()
            if isinstance(doc.get("sections"), str):
                try:    doc["sections"] = json.loads(doc["sections"])
                except: doc["sections"] = {}
            if not doc.get("full_text") and doc.get("source_pdf"):
                jp = Path("./pdf_ingestion_output") / f"{doc['grant_id']}.json"
                if jp.exists():
                    stored = json.loads(jp.read_text())
                    doc["sections"]  = stored.get("sections", {})
                    doc["full_text"] = stored.get("full_text", "")
            all_chunks.extend(self.chunk_document(doc))
        chunks_df = pd.DataFrame(all_chunks)
        print(f"✅ {len(chunks_df)} chunks from {len(df)} documents")
        if not chunks_df.empty and "section_name" in chunks_df.columns:
            print("   Sections:")
            for s, c in chunks_df["section_name"].value_counts().head(8).items():
                print(f"     {s}: {c}")
        return chunks_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingester = NIHGrantPDFIngester(
        pdf_dir="./grant_documents",
        output_dir="./pdf_ingestion_output",
        min_quality="low",
        use_ocr=True,        # set False to skip scanned PDFs
        force_ingest=False,  # set True to bypass corpus validation
    )
    df = ingester.ingest_all(force_reprocess=False)

    if df.empty:
        print("⚠️  No PDFs ingested — add PDFs to ./grant_documents/ and retry.")
    else:
        chunker   = SectionAwareChunker(max_words=400, overlap_sentences=2)
        chunks_df = chunker.chunk_all_documents(df)
        chunks_df.to_csv("./pdf_ingestion_output/pdf_chunks.csv", index=False)
        print(f"💾 {len(chunks_df)} chunks → ./pdf_ingestion_output/pdf_chunks.csv")

        phase2 = Path("./phase2_output/nih_research_abstracts.csv")
        if phase2.exists():
            combined = PDFAwareChunkBasedRAG.merge_with_existing(df, pd.read_csv(phase2))
            combined.to_csv("./pdf_ingestion_output/combined_grants.csv", index=False)