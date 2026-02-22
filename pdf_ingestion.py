# ============================================================================
# 📁 pdf_ingestion.py - NIH GRANT PDF INGESTION PIPELINE
# ============================================================================

"""
PDF INGESTION PIPELINE FOR NIH GRANT DOCUMENTS
Extracts clean text from downloaded NIH grant PDFs and feeds into Phase 3.

USAGE:
    # Ingest a folder of PDFs
    ingester = NIHGrantPDFIngester(pdf_dir="./grant_documents")
    df = ingester.ingest_all()

    # Then pass to Phase 3
    rag_system = ChunkBasedRAG(documents_df=df)

REQUIREMENTS:
    pip install pdfplumber pymupdf pandas numpy
"""

import os
import re
import json
import time
import hashlib
import logging
import pdfplumber
import fitz  # pymupdf - fallback extractor
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ NIH GRANT SECTION PATTERNS ============

# NIH grants follow fairly consistent section naming conventions.
# These patterns cover the most common variants seen in R01, R34, U01, etc.
NIH_SECTION_PATTERNS = {
    "project_summary":    r"(?:PROJECT\s+SUMMARY|ABSTRACT)\s*[:\-]?\s*\n",
    "project_narrative":  r"PROJECT\s+NARRATIVE\s*[:\-]?\s*\n",
    "specific_aims":      r"SPECIFIC\s+AIMS?\s*[:\-]?\s*\n",
    "background":         r"(?:BACKGROUND|SIGNIFICANCE\s+AND\s+INNOVATION|BACKGROUND\s+AND\s+SIGNIFICANCE)\s*[:\-]?\s*\n",
    "significance":       r"(?:A\.|1\.)\s*SIGNIFICANCE\s*[:\-]?\s*\n|^SIGNIFICANCE\s*[:\-]?\s*\n",
    "innovation":         r"(?:B\.|2\.)\s*INNOVATION\s*[:\-]?\s*\n|^INNOVATION\s*[:\-]?\s*\n",
    "approach":           r"(?:C\.|3\.)\s*APPROACH\s*[:\-]?\s*\n|^APPROACH\s*[:\-]?\s*\n",
    "methods":            r"(?:RESEARCH\s+)?(?:DESIGN\s+AND\s+)?METHODS?\s*[:\-]?\s*\n",
    "human_subjects":     r"(?:PROTECTION\s+OF\s+)?HUMAN\s+SUBJECTS?\s*[:\-]?\s*\n",
    "bibliography":       r"(?:BIBLIOGRAPHY|REFERENCES?\s+CITED)\s*[:\-]?\s*\n",
    "specific_aim_1":     r"(?:Specific\s+)?Aim\s+1[\.:]\s*",
    "specific_aim_2":     r"(?:Specific\s+)?Aim\s+2[\.:]\s*",
    "specific_aim_3":     r"(?:Specific\s+)?Aim\s+3[\.:]\s*",
}

# Metadata we try to parse from filename or PDF text
GRANT_NUMBER_PATTERN = re.compile(
    r"\b([A-Z]\d{2}[A-Z]{2}\d{6}(?:-\d{2}[A-Z]?)?)\b"
)
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
INSTITUTE_MAP = {
    "CA": "NCI", "MD": "NIMHD", "MH": "NIMH", "HL": "NHLBI",
    "AG": "NIA", "DK": "NIDDK", "HD": "NICHD", "NR": "NINR",
    "NS": "NINDS", "EY": "NEI", "DC": "NIDCD", "AA": "NIAAA",
    "DA": "NIDA", "ES": "NIEHS", "GM": "NIGMS", "LM": "NLM",
    "AI": "NIAID", "AR": "NIAMS", "HG": "NHGRI", "RR": "NCRR",
}
FQHC_TERMS = [
    "federally qualified health center", "fqhc", "community health center",
    "safety-net", "medically underserved", "health disparities",
    "low-income", "uninsured", "medicaid", "underserved"
]


# ============ TEXT EXTRACTION ============

class PDFTextExtractor:
    """
    Extracts text from NIH grant PDFs using pdfplumber (primary)
    with pymupdf as fallback for scanned/problematic PDFs.
    """

    def __init__(self, min_chars_per_page: int = 100):
        self.min_chars_per_page = min_chars_per_page

    def extract(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract full text and basic metadata from a PDF.
        Returns (text, extraction_metadata).
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        metadata = {
            "filename": path.name,
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "extraction_method": None,
            "page_count": 0,
            "extraction_quality": None,
            "warnings": []
        }

        # Try pdfplumber first
        text, meta = self._extract_pdfplumber(pdf_path, metadata)

        # Fall back to pymupdf if pdfplumber yields poor results
        if self._is_poor_extraction(text, meta["page_count"]):
            logger.warning(f"pdfplumber yielded poor results for {path.name}, trying pymupdf...")
            metadata["warnings"].append("pdfplumber poor quality - used pymupdf fallback")
            text, meta = self._extract_pymupdf(pdf_path, metadata)

        # Final quality assessment
        metadata["extraction_quality"] = self._assess_quality(text, meta["page_count"])
        metadata.update(meta)

        return text, metadata

    def _extract_pdfplumber(self, pdf_path: str, metadata: Dict) -> Tuple[str, Dict]:
        """Extract using pdfplumber - better at multi-column layouts"""
        pages_text = []
        page_count = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)

                for page in pdf.pages:
                    # Extract text, preserving layout for multi-column
                    page_text = page.extract_text(
                        x_tolerance=2,
                        y_tolerance=2,
                        layout=True,
                        x_density=7.25,
                        y_density=13
                    )

                    if page_text:
                        # Clean up common PDF artifacts
                        page_text = self._clean_page_text(page_text)
                        pages_text.append(page_text)
                    else:
                        pages_text.append("")  # Keep page count consistent

        except Exception as e:
            logger.error(f"pdfplumber error on {pdf_path}: {e}")
            return "", {"page_count": 0, "extraction_method": "pdfplumber_failed"}

        full_text = "\n\n".join(p for p in pages_text if p)
        return full_text, {
            "page_count": page_count,
            "extraction_method": "pdfplumber",
            "pages_with_text": sum(1 for p in pages_text if p)
        }

    def _extract_pymupdf(self, pdf_path: str, metadata: Dict) -> Tuple[str, Dict]:
        """Extract using pymupdf - more robust for problematic PDFs"""
        pages_text = []
        page_count = 0

        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)

            for page in doc:
                # Get text blocks sorted by reading order
                blocks = page.get_text("blocks", sort=True)
                page_lines = []

                for block in blocks:
                    if block[6] == 0:  # Text block (not image)
                        block_text = block[4].strip()
                        if block_text:
                            page_lines.append(block_text)

                page_text = "\n".join(page_lines)
                page_text = self._clean_page_text(page_text)
                pages_text.append(page_text)

            doc.close()

        except Exception as e:
            logger.error(f"pymupdf error on {pdf_path}: {e}")
            return "", {"page_count": 0, "extraction_method": "pymupdf_failed"}

        full_text = "\n\n".join(p for p in pages_text if p)
        return full_text, {
            "page_count": page_count,
            "extraction_method": "pymupdf",
            "pages_with_text": sum(1 for p in pages_text if p)
        }

    def _clean_page_text(self, text: str) -> str:
        """Clean common PDF extraction artifacts"""
        if not text:
            return ""

        # Remove headers/footers that repeat (page numbers, running titles)
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip pure page numbers
            if re.match(r"^\d+$", stripped):
                continue

            # Skip very short lines that are likely artifacts
            if len(stripped) < 3 and stripped not in [".", "-", "–"]:
                continue

            # Fix broken hyphenation (word- \n word)
            if cleaned_lines and cleaned_lines[-1].endswith("-"):
                cleaned_lines[-1] = cleaned_lines[-1][:-1] + stripped
                continue

            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)    # Max 2 newlines
        text = re.sub(r"[ \t]{2,}", " ", text)     # Collapse spaces/tabs
        text = re.sub(r"\f", "\n\n", text)          # Form feeds -> paragraph breaks

        # Remove common NIH PDF boilerplate
        boilerplate_patterns = [
            r"PHS\s+\d{3,}\s*\(Rev\.\s*[\d/]+\)",
            r"OMB\s+No\.\s*[\d\-]+",
            r"Continuation\s+Format\s+Page",
            r"Page\s+\d+\s+of\s+\d+",
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        return text.strip()

    def _is_poor_extraction(self, text: str, page_count: int) -> bool:
        """Check if extraction quality is too low to use"""
        if not text or page_count == 0:
            return True

        chars_per_page = len(text) / max(page_count, 1)

        # If average chars per page is below threshold, likely a scanned PDF
        if chars_per_page < self.min_chars_per_page:
            return True

        # Check for garbled text (too many non-ASCII characters)
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
        if non_ascii_ratio > 0.15:
            return True

        return False

    def _assess_quality(self, text: str, page_count: int) -> str:
        """Assess extraction quality: high / medium / low / failed"""
        if not text or page_count == 0:
            return "failed"

        chars_per_page = len(text) / max(page_count, 1)

        if chars_per_page >= 800:
            return "high"
        elif chars_per_page >= 300:
            return "medium"
        elif chars_per_page >= 100:
            return "low"
        else:
            return "failed"


# ============ SECTION PARSER ============

class NIHSectionParser:
    """
    Parses NIH grant text into named sections.
    Designed for R01/R34/U01 format but handles variations.
    """

    def __init__(self):
        self.section_patterns = NIH_SECTION_PATTERNS

    def parse_sections(self, text: str) -> Dict[str, str]:
        """
        Split grant text into named sections.
        Returns dict of {section_name: section_text}.
        """
        if not text:
            return {"full_text": ""}

        # Find all section boundaries
        boundaries = []
        for section_name, pattern in self.section_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                boundaries.append({
                    "pos": match.start(),
                    "end": match.end(),
                    "name": section_name,
                    "header": match.group().strip()
                })

        # Sort by position
        boundaries.sort(key=lambda x: x["pos"])

        # Deduplicate overlapping matches
        boundaries = self._deduplicate_boundaries(boundaries)

        if not boundaries:
            # No sections found - return as single block
            return {"full_text": text}

        sections = {}

        # Text before first section
        if boundaries[0]["pos"] > 100:
            preamble = text[:boundaries[0]["pos"]].strip()
            if preamble:
                sections["preamble"] = preamble

        # Extract each section
        for i, boundary in enumerate(boundaries):
            start = boundary["end"]
            end = boundaries[i + 1]["pos"] if i + 1 < len(boundaries) else len(text)

            section_text = text[start:end].strip()

            if section_text and len(section_text.split()) >= 10:
                # Use first occurrence if section name appears multiple times
                section_name = boundary["name"]
                if section_name not in sections:
                    sections[section_name] = section_text
                else:
                    # Append to existing section with disambiguator
                    sections[f"{section_name}_{i}"] = section_text

        # Always keep full text for fallback search
        sections["full_text"] = text

        return sections

    def _deduplicate_boundaries(self, boundaries: List[Dict]) -> List[Dict]:
        """Remove overlapping section matches, keeping the first"""
        if len(boundaries) <= 1:
            return boundaries

        deduped = [boundaries[0]]
        for boundary in boundaries[1:]:
            last = deduped[-1]
            # Skip if too close to previous match (within 50 chars)
            if boundary["pos"] - last["pos"] > 50:
                deduped.append(boundary)

        return deduped


# ============ METADATA EXTRACTOR ============

class NIHMetadataExtractor:
    """
    Extracts grant metadata from filename and PDF text.
    Filename conventions: R01MD123456_2024.pdf, grant_doc_1.pdf, etc.
    """

    def extract(self, filename: str, text: str, sections: Dict[str, str]) -> Dict:
        """Extract all available metadata"""
        metadata = {
            "grant_id": None,
            "title": None,
            "year": None,
            "institute": None,
            "is_fqhc_focused": False,
            "fqhc_score": 0.0,
            "primary_condition": None,
            "study_type": None,
            "population": None,
        }

        # 1. Try filename first
        self._extract_from_filename(filename, metadata)

        # 2. Supplement from text
        if text:
            self._extract_from_text(text, metadata)

        # 3. Extract from specific sections
        if sections:
            self._extract_from_sections(sections, metadata)

        # 4. Compute FQHC metrics
        full_text = text.lower() if text else ""
        metadata["is_fqhc_focused"] = self._detect_fqhc(full_text)
        metadata["fqhc_score"] = self._score_fqhc(full_text)
        metadata["primary_condition"] = self._detect_condition(full_text)
        metadata["study_type"] = self._detect_study_type(full_text)
        metadata["population"] = self._detect_population(full_text)

        # 5. Fallback grant ID from filename
        if not metadata["grant_id"]:
            stem = Path(filename).stem
            metadata["grant_id"] = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)

        return metadata

    def _extract_from_filename(self, filename: str, metadata: Dict):
        """Parse grant number and year from filename"""
        # Match NIH grant number: R01MD123456, U01CA234567-02, etc.
        match = GRANT_NUMBER_PATTERN.search(filename)
        if match:
            grant_num = match.group(1)
            metadata["grant_id"] = grant_num

            # Extract institute from grant number
            ic_match = re.match(r"[A-Z]\d{2}([A-Z]{2})", grant_num)
            if ic_match:
                ic_code = ic_match.group(1)
                metadata["institute"] = INSTITUTE_MAP.get(ic_code, ic_code)

        # Extract year from filename
        year_match = YEAR_PATTERN.search(filename)
        if year_match:
            metadata["year"] = int(year_match.group(1))

    def _extract_from_text(self, text: str, metadata: Dict):
        """Extract metadata from grant text"""
        # Look for grant number in first 2000 chars
        preamble = text[:2000]

        if not metadata["grant_id"]:
            match = GRANT_NUMBER_PATTERN.search(preamble)
            if match:
                grant_num = match.group(1)
                metadata["grant_id"] = grant_num

                ic_match = re.match(r"[A-Z]\d{2}([A-Z]{2})", grant_num)
                if ic_match:
                    ic_code = ic_match.group(1)
                    metadata["institute"] = INSTITUTE_MAP.get(ic_code, ic_code)

        # Extract title - often after "Project Title:" or in all caps on early pages
        title_patterns = [
            r"(?:Project\s+Title|Title)[:\s]+([^\n]{10,150})",
            r"(?:TITLE)[:\s]+([^\n]{10,150})",
        ]
        if not metadata["title"]:
            for pattern in title_patterns:
                match = re.search(pattern, preamble, re.IGNORECASE)
                if match:
                    metadata["title"] = match.group(1).strip()
                    break

        # Extract year if not found in filename
        if not metadata["year"]:
            year_match = YEAR_PATTERN.search(preamble)
            if year_match:
                year = int(year_match.group(1))
                if 2000 <= year <= 2030:
                    metadata["year"] = year

    def _extract_from_sections(self, sections: Dict[str, str], metadata: Dict):
        """Extract metadata from parsed sections"""
        # Use project summary for title if not found yet
        if not metadata["title"] and "project_summary" in sections:
            first_line = sections["project_summary"].split("\n")[0].strip()
            if 10 < len(first_line) < 200:
                metadata["title"] = first_line

    def _detect_fqhc(self, text_lower: str) -> bool:
        primary_terms = [
            "federally qualified health center", "fqhc", "community health center"
        ]
        return any(term in text_lower for term in primary_terms)

    def _score_fqhc(self, text_lower: str) -> float:
        term_weights = {
            "federally qualified health center": 3.0, "fqhc": 3.0,
            "community health center": 2.5, "safety-net": 2.0,
            "medically underserved": 2.0, "health disparities": 1.5,
            "low-income": 1.0, "uninsured": 1.0, "medicaid": 1.0,
            "underserved": 1.0, "primary care access": 1.5,
        }
        total = sum(w for term, w in term_weights.items() if term in text_lower)
        max_possible = sum(term_weights.values())
        return round(min(total / max_possible, 1.0), 3)

    def _detect_condition(self, text_lower: str) -> Optional[str]:
        conditions = {
            "diabetes": ["diabetes", "hba1c", "glycemic"],
            "hypertension": ["hypertension", "blood pressure", "cardiovascular"],
            "depression": ["depression", "phq-9", "depressive"],
            "anxiety": ["anxiety", "gad-7"],
            "HIV": ["hiv", "aids", "antiretroviral"],
            "asthma": ["asthma", "inhaler", "bronchial"],
            "cancer": ["cancer", "oncology", "tumor", "carcinoma"],
            "obesity": ["obesity", "bmi", "weight loss", "bariatric"],
            "substance_use": ["substance use", "opioid", "addiction", "alcohol"],
        }
        for condition, terms in conditions.items():
            if any(t in text_lower for t in terms):
                return condition
        return "general"

    def _detect_study_type(self, text_lower: str) -> Optional[str]:
        if "randomized controlled trial" in text_lower or "rct" in text_lower:
            return "RCT"
        elif "stepped-wedge" in text_lower:
            return "stepped_wedge"
        elif "cluster randomized" in text_lower:
            return "cluster_RCT"
        elif "pragmatic" in text_lower and "trial" in text_lower:
            return "pragmatic_trial"
        elif "implementation" in text_lower and "science" in text_lower:
            return "implementation_science"
        elif "qualitative" in text_lower:
            return "qualitative"
        elif "mixed method" in text_lower:
            return "mixed_methods"
        elif "observational" in text_lower:
            return "observational"
        return "unspecified"

    def _detect_population(self, text_lower: str) -> Optional[str]:
        populations = {
            "Latino": ["latino", "hispanic", "latinx"],
            "African_American": ["african american", "black", "african-american"],
            "pediatric": ["pediatric", "children", "adolescent", "youth"],
            "geriatric": ["older adult", "geriatric", "elderly", "aging"],
            "rural": ["rural", "appalachian", "frontier"],
            "low_income": ["low-income", "low income", "poverty"],
            "Medicaid": ["medicaid", "uninsured"],
        }
        found = []
        for pop, terms in populations.items():
            if any(t in text_lower for t in terms):
                found.append(pop)
        return ", ".join(found) if found else "general"


# ============ MAIN INGESTER ============

class NIHGrantPDFIngester:
    """
    Main ingestion pipeline for downloaded NIH grant PDFs.

    Scans a directory, extracts text + metadata, chunks documents,
    and returns a DataFrame compatible with Phase 3's ChunkBasedRAG.

    DIRECTORY STRUCTURE:
        grant_documents/
            R01MD123456_2024.pdf
            R34MH234567_2023.pdf
            ...

    OUTPUT:
        DataFrame with columns matching Phase 3 expectations:
        grant_id, title, abstract, year, institute, is_fqhc_focused,
        fqhc_score, full_text, sections, data_source, ...
    """

    def __init__(self,
                 pdf_dir: str = "./grant_documents",
                 output_dir: str = "./pdf_ingestion_output",
                 min_quality: str = "low"):
        """
        Args:
            pdf_dir: Directory containing grant PDFs
            output_dir: Where to save extracted data
            min_quality: Minimum extraction quality to include (high/medium/low)
        """
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.min_quality = min_quality

        self.extractor = PDFTextExtractor()
        self.section_parser = NIHSectionParser()
        self.metadata_extractor = NIHMetadataExtractor()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ingestion log
        self.ingestion_log = []

    def ingest_all(self, force_reprocess: bool = False) -> pd.DataFrame:
        """
        Ingest all PDFs in pdf_dir.

        Args:
            force_reprocess: If False, skip PDFs already in cache.
                             If True, reprocess everything from scratch.

        Returns:
            DataFrame of all successfully ingested documents.
        """
        print(f"\n{'='*70}")
        print(f"📄 NIH GRANT PDF INGESTION PIPELINE")
        print(f"{'='*70}")
        print(f"   Source: {self.pdf_dir}")
        print(f"   Output: {self.output_dir}")

        # Find PDFs
        pdf_files = list(self.pdf_dir.glob("**/*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {self.pdf_dir}")
            return pd.DataFrame()

        print(f"   Found: {len(pdf_files)} PDF files\n")

        # Load cache if exists and not forcing reprocess
        cache_path = self.output_dir / "ingestion_cache.json"
        cache = {}
        if not force_reprocess and cache_path.exists():
            with open(cache_path, "r") as f:
                cache = json.load(f)
            print(f"📦 Loaded cache: {len(cache)} previously processed files")

        # Process each PDF
        all_documents = []
        stats = {"success": 0, "skipped": 0, "failed": 0, "low_quality": 0}

        for i, pdf_path in enumerate(pdf_files):
            print(f"[{i+1}/{len(pdf_files)}] Processing: {pdf_path.name}")

            # Check cache
            file_hash = self._file_hash(pdf_path)
            if file_hash in cache and not force_reprocess:
                print(f"  ✓ Cached - skipping")
                stats["skipped"] += 1
                # Load cached document
                cached_path = self.output_dir / f"{cache[file_hash]}.json"
                if cached_path.exists():
                    with open(cached_path, "r") as f:
                        all_documents.append(json.load(f))
                continue

            # Process PDF
            doc = self._process_single_pdf(pdf_path)

            if doc is None:
                stats["failed"] += 1
                continue

            quality = doc.get("extraction_quality", "failed")
            quality_order = {"high": 3, "medium": 2, "low": 1, "failed": 0}
            min_order = quality_order.get(self.min_quality, 1)

            if quality_order.get(quality, 0) < min_order:
                print(f"  ⚠️  Quality too low ({quality}) - skipping")
                stats["low_quality"] += 1
                continue

            # Save individual document
            doc_id = doc["grant_id"]
            doc_path = self.output_dir / f"{doc_id}.json"
            with open(doc_path, "w") as f:
                json.dump(doc, f, indent=2)

            # Update cache
            cache[file_hash] = doc_id

            all_documents.append(doc)
            stats["success"] += 1

            print(f"  ✅ Quality: {quality} | FQHC: {doc['is_fqhc_focused']} | "
                  f"Sections: {len(doc.get('sections', {}))} | "
                  f"Words: {doc.get('word_count', 0)}")

        # Save updated cache
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

        # Build DataFrame
        if not all_documents:
            print("\n❌ No documents successfully ingested")
            return pd.DataFrame()

        df = self._build_dataframe(all_documents)

        # Save consolidated output
        output_csv = self.output_dir / "ingested_grants.csv"
        df.to_csv(output_csv, index=False)

        # Save ingestion report
        self._save_report(df, stats)

        print(f"\n{'='*70}")
        print(f"✅ INGESTION COMPLETE")
        print(f"{'='*70}")
        print(f"   Successful:   {stats['success']}")
        print(f"   Skipped:      {stats['skipped']}")
        print(f"   Low quality:  {stats['low_quality']}")
        print(f"   Failed:       {stats['failed']}")
        print(f"   Total docs:   {len(df)}")
        print(f"   FQHC-focused: {df['is_fqhc_focused'].sum()}")
        print(f"\n📁 Saved to: {output_csv}")

        return df

    def ingest_single(self, pdf_path: str) -> Optional[Dict]:
        """
        Ingest a single PDF file.
        Returns document dict or None on failure.
        """
        return self._process_single_pdf(Path(pdf_path))

    def _process_single_pdf(self, pdf_path: Path) -> Optional[Dict]:
        """Full processing pipeline for one PDF"""
        start_time = time.time()

        try:
            # 1. Extract text
            text, extraction_meta = self.extractor.extract(str(pdf_path))

            if extraction_meta["extraction_quality"] == "failed":
                logger.warning(f"Extraction failed for {pdf_path.name}")
                return None

            # 2. Parse sections
            sections = self.section_parser.parse_sections(text)

            # 3. Extract metadata
            metadata = self.metadata_extractor.extract(
                pdf_path.name, text, sections
            )

            # 4. Build abstract from sections (priority order)
            abstract = self._build_abstract(sections, text)

            # 5. Compile document record
            doc = {
                # Core fields (Phase 3 compatible)
                "grant_id": metadata["grant_id"],
                "title": metadata["title"] or f"Grant {metadata['grant_id']}",
                "abstract": abstract,
                "full_text": text,
                "year": metadata["year"] or 2024,
                "institute": metadata["institute"] or "Unknown",
                "is_fqhc_focused": metadata["is_fqhc_focused"],
                "fqhc_score": metadata["fqhc_score"],
                "primary_condition": metadata["primary_condition"],
                "study_type": metadata["study_type"],
                "population": metadata["population"],
                "data_source": "pdf_ingestion",

                # Section data (for Phase 3 DocumentChunker)
                "sections": sections,
                "section_names": list(sections.keys()),

                # Extraction metadata
                "source_pdf": pdf_path.name,
                "extraction_method": extraction_meta["extraction_method"],
                "extraction_quality": extraction_meta["extraction_quality"],
                "page_count": extraction_meta["page_count"],
                "word_count": len(text.split()),
                "char_count": len(text),
                "extraction_warnings": extraction_meta.get("warnings", []),

                # Processing metadata
                "ingested_at": datetime.now().isoformat(),
                "processing_time_s": round(time.time() - start_time, 2),
            }

            return doc

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            return None

    def _build_abstract(self, sections: Dict[str, str], full_text: str) -> str:
        """Build abstract from sections, falling back to first 500 words"""
        # Priority order for abstract content
        for section_name in ["project_summary", "abstract", "specific_aims", "preamble"]:
            if section_name in sections and len(sections[section_name].split()) >= 50:
                # Truncate to ~500 words
                words = sections[section_name].split()
                return " ".join(words[:500])

        # Fallback: first 500 words of full text
        words = full_text.split()
        return " ".join(words[:500])

    def _build_dataframe(self, documents: List[Dict]) -> pd.DataFrame:
        """Convert document list to DataFrame, handling nested fields"""
        rows = []
        for doc in documents:
            row = {k: v for k, v in doc.items()
                   if k not in ["sections", "section_names", "extraction_warnings"]}

            # Flatten section names to string
            row["section_names"] = ", ".join(doc.get("section_names", []))
            row["has_specific_aims"] = "specific_aims" in doc.get("sections", {})
            row["has_methods"] = any(k in doc.get("sections", {})
                                      for k in ["methods", "approach"])
            row["has_background"] = "background" in doc.get("sections", {})

            # Compute section count
            row["section_count"] = len([k for k in doc.get("sections", {})
                                         if k != "full_text"])

            rows.append(row)

        df = pd.DataFrame(rows)

        # Ensure Phase 3 compatible columns exist
        required_cols = ["grant_id", "title", "abstract", "year", "institute",
                         "is_fqhc_focused", "fqhc_score", "data_source", "word_count"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        return df

    def _file_hash(self, path: Path) -> str:
        """Generate hash for file caching"""
        h = hashlib.md5()
        h.update(str(path.stat().st_size).encode())
        h.update(path.name.encode())
        return h.hexdigest()

    def _save_report(self, df: pd.DataFrame, stats: Dict):
        """Save ingestion report"""
        report = {
            "ingestion_timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "total_documents": len(df),
            "fqhc_focused": int(df["is_fqhc_focused"].sum()),
            "quality_distribution": df["extraction_quality"].value_counts().to_dict()
                if "extraction_quality" in df.columns else {},
            "institute_distribution": df["institute"].value_counts().to_dict()
                if "institute" in df.columns else {},
            "condition_distribution": df["primary_condition"].value_counts().to_dict()
                if "primary_condition" in df.columns else {},
            "avg_word_count": float(df["word_count"].mean())
                if "word_count" in df.columns else 0,
        }

        report_path = self.output_dir / "ingestion_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📋 Report saved to: {report_path}")


# ============ PHASE 3 INTEGRATION HELPER ============

class PDFAwareChunkBasedRAG:
    """
    Drop-in extension of Phase 3's ChunkBasedRAG that understands
    section structure from PDF ingestion. Inherits all Phase 3 behavior
    and adds section-aware chunking.
    """

    @staticmethod
    def load_ingested_pdfs(ingestion_dir: str = "./pdf_ingestion_output") -> pd.DataFrame:
        """
        Load previously ingested PDFs for use in Phase 3.

        Usage in Phase 3:
            from pdf_ingestion import PDFAwareChunkBasedRAG

            pdf_df = PDFAwareChunkBasedRAG.load_ingested_pdfs()
            rag = ChunkBasedRAG(documents_df=pdf_df)
        """
        csv_path = Path(ingestion_dir) / "ingested_grants.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No ingested data found at {csv_path}. "
                f"Run NIHGrantPDFIngester().ingest_all() first."
            )

        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} ingested PDFs from {csv_path}")
        print(f"   FQHC-focused: {df['is_fqhc_focused'].sum()}")
        return df

    @staticmethod
    def merge_with_existing(
        pdf_df: pd.DataFrame,
        existing_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge PDF documents with existing Phase 2/synthetic data.
        Avoids duplicates by grant_id.

        Usage:
            pdf_df = PDFAwareChunkBasedRAG.load_ingested_pdfs()
            phase2_df = pd.read_csv('./phase2_output/nih_research_abstracts.csv')
            combined_df = PDFAwareChunkBasedRAG.merge_with_existing(pdf_df, phase2_df)
        """
        combined = pd.concat([existing_df, pdf_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["grant_id"], keep="last")

        print(f"✅ Merged dataset: {len(combined)} documents")
        print(f"   From PDFs: {len(pdf_df)}")
        print(f"   From existing: {len(existing_df)}")
        print(f"   Duplicates removed: {len(existing_df) + len(pdf_df) - len(combined)}")

        return combined


# ============ SECTION-AWARE CHUNKER (Phase 3 upgrade) ============

class SectionAwareChunker:
    """
    Upgraded chunker that uses parsed PDF sections for higher-quality chunks.
    Produces chunks that preserve full section context rather than splitting
    across arbitrary word boundaries.

    Replaces DocumentChunker in Phase 3 when working with PDF-ingested data.
    """

    # Sections most useful for RFP matching
    HIGH_VALUE_SECTIONS = [
        "specific_aims", "significance", "innovation", "approach",
        "methods", "background", "project_summary"
    ]

    def __init__(self, max_words_per_chunk: int = 400, overlap_sentences: int = 2):
        self.max_words = max_words_per_chunk
        self.overlap_sentences = overlap_sentences

    def chunk_document(self, doc: Dict) -> List[Dict]:
        """
        Chunk a single document using section structure.
        Falls back to paragraph chunking for documents without sections.
        """
        chunks = []
        grant_id = doc.get("grant_id", "unknown")
        sections = doc.get("sections", {})

        if sections and len(sections) > 1:
            # Section-aware chunking
            for section_name, section_text in sections.items():
                if section_name == "full_text":
                    continue  # Skip the combined text

                if not section_text or len(section_text.split()) < 20:
                    continue

                section_chunks = self._chunk_section(
                    text=section_text,
                    section_name=section_name,
                    grant_id=grant_id,
                    doc=doc
                )
                chunks.extend(section_chunks)
        else:
            # Fallback: chunk full text by paragraphs
            full_text = doc.get("full_text", doc.get("abstract", ""))
            chunks = self._chunk_by_paragraphs(full_text, grant_id, doc)

        return chunks

    def _chunk_section(self, text: str, section_name: str,
                       grant_id: str, doc: Dict) -> List[Dict]:
        """Chunk a single section, keeping it whole if small enough"""
        words = text.split()
        base_meta = self._base_metadata(doc, section_name)
        base_meta["is_high_value"] = section_name in self.HIGH_VALUE_SECTIONS

        if len(words) <= self.max_words:
            # Section fits in one chunk
            return [{
                **base_meta,
                "text": text,
                "chunk_type": section_name,
                "section_name": section_name,
                "word_count": len(words),
                "chunk_index": 0,
                "total_chunks_in_section": 1,
            }]

        # Split large sections into overlapping chunks
        return self._split_with_overlap(text, section_name, base_meta)

    def _split_with_overlap(self, text: str, section_name: str,
                            base_meta: Dict) -> List[Dict]:
        """Split text into overlapping chunks at sentence boundaries"""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_sentences = []
        current_words = 0

        for sentence in sentences:
            sent_words = len(sentence.split())

            if current_words + sent_words > self.max_words and current_sentences:
                # Save current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    **base_meta,
                    "text": chunk_text,
                    "chunk_type": section_name,
                    "section_name": section_name,
                    "word_count": current_words,
                    "chunk_index": len(chunks),
                })

                # Overlap: keep last N sentences for context
                overlap = current_sentences[-self.overlap_sentences:]
                current_sentences = overlap + [sentence]
                current_words = sum(len(s.split()) for s in current_sentences)
            else:
                current_sentences.append(sentence)
                current_words += sent_words

        # Final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                **base_meta,
                "text": chunk_text,
                "chunk_type": section_name,
                "section_name": section_name,
                "word_count": current_words,
                "chunk_index": len(chunks),
            })

        # Set total chunks in section
        for chunk in chunks:
            chunk["total_chunks_in_section"] = len(chunks)

        return chunks

    def _chunk_by_paragraphs(self, text: str, grant_id: str, doc: Dict) -> List[Dict]:
        """Fallback paragraph-based chunking"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        base_meta = self._base_metadata(doc, "full_text")
        chunks = []
        current_paras = []
        current_words = 0

        for para in paragraphs:
            para_words = len(para.split())
            if current_words + para_words > self.max_words and current_paras:
                chunk_text = "\n\n".join(current_paras)
                chunks.append({
                    **base_meta,
                    "text": chunk_text,
                    "chunk_type": "paragraph_block",
                    "section_name": "unstructured",
                    "word_count": current_words,
                    "chunk_index": len(chunks),
                    "is_high_value": False,
                })
                current_paras = [para]
                current_words = para_words
            else:
                current_paras.append(para)
                current_words += para_words

        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append({
                **base_meta,
                "text": chunk_text,
                "chunk_type": "paragraph_block",
                "section_name": "unstructured",
                "word_count": current_words,
                "chunk_index": len(chunks),
                "is_high_value": False,
            })

        return chunks

    def _base_metadata(self, doc: Dict, section_name: str) -> Dict:
        grant_id = doc.get("grant_id", "unknown")
        return {
            "chunk_id": f"{grant_id}_{section_name}_{int(time.time()*1000) % 10000}",
            "grant_id": grant_id,
            "document_title": doc.get("title", "Untitled"),
            "year": doc.get("year", 2024),
            "institute": doc.get("institute", "Unknown"),
            "is_fqhc_focused": doc.get("is_fqhc_focused", False),
            "fqhc_score": doc.get("fqhc_score", 0.0),
            "data_source": doc.get("data_source", "pdf_ingestion"),
            "source_pdf": doc.get("source_pdf", ""),
            "primary_condition": doc.get("primary_condition", "general"),
            "study_type": doc.get("study_type", "unspecified"),
            "population": doc.get("population", "general"),
            "extraction_quality": doc.get("extraction_quality", "unknown"),
        }

    def chunk_all_documents(self, documents_df: pd.DataFrame) -> pd.DataFrame:
        """Chunk all documents in a DataFrame"""
        print(f"\n🔪 Section-aware chunking: {len(documents_df)} documents...")

        all_chunks = []
        for _, row in documents_df.iterrows():
            doc = row.to_dict()

            # Load sections if stored as JSON string
            if isinstance(doc.get("sections"), str):
                try:
                    doc["sections"] = json.loads(doc["sections"])
                except Exception:
                    doc["sections"] = {}

            # Load full_text from individual JSON if available
            if not doc.get("full_text") and doc.get("source_pdf"):
                json_path = Path("./pdf_ingestion_output") / f"{doc['grant_id']}.json"
                if json_path.exists():
                    with open(json_path) as f:
                        stored = json.load(f)
                        doc["sections"] = stored.get("sections", {})
                        doc["full_text"] = stored.get("full_text", "")

            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        chunks_df = pd.DataFrame(all_chunks)

        print(f"✅ Created {len(chunks_df)} chunks from {len(documents_df)} documents")
        if not chunks_df.empty and "section_name" in chunks_df.columns:
            print(f"   Section distribution:")
            for section, count in chunks_df["section_name"].value_counts().head(8).items():
                print(f"     {section}: {count}")

        return chunks_df


# ============ MAIN / USAGE EXAMPLE ============

if __name__ == "__main__":
    import sys

    print("📄 NIH Grant PDF Ingestion Pipeline")
    print("=" * 50)

    # --- STEP 1: Ingest PDFs ---
    ingester = NIHGrantPDFIngester(
        pdf_dir="./grant_documents",        # Your PDF folder
        output_dir="./pdf_ingestion_output",
        min_quality="low"                   # Include all non-failed extractions
    )

    df = ingester.ingest_all(force_reprocess=False)  # Uses cache for speed

    if df.empty:
        print("\n⚠️  No PDFs ingested. Add PDF files to ./grant_documents/ and retry.")
        sys.exit(0)

    print(f"\n📊 Ingested {len(df)} documents")

    # --- STEP 2: Section-aware chunking ---
    chunker = SectionAwareChunker(max_words_per_chunk=400, overlap_sentences=2)
    chunks_df = chunker.chunk_all_documents(df)

    # Save chunks for Phase 3
    chunks_df.to_csv("./pdf_ingestion_output/pdf_chunks.csv", index=False)
    print(f"💾 Saved {len(chunks_df)} chunks to ./pdf_ingestion_output/pdf_chunks.csv")

    # --- STEP 3: Merge with existing Phase 2/3 data (optional) ---
    phase2_path = "./phase2_output/nih_research_abstracts.csv"
    if Path(phase2_path).exists():
        phase2_df = pd.read_csv(phase2_path)
        combined_df = PDFAwareChunkBasedRAG.merge_with_existing(df, phase2_df)
        combined_df.to_csv("./pdf_ingestion_output/combined_grants.csv", index=False)
        print(f"💾 Combined dataset saved: {len(combined_df)} documents")

    print("\n✅ PDF ingestion complete!")
    print("\nTo use in Phase 3:")
    print("  from pdf_ingestion import NIHGrantPDFIngester, PDFAwareChunkBasedRAG")
    print("  pdf_df = PDFAwareChunkBasedRAG.load_ingested_pdfs()")
    print("  # Pass to ChunkBasedRAG as before")