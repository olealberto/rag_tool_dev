# ============================================================================
# 📁 application_assistant.py - GRANT APPLICATION ASSISTANT
# ============================================================================

"""
USER-FACING GRANT WRITING ASSISTANT

Given an empty grant application PDF, detects its sections and retrieves
the most relevant chunks from the user's own previously uploaded grants
for each section — so the user can select and adapt their best prior work.

Usage in Colab:
    from query_pipeline import GrantQueryPipeline
    from application_assistant import ApplicationAssistant

    pipeline = GrantQueryPipeline()
    pipeline.setup()

    assistant = ApplicationAssistant(pipeline)

    # Index the user's own grants (run once per session)
    assistant.set_user_grants(["anderson-melissa-application", "smith-r01-2022"])

    # Find relevant chunks for each section of a new application
    results = assistant.find_for_application(
        pdf_path="./my_empty_application.pdf",
        topic="FQHC diabetes CHW program serving Latino patients"
    )

    # Or run interactively
    assistant.interactive()
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# ============ SECTION DETECTOR ============

class SectionDetector:
    """
    Detects section headers from a grant application PDF.
    Uses heuristics — no ML model needed for structured grant templates.

    Handles:
    - All-caps headers:         "SPECIFIC AIMS"
    - Numbered sections:        "1. Background and Significance"
    - Lettered sections:        "A. Specific Aims"
    - Title-case short lines:   "Project Narrative"
    - Colon-terminated:         "Research Strategy:"
    - Roman numerals:           "III. Innovation"
    """

    # Headers we always want to detect regardless of formatting
    KNOWN_HEADERS = [
        # NIH
        "specific aims", "significance", "innovation", "approach",
        "background", "project summary", "project narrative",
        "research strategy", "methods", "preliminary studies",
        "human subjects", "vertebrate animals", "bibliography",
        "resource sharing", "authentication of key resources",
        # HRSA / FQHC
        "need", "community need", "problem statement", "target population",
        "proposed solution", "work plan", "evaluation plan",
        "organizational capacity", "budget narrative", "budget justification",
        "sustainability", "organizational background",
        # SAMHSA / behavioral health
        "project description", "goals and objectives", "logic model",
        "evidence base", "implementation plan", "staff qualifications",
        # NSF
        "intellectual merit", "broader impacts", "facilities",
        "data management", "postdoctoral mentoring",
        # Generic
        "introduction", "executive summary", "abstract",
        "literature review", "theoretical framework",
        "timeline", "dissemination", "references",
    ]

    # Patterns that strongly indicate a header line
    HEADER_PATTERNS = [
        r"^[A-Z][A-Z\s\-/&]{4,}$",              # ALL CAPS (min 5 chars)
        r"^\d+[\.\)]\s+[A-Z][A-Za-z\s]{3,}$",   # 1. Title or 1) Title
        r"^[A-Z][\.\)]\s+[A-Z][A-Za-z\s]{3,}$", # A. Title or A) Title
        r"^(?:I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+",  # Roman numerals
        r"^Section\s+\d+",                        # Section 1, Section 2
        r"^Part\s+[A-Z\d]",                       # Part A, Part 1
    ]

    def __init__(self):
        self._known_lower = {h.lower() for h in self.KNOWN_HEADERS}
        self._patterns    = [re.compile(p) for p in self.HEADER_PATTERNS]

    def detect_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Extract section headers from a PDF file.
        Returns list of detected section names in document order.
        """
        text_lines = self._extract_lines(pdf_path)
        if not text_lines:
            print(f"  ⚠️  Could not extract text from {pdf_path}")
            return []

        headers = self._detect_headers(text_lines)
        print(f"  ✅ Detected {len(headers)} sections in {os.path.basename(pdf_path)}")
        for h in headers:
            print(f"     • {h}")
        return headers

    def detect_from_text(self, text: str) -> List[str]:
        """Detect sections from raw text string."""
        lines = [l.strip() for l in text.split("\n")]
        return self._detect_headers(lines)

    def detect_from_list(self, sections: List[str]) -> List[str]:
        """Pass-through for manually specified sections."""
        return [s.strip() for s in sections if s.strip()]

    def _extract_lines(self, pdf_path: str) -> List[str]:
        """Extract text lines from PDF using pdfplumber, fall back to pypdf."""
        lines = []
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Try to get words with font size for better header detection
                    try:
                        words = page.extract_words(extra_attrs=["size", "fontname"])
                        if words:
                            # Group words into lines by approximate y position
                            by_y = defaultdict(list)
                            for w in words:
                                y_key = round(float(w.get("top", 0)) / 3) * 3
                                by_y[y_key].append(w)
                            for y in sorted(by_y.keys()):
                                line_words = sorted(by_y[y], key=lambda w: w.get("x0", 0))
                                line_text  = " ".join(w["text"] for w in line_words).strip()
                                # Store with font size of first word for header detection
                                size = float(line_words[0].get("size", 10))
                                lines.append((line_text, size))
                            continue
                    except Exception:
                        pass
                    # Fallback: plain text extraction
                    page_text = page.extract_text() or ""
                    for line in page_text.split("\n"):
                        lines.append((line.strip(), 10.0))
            return lines
        except ImportError:
            pass

        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                for line in (page.extract_text() or "").split("\n"):
                    lines.append((line.strip(), 10.0))
            return lines
        except ImportError:
            pass

        # Plain text fallback if it's actually a text file
        try:
            with open(pdf_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    lines.append((line.strip(), 10.0))
        except Exception:
            pass

        return lines

    def _detect_headers(self, lines) -> List[str]:
        """
        Score each line for likelihood of being a section header.
        Returns deduplicated list of detected headers in order.
        """
        # Normalise: accept both (text, size) tuples and plain strings
        normalized = []
        sizes      = []
        for item in lines:
            if isinstance(item, tuple):
                normalized.append(item[0])
                sizes.append(item[1])
            else:
                normalized.append(str(item))
                sizes.append(10.0)

        # Compute median font size to identify large-font headers
        if sizes:
            sorted_s = sorted(sizes)
            median_size = sorted_s[len(sorted_s) // 2]
        else:
            median_size = 10.0

        headers = []
        seen    = set()

        for i, (line, size) in enumerate(zip(normalized, sizes)):
            if not line or len(line) < 3:
                continue

            score  = 0
            line_l = line.lower().strip().rstrip(":")

            # Known header match (highest confidence)
            if line_l in self._known_lower:
                score += 10
            else:
                # Partial match — line contains a known header
                for known in self._known_lower:
                    if known in line_l and len(line_l) < len(known) + 20:
                        score += 5
                        break

            # Regex pattern match
            for pattern in self._patterns:
                if pattern.match(line):
                    score += 4
                    break

            # Font size larger than median
            if size > median_size * 1.1:
                score += 3

            # Short line (headers are rarely long)
            if len(line) < 60:
                score += 2

            # Title case
            if line.istitle():
                score += 1

            # Ends with colon
            if line.endswith(":"):
                score += 1

            # Followed by blank line or short line (template structure)
            if i + 1 < len(normalized):
                next_line = normalized[i + 1].strip()
                if not next_line or len(next_line) < 10:
                    score += 1

            # Reject if it looks like body text
            if len(line) > 80:
                score -= 5
            if line.endswith(".") or line.endswith(","):
                score -= 3
            if line.count(" ") > 12:
                score -= 2

            if score >= 5:
                # Clean up the header name
                clean = line.rstrip(":").strip()
                clean = re.sub(r"^\d+[\.\)]\s*", "", clean)   # remove leading numbers
                clean = re.sub(r"^[A-Z][\.\)]\s*", "", clean) # remove leading letters
                clean = re.sub(r"^(?:I{1,3}|IV|V|VI{0,3}|IX|X)\.\s*", "", clean)
                clean = clean.strip()

                if clean and clean.lower() not in seen:
                    seen.add(clean.lower())
                    headers.append(clean)

        return headers

    def map_to_section_type(self, header: str) -> Optional[str]:
        """
        Map a detected header to a Phase 3 section_type for filtering.
        Returns None if no close match — falls back to unfiltered search.
        """
        MAPPING = {
            # Specific aims
            "specific aims": "specific_aims",
            "aims": "specific_aims",
            "goals and objectives": "specific_aims",
            "proposed solution": "specific_aims",
            # Significance
            "significance": "significance",
            "need": "significance",
            "community need": "significance",
            "problem statement": "significance",
            "background and significance": "significance",
            # Innovation
            "innovation": "innovation",
            "evidence base": "innovation",
            # Approach / methods
            "approach": "approach",
            "research strategy": "approach",
            "methods": "methods",
            "implementation plan": "methods",
            "work plan": "methods",
            # Background
            "background": "background",
            "literature review": "background",
            "preliminary studies": "background",
            "organizational background": "background",
            "organizational capacity": "background",
            # Project summary
            "project summary": "project_summary",
            "abstract": "project_summary",
            "executive summary": "project_summary",
            "project description": "project_summary",
            "project narrative": "project_summary",
        }
        h = header.lower().strip()
        # Exact match
        if h in MAPPING:
            return MAPPING[h]
        # Partial match
        for key, val in MAPPING.items():
            if key in h or h in key:
                return val
        return None  # No match — search without section filter


# ============ APPLICATION ASSISTANT ============

class ApplicationAssistant:
    """
    User-facing grant writing assistant.

    Takes a live GrantQueryPipeline instance — no separate setup needed.
    Detects sections from an empty application PDF, then retrieves
    relevant chunks from the user's own past grants for each section.
    """

    def __init__(self, pipeline, candidates_per_section: int = 3):
        """
        Args:
            pipeline:                Live GrantQueryPipeline (already set up)
            candidates_per_section:  How many chunk candidates to return per section
        """
        self.pipeline   = pipeline
        self.n          = candidates_per_section
        self.detector   = SectionDetector()
        self.user_grants: List[str] = []  # grant_ids belonging to this user

    # ── Public API ────────────────────────────────────────────────────────

    def set_user_grants(self, grant_ids: List[str]):
        """
        Register which grant IDs belong to the current user.
        These are the grants we'll retrieve from.
        """
        self.user_grants = [str(g) for g in grant_ids]
        print(f"  ✅ User corpus: {len(self.user_grants)} grants registered")
        for g in self.user_grants:
            print(f"     • {g}")

    def find_user_grants(self) -> List[str]:
        """
        Auto-detect which grants in Weaviate came from PDF ingestion
        (i.e. the user's uploaded grants, not NIH abstracts).
        Returns their grant_ids.
        """
        if not self.pipeline.weaviate.collection:
            print("  ❌ Pipeline not ready")
            return []
        try:
            from weaviate.classes.query import Filter
            # Fetch PDF-sourced chunks — these are the user's uploaded grants
            resp = self.pipeline.weaviate.collection.query.fetch_objects(
                limit=500,
                filters=Filter.by_property("chunkType").not_equal("abstract"),
                return_properties=["grantId", "chunkType"]
            )
            ids = {obj.properties.get("grantId", "")
                   for obj in resp.objects
                   if obj.properties.get("grantId", "")}
            # Exclude synthetic
            ids = {g for g in ids if not any(
                g.startswith(p) for p in ("FQHC_SYNTH_", "SYNTH_")
            )}
            self.user_grants = sorted(ids)
            print(f"  ✅ Auto-detected {len(self.user_grants)} user grants")
            for g in self.user_grants:
                print(f"     • {g}")
            return self.user_grants
        except Exception as e:
            print(f"  ⚠️  Auto-detect failed: {e}")
            return []

    def find_for_application(
        self,
        topic: str,
        pdf_path: str           = None,
        sections: List[str]     = None,
        user_grant_ids: List[str] = None,
        verbose: bool           = True,
    ) -> Dict[str, List[Dict]]:
        """
        Main entry point.

        For each section in the application, retrieves the top N most
        relevant chunks from the user's own past grants.

        Args:
            topic:          Description of the new application
                            e.g. "FQHC diabetes CHW program Latino patients"
            pdf_path:       Path to empty application PDF (sections auto-detected)
            sections:       Manual list of section names (if no PDF)
            user_grant_ids: Override user grants for this call
            verbose:        Print results to console

        Returns:
            Dict mapping section_name → list of ranked chunk candidates
        """
        if not self.pipeline._ready:
            print("❌ Pipeline not ready — call pipeline.setup() first")
            return {}

        # Resolve user grants
        grants = user_grant_ids or self.user_grants
        if not grants:
            print("⚠️  No user grants set — call set_user_grants() or find_user_grants() first")
            print("   Searching full corpus instead (results may include others' grants)")

        # Detect sections
        if pdf_path and os.path.exists(pdf_path):
            print(f"\n📄 Scanning {os.path.basename(pdf_path)} for sections...")
            detected = self.detector.detect_from_pdf(pdf_path)
        elif sections:
            detected = self.detector.detect_from_list(sections)
            print(f"\n📋 Using {len(detected)} provided sections")
        else:
            print("\n⚠️  No PDF or sections provided — using standard NIH sections")
            detected = [
                "Specific Aims", "Significance", "Innovation",
                "Approach", "Methods", "Background"
            ]

        if not detected:
            print("❌ No sections detected")
            return {}

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"📝 APPLICATION ASSISTANT")
            print(f"{'=' * 70}")
            print(f"   Topic:    {topic}")
            print(f"   Sections: {len(detected)}")
            print(f"   Corpus:   {len(grants)} user grants" if grants
                  else "   Corpus:   full system (no user filter)")
            print(f"   Per slot: {self.n} candidates")

        t       = time.time()
        results = {}

        for section in detected:
            section_type = self.detector.map_to_section_type(section)
            candidates   = self._retrieve_for_section(
                topic        = topic,
                section_name = section,
                section_type = section_type,
                grant_ids    = grants,
            )
            results[section] = candidates

            if verbose:
                self._print_section(section, candidates)

        elapsed = round(time.time() - t, 2)
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"✅ Done ({elapsed}s) — {len(detected)} sections, "
                  f"{sum(len(v) for v in results.values())} total candidates")
            print(f"{'=' * 70}")

        return results

    def describe_application(self, description: str, use_llm: bool = False,
                              api_key: str = None, provider: str = "anthropic") -> Dict:
        """
        Parse a free-text application description and run retrieval.

        Args:
            description: Natural language description of the grant application
            use_llm:     Use LLM-powered extraction (Option B) vs rule-based (Option A)
            api_key:     Optional API key (also reads from env ANTHROPIC_API_KEY / OPENAI_API_KEY)
            provider:    "anthropic" or "openai"

        Returns:
            Dict with keys "parsed" (extraction result) and "results" (retrieval results)
        """
        parser = ApplicationDescriptionParser()

        print(f"\n  🔎 Parsing application description...")
        if use_llm:
            parsed = parser.parse_llm(description, api_key=api_key, provider=provider)
        else:
            parsed = parser.parse_rule_based(description)

        parser.print_parsed(parsed)

        topic    = parsed["enriched_topic"]
        sections = parsed["sections"]

        results = self.find_for_application(
            topic    = topic,
            sections = sections,
        )
        return {"parsed": parsed, "results": results}

    def interactive(self):
        """Run interactive application assistant session."""
        print("\n" + "=" * 70)
        print("📝 GRANT APPLICATION ASSISTANT")
        print("=" * 70)
        print("Commands:")
        print("  /grants                    — show registered user grants")
        print("  /autodetect                — auto-detect user grants from Weaviate")
        print("  /setgrants <id1> <id2> ... — set user grants manually")
        print("  /pdf <path> <topic>        — run assistant on a PDF application")
        print("  /sections <topic>          — run with standard NIH sections")
        print("  /custom <s1>|<s2> <topic>  — run with custom sections (pipe-separated)")
        print("  /describe <description>    — [Option A] rule-based intent extraction + search")
        print("  /smart <description>       — [Option B] LLM-powered extraction + search")
        print("  /apikey <key>              — set API key for /smart (or set env var)")
        print("  /n <number>                — set candidates per section (default 3)")
        print("  /quit                      — exit\n")

        self._llm_api_key = None

        while True:
            try:
                ui = input("Assistant > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not ui: continue
            if ui.lower() in ["/quit", "/exit", "quit", "exit"]: break

            elif ui.lower() == "/grants":
                if self.user_grants:
                    print(f"\n  User grants ({len(self.user_grants)}):")
                    for g in self.user_grants:
                        print(f"    • {g}")
                else:
                    print("  No user grants set. Use /autodetect or /setgrants")

            elif ui.lower() == "/autodetect":
                self.find_user_grants()

            elif ui.startswith("/setgrants "):
                ids = ui[11:].strip().split()
                self.set_user_grants(ids)

            elif ui.startswith("/apikey "):
                self._llm_api_key = ui[8:].strip()
                print(f"  ✅ API key set ({self._llm_api_key[:8]}...)")

            elif ui.startswith("/n "):
                try:
                    self.n = int(ui[3:].strip())
                    print(f"  Candidates per section: {self.n}")
                except ValueError:
                    print("  Usage: /n 3")

            elif ui.startswith("/pdf "):
                parts = ui[5:].strip().split(" ", 1)
                if len(parts) == 2:
                    pdf_path, topic = parts
                    if os.path.exists(pdf_path):
                        self.find_for_application(topic=topic, pdf_path=pdf_path)
                    else:
                        print(f"  ❌ File not found: {pdf_path}")
                else:
                    print("  Usage: /pdf <path> <topic description>")

            elif ui.startswith("/sections "):
                topic = ui[10:].strip()
                self.find_for_application(topic=topic)

            elif ui.startswith("/custom "):
                rest = ui[8:].strip()
                if "|" in rest:
                    parts   = rest.split(" ", 1)
                    sec_str = parts[0]
                    topic   = parts[1] if len(parts) > 1 else ""
                    if not topic:
                        print("  Usage: /custom <s1>|<s2>|<s3> <topic>")
                        continue
                    sections = [s.strip() for s in sec_str.split("|")]
                    self.find_for_application(topic=topic, sections=sections)
                else:
                    print("  Usage: /custom <s1>|<s2>|<s3> <topic>")
                    print("  Example: /custom Specific Aims|Significance|Methods diabetes CHW FQHC")

            elif ui.startswith("/describe "):
                description = ui[10:].strip()
                if description:
                    self.describe_application(description, use_llm=False)
                else:
                    print("  Usage: /describe <natural language application description>")
                    print("  Example: /describe We are a FQHC in Chicago serving Latino patients")
                    print("           and want to expand diabetes prevention using CHWs and telehealth")

            elif ui.startswith("/smart "):
                description = ui[7:].strip()
                if description:
                    key = self._llm_api_key or \
                          os.environ.get("ANTHROPIC_API_KEY") or \
                          os.environ.get("OPENAI_API_KEY")
                    if not key:
                        print("  ⚠️  No API key found.")
                        print("  Set one with: /apikey <your-key>")
                        print("  Or set env var ANTHROPIC_API_KEY / OPENAI_API_KEY")
                        print("  Falling back to rule-based (/describe) instead...")
                        self.describe_application(description, use_llm=False)
                    else:
                        provider = "anthropic" if (
                            self._llm_api_key or os.environ.get("ANTHROPIC_API_KEY")
                        ) else "openai"
                        self.describe_application(
                            description, use_llm=True,
                            api_key=key, provider=provider
                        )
                else:
                    print("  Usage: /smart <natural language application description>")
                    print("  Requires ANTHROPIC_API_KEY or OPENAI_API_KEY (or /apikey <key>)")

            else:
                # Treat as free topic — run with standard sections
                print(f"  Running standard NIH sections for: {ui}")
                self.find_for_application(topic=ui)

    # ── Private helpers ───────────────────────────────────────────────────

    def _retrieve_for_section(
        self,
        topic:        str,
        section_name: str,
        section_type: Optional[str],
        grant_ids:    List[str],
    ) -> List[Dict]:
        """
        Retrieve top N candidates for a single section slot.

        Strategy:
        1. Build section-prefixed query for embedding alignment
        2. Run hybrid search with section_type filter (if mapped)
        3. If user grants specified, filter results to those grants
        4. If too few results with filter, retry without section filter
        """
        model = self.pipeline.model
        if not model:
            return []

        # Section-prefixed encoding matches how Phase 3 embedded chunks
        sec_label   = section_type.replace("_", " ") if section_type else section_name.lower()
        encode_text = f"{sec_label}: {topic}"
        query_vec   = model.encode(encode_text).tolist()

        # Try with section filter first
        results = self.pipeline.weaviate.hybrid_search(
            query        = topic,
            query_vector = query_vec,
            alpha        = self.pipeline.alpha,
            top_k        = 40,  # fetch more — quality filter will discard boilerplate
            section_filter = section_type,
        )

        # Filter to user's grants if specified
        if grant_ids:
            user_results = [r for r in results
                            if r.get("grant_id") in grant_ids]

            # If too few, retry without section filter to broaden
            if len(user_results) < self.n and section_type:
                broader = self.pipeline.weaviate.hybrid_search(
                    query        = topic,
                    query_vector = model.encode(topic).tolist(),
                    alpha        = self.pipeline.alpha,
                    top_k        = 20,
                    section_filter = None,
                )
                broader_user = [r for r in broader
                                if r.get("grant_id") in grant_ids
                                and r not in user_results]
                user_results = (user_results + broader_user)[:self.n * 2]

            results = user_results

        # ── Content quality filter ────────────────────────────────────────────
        # Remove boilerplate / table-of-contents chunks that match section names
        # but contain no scientific content.
        def _is_substantive(r: Dict) -> bool:
            text = r.get("text", "")
            words = text.split()
            # Too short to be useful
            if len(words) < 40:
                return False
            # TOC pattern: numbered lines like "3. Research Strategy\n4. Progress..."
            numbered_lines = sum(
                1 for line in text.splitlines()
                if line.strip() and line.strip()[0].isdigit() and ". " in line[:10]
            )
            if numbered_lines >= 3:
                return False
            # Pure form fields: mostly short lines with no sentences
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines:
                avg_line_len = sum(len(l) for l in lines) / len(lines)
                if avg_line_len < 30 and len(lines) > 5:
                    return False
            return True

        results = [r for r in results if _is_substantive(r)]

        # Deduplicate by grant_id + section_type, keep highest score
        seen_keys = {}
        for r in results:
            key = (r.get("grant_id"), r.get("section_type"))
            if key not in seen_keys or r["score"] > seen_keys[key]["score"]:
                seen_keys[key] = r

        final = sorted(seen_keys.values(), key=lambda x: x["score"], reverse=True)
        return final[:self.n]

    def _print_section(self, section_name: str, candidates: List[Dict]):
        """Pretty print candidates for one section."""
        print(f"\n{'─' * 70}")
        print(f"📌 {section_name.upper()}")
        print(f"{'─' * 70}")

        if not candidates:
            print("  ⚠️  No candidates found")
            return

        for i, c in enumerate(candidates, 1):
            gid   = c.get("grant_id", "N/A")
            score = c.get("score", 0)
            sec   = c.get("section_type", "")
            title = c.get("title", "")[:70]
            text  = c.get("text", "")[:250].strip()
            year  = c.get("year", "")

            print(f"\n  [{i}] Score: {score:.4f}  |  {gid}  ({year})")
            if title: print(f"      {title}")
            if sec:   print(f"      from: {sec} section")
            if text:
                # Indent the text block
                wrapped = "\n      ".join(
                    text[j:j+80] for j in range(0, len(text), 80)
                )
                print(f"\n      \"{wrapped}\"")


# ============ APPLICATION DESCRIPTION PARSER ============

class ApplicationDescriptionParser:
    """
    Extracts structured grant intent from a free-text application description.

    Two modes:
      A) Rule-based  — fast, no API key, works offline
      B) LLM-powered — richer, handles messy/implicit descriptions

    Output schema:
        {
          "conditions":    ["diabetes", "hypertension"],
          "populations":   ["Latino", "pediatric", "low-income"],
          "interventions": ["CHW", "telehealth"],
          "settings":      ["FQHC", "community health center"],
          "funder_type":   "NIH" | "HRSA" | "SAMHSA" | "foundation" | "unknown",
          "grant_type":    "R01" | "R21" | "K" | "HRSA" | "unknown",
          "is_fqhc":       True/False,
          "enriched_topic": "<flattened keyword string for retrieval>",
          "sections":       ["Specific Aims", ...],   # suggested sections
          "summary":        "<one-line human-readable summary>"
        }
    """

    # ── Rule-based vocabulary ──────────────────────────────────────────────

    CONDITION_MAP = {
        "diabetes":           ["diabetes", "diabetic", "glycemic", "hba1c", "insulin"],
        "hypertension":       ["hypertension", "blood pressure", "cardiovascular", "heart disease", "stroke"],
        "cancer":             ["cancer", "oncology", "screening", "mammogram", "colonoscopy"],
        "behavioral_health":  ["mental health", "depression", "anxiety", "psychiatric", "behavioral health",
                               "substance use", "opioid", "addiction", "alcohol", "suds"],
        "HIV":                ["hiv", "aids", "prep", "antiretroviral"],
        "obesity":            ["obesity", "overweight", "bmi", "weight loss", "nutrition"],
        "asthma":             ["asthma", "copd", "respiratory", "lung"],
        "maternal_health":    ["maternal", "prenatal", "pregnancy", "postpartum", "infant mortality"],
        "pediatric":          ["pediatric", "child", "children", "adolescent", "youth", "school-based"],
        "social_determinants":["housing", "food insecurity", "transportation", "social determinants", "sdoh"],
    }

    POPULATION_MAP = {
        "Latino":      ["latino", "hispanic", "spanish", "promotora", "spanish-speaking"],
        "African American": ["african american", "black", "african-american"],
        "low_income":  ["low income", "low-income", "poverty", "medicaid", "uninsured", "underserved", "vulnerable"],
        "rural":       ["rural", "frontier", "remote", "underserved rural"],
        "elderly":     ["elderly", "older adult", "aging", "geriatric", "senior"],
        "pediatric":   ["child", "pediatric", "youth", "adolescent", "teen"],
        "immigrant":   ["immigrant", "refugee", "undocumented", "limited english"],
        "LGBTQ":       ["lgbtq", "transgender", "sexual minority", "msm"],
    }

    INTERVENTION_MAP = {
        "CHW":           ["community health worker", "chw", "promotora", "lay health", "patient navigator"],
        "telehealth":    ["telehealth", "telemedicine", "digital health", "remote monitoring", "mhealth", "app"],
        "EHR":           ["ehr", "electronic health record", "clinical decision support", "health it"],
        "care_management":["care management", "care coordination", "case management", "disease management"],
        "screening":     ["screening", "early detection", "preventive", "prevention program"],
        "education":     ["health education", "literacy", "training", "curriculum", "workshop"],
        "peer_support":  ["peer support", "peer counselor", "lived experience", "recovery coach"],
    }

    SETTING_MAP = {
        "FQHC":          ["fqhc", "federally qualified", "community health center", "chc", "health center"],
        "hospital":      ["hospital", "inpatient", "clinical", "medical center"],
        "school":        ["school", "classroom", "school-based", "education setting"],
        "community":     ["community", "neighborhood", "faith-based", "church", "community organization"],
        "pharmacy":      ["pharmacy", "pharmacist", "medication management"],
        "home":          ["home visit", "home-based", "in-home"],
    }

    FUNDER_SECTIONS = {
        "NIH":   ["Specific Aims", "Significance", "Innovation", "Approach", "Methods", "Background"],
        "HRSA":  ["Need", "Proposed Solution", "Organizational Capacity",
                  "Work Plan", "Evaluation Plan", "Budget Narrative"],
        "SAMHSA":["Project Description", "Goals and Objectives", "Evidence Base",
                  "Implementation Plan", "Staff Qualifications", "Evaluation"],
        "foundation": ["Executive Summary", "Needs Statement", "Project Description",
                       "Evaluation", "Budget Narrative", "Sustainability"],
    }

    FUNDER_SIGNALS = {
        "NIH":        ["r01", "r21", "k01", "k23", "r03", "nih", "national institutes", "nimh", "nida", "nhlbi"],
        "HRSA":       ["hrsa", "health resources", "fqhc", "bureau of health", "bureau of primary care",
                       "look-alike", "section 330"],
        "SAMHSA":     ["samhsa", "substance abuse", "mental health services administration", "sow"],
        "foundation": ["foundation", "trust", "award", "rfp", "grant opportunity", "robert wood johnson",
                       "kellogg", "commonwealth"],
    }

    # ── Public API ─────────────────────────────────────────────────────────

    def parse_rule_based(self, description: str) -> Dict:
        """
        Option A: Rule-based extraction. Fast, no API needed.
        Returns structured dict with detected fields + enriched_topic.
        """
        text = description.lower()

        conditions    = self._match(text, self.CONDITION_MAP)
        populations   = self._match(text, self.POPULATION_MAP)
        interventions = self._match(text, self.INTERVENTION_MAP)
        settings      = self._match(text, self.SETTING_MAP)

        funder_type = self._detect_funder(text)
        grant_type  = self._detect_grant_type(text)
        is_fqhc     = "FQHC" in settings or any(
            kw in text for kw in ["fqhc", "federally qualified", "community health center", "chc"]
        )

        # Build enriched topic: original description + key extracted terms
        extras = conditions + populations + interventions + settings
        enriched = description.strip()
        if extras:
            enriched += " " + " ".join(extras)

        sections = self.FUNDER_SECTIONS.get(funder_type,
                   self.FUNDER_SECTIONS["NIH"])

        summary = self._build_summary(conditions, populations, interventions,
                                      settings, funder_type, is_fqhc)

        return {
            "conditions":    conditions,
            "populations":   populations,
            "interventions": interventions,
            "settings":      settings,
            "funder_type":   funder_type,
            "grant_type":    grant_type,
            "is_fqhc":       is_fqhc,
            "enriched_topic": enriched,
            "sections":      sections,
            "summary":       summary,
            "method":        "rule-based",
        }

    def parse_llm(self, description: str, api_key: str = None,
                  provider: str = "anthropic") -> Dict:
        """
        Option B: LLM-powered extraction.
        provider: "anthropic" or "openai"
        Falls back to rule-based if API call fails.
        """
        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or \
                         os.environ.get("OPENAI_API_KEY")
        if not key:
            print("  ⚠️  No API key found — falling back to rule-based parsing")
            return self.parse_rule_based(description)

        prompt = f"""You are a grant writing analyst. Extract structured information from this grant application description.

Application description:
\"\"\"{description}\"\"\"

Return ONLY a JSON object with these exact keys (no markdown, no explanation):
{{
  "conditions": ["list of medical/health conditions mentioned"],
  "populations": ["target populations e.g. Latino, pediatric, low-income, rural"],
  "interventions": ["interventions e.g. CHW, telehealth, screening, care management"],
  "settings": ["care settings e.g. FQHC, community health center, school, hospital"],
  "funder_type": "NIH or HRSA or SAMHSA or foundation or unknown",
  "grant_type": "R01 or R21 or K or HRSA or foundation or unknown",
  "is_fqhc": true or false,
  "sections": ["suggested NIH/HRSA section names appropriate for this application type"],
  "summary": "one sentence describing this grant application",
  "enriched_topic": "a rich keyword string for semantic search combining all extracted fields"
}}"""

        try:
            if provider == "anthropic" or "ANTHROPIC_API_KEY" in os.environ:
                result = self._call_anthropic(prompt,
                    api_key or os.environ.get("ANTHROPIC_API_KEY"))
            else:
                result = self._call_openai(prompt,
                    api_key or os.environ.get("OPENAI_API_KEY"))

            result["method"] = "llm"

            # Fill any missing keys with rule-based fallback
            rb = self.parse_rule_based(description)
            for k in ["conditions", "populations", "interventions", "settings",
                      "funder_type", "grant_type", "is_fqhc", "sections",
                      "enriched_topic", "summary"]:
                if k not in result or not result[k]:
                    result[k] = rb[k]
            return result

        except Exception as e:
            print(f"  ⚠️  LLM extraction failed ({e}) — falling back to rule-based")
            return self.parse_rule_based(description)

    def print_parsed(self, parsed: Dict):
        """Pretty-print the parsed application intent."""
        method = parsed.get("method", "?")
        print(f"\n  {'─'*60}")
        print(f"  🔍 Application Analysis  [{method}]")
        print(f"  {'─'*60}")
        if parsed.get("summary"):
            print(f"  Summary:       {parsed['summary']}")
        if parsed.get("conditions"):
            print(f"  Conditions:    {', '.join(parsed['conditions'])}")
        if parsed.get("populations"):
            print(f"  Populations:   {', '.join(parsed['populations'])}")
        if parsed.get("interventions"):
            print(f"  Interventions: {', '.join(parsed['interventions'])}")
        if parsed.get("settings"):
            print(f"  Settings:      {', '.join(parsed['settings'])}")
        print(f"  Funder type:   {parsed.get('funder_type','?')}  "
              f"| Grant type: {parsed.get('grant_type','?')}  "
              f"| FQHC: {'✅' if parsed.get('is_fqhc') else '❌'}")
        print(f"  Sections:      {', '.join(parsed.get('sections',[]))}")
        print(f"  Search query:  \"{parsed.get('enriched_topic','')}\"")
        print(f"  {'─'*60}\n")

    # ── Private helpers ────────────────────────────────────────────────────

    def _match(self, text: str, vocab: Dict) -> List[str]:
        matched = []
        for label, keywords in vocab.items():
            if any(kw in text for kw in keywords):
                matched.append(label)
        return matched

    def _detect_funder(self, text: str) -> str:
        for funder, signals in self.FUNDER_SIGNALS.items():
            if any(s in text for s in signals):
                return funder
        return "NIH"  # default assumption

    def _detect_grant_type(self, text: str) -> str:
        for gtype in ["r01", "r21", "r03", "k01", "k23", "k99"]:
            if gtype in text:
                return gtype.upper()
        if any(s in text for s in ["hrsa", "fqhc", "bureau"]):
            return "HRSA"
        return "unknown"

    def _build_summary(self, conditions, populations, interventions,
                       settings, funder_type, is_fqhc) -> str:
        parts = []
        if settings:   parts.append(f"{'/'.join(settings)} setting")
        if conditions: parts.append(f"{'/'.join(conditions[:2])} focused")
        if populations:parts.append(f"serving {'/'.join(populations[:2])}")
        if interventions: parts.append(f"using {'/'.join(interventions[:2])}")
        if is_fqhc:    parts.append("FQHC-eligible")
        parts.append(f"[{funder_type}]")
        return ", ".join(parts) if parts else "General health grant"

    def _call_anthropic(self, prompt: str, api_key: str) -> Dict:
        import urllib.request
        data = json.dumps({
            "model":      "claude-haiku-4-5-20251001",
            "max_tokens": 800,
            "messages":   [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data,
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        text = body["content"][0]["text"].strip()
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    def _call_openai(self, prompt: str, api_key: str) -> Dict:
        import urllib.request
        data = json.dumps({
            "model":      "gpt-4o-mini",
            "max_tokens": 800,
            "messages":   [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type":  "application/json",
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        text = body["choices"][0]["message"]["content"].strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)


# ============ STANDALONE RUNNER ============

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Grant Application Assistant — retrieve relevant chunks from user's grants"
    )
    parser.add_argument("--topic",    type=str, help="New application topic description")
    parser.add_argument("--pdf",      type=str, help="Path to empty application PDF")
    parser.add_argument("--sections", type=str, help="Comma-separated section names")
    parser.add_argument("--grants",   type=str, help="Comma-separated user grant IDs")
    parser.add_argument("--n",        type=int, default=3, help="Candidates per section")
    parser.add_argument("--autodetect", action="store_true",
                        help="Auto-detect user grants from Weaviate")
    args = parser.parse_args()

    # Boot pipeline
    from query_pipeline import GrantQueryPipeline
    pipeline = GrantQueryPipeline()
    if not pipeline.setup():
        print("❌ Pipeline setup failed"); return

    assistant = ApplicationAssistant(pipeline, candidates_per_section=args.n)

    # Set user grants
    if args.grants:
        assistant.set_user_grants(args.grants.split(","))
    elif args.autodetect:
        assistant.find_user_grants()

    # Run or go interactive
    try:
        if args.topic:
            sections = args.sections.split(",") if args.sections else None
            assistant.find_for_application(
                topic    = args.topic,
                pdf_path = args.pdf,
                sections = sections,
            )
        else:
            assistant.interactive()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()