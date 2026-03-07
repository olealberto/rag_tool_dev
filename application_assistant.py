# ============================================================================
# application_assistant.py - GRANT APPLICATION ASSISTANT
# ============================================================================

"""
USER-FACING GRANT WRITING ASSISTANT

Given a free-text RFP description, the assistant:
  1. Parses intent (conditions, populations, interventions, funder type)
  2. Runs corpus-level discovery via query_pipeline.discover_grants()
  3. Merges discovered grants with any manually registered user grants
     (user grants take priority; corpus fills remaining slots)
  4. Retrieves top N chunks per section from the merged grant set
     with source_url linking to the original PDF or NIH Reporter page

Usage in Colab:
    from query_pipeline import GrantQueryPipeline
    from application_assistant import ApplicationAssistant

    pipeline = GrantQueryPipeline()
    pipeline.setup()

    assistant = ApplicationAssistant(pipeline)
    assistant.describe_application(
        "FQHC in Chicago serving Latino patients, expanding diabetes "
        "prevention using CHWs and telehealth, HRSA funding"
    )
"""

import os
import re
import json
import time
from typing import List, Dict, Optional
from collections import defaultdict


# ============ SECTION DETECTOR ============

class SectionDetector:
    KNOWN_HEADERS = [
        "specific aims", "significance", "innovation", "approach",
        "background", "project summary", "project narrative",
        "research strategy", "methods", "preliminary studies",
        "human subjects", "vertebrate animals", "bibliography",
        "resource sharing", "authentication of key resources",
        "need", "community need", "problem statement", "target population",
        "proposed solution", "work plan", "evaluation plan",
        "organizational capacity", "budget narrative", "budget justification",
        "sustainability", "organizational background",
        "project description", "goals and objectives", "logic model",
        "evidence base", "implementation plan", "staff qualifications",
        "intellectual merit", "broader impacts", "facilities",
        "data management", "postdoctoral mentoring",
        "introduction", "executive summary", "abstract",
        "literature review", "theoretical framework",
        "timeline", "dissemination", "references",
    ]

    HEADER_PATTERNS = [
        r"^[A-Z][A-Z\s\-/&]{4,}$",
        r"^\d+[\.\)]\s+[A-Z][A-Za-z\s]{3,}$",
        r"^[A-Z][\.\)]\s+[A-Z][A-Za-z\s]{3,}$",
        r"^(?:I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+",
        r"^Section\s+\d+",
        r"^Part\s+[A-Z\d]",
    ]

    def __init__(self):
        self._known_lower = {h.lower() for h in self.KNOWN_HEADERS}
        self._patterns    = [re.compile(p) for p in self.HEADER_PATTERNS]

    def detect_from_pdf(self, pdf_path: str) -> List[str]:
        text_lines = self._extract_lines(pdf_path)
        if not text_lines:
            print(f"  \u26a0\ufe0f  Could not extract text from {pdf_path}")
            return []
        headers = self._detect_headers(text_lines)
        print(f"  \u2705 Detected {len(headers)} sections in {os.path.basename(pdf_path)}")
        for h in headers:
            print(f"     \u2022 {h}")
        return headers

    def detect_from_list(self, sections: List[str]) -> List[str]:
        return [s.strip() for s in sections if s.strip()]

    def _extract_lines(self, pdf_path: str) -> List:
        lines = []
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    try:
                        words = page.extract_words(extra_attrs=["size", "fontname"])
                        if words:
                            by_y = defaultdict(list)
                            for w in words:
                                y_key = round(float(w.get("top", 0)) / 3) * 3
                                by_y[y_key].append(w)
                            for y in sorted(by_y.keys()):
                                lw = sorted(by_y[y], key=lambda w: w.get("x0", 0))
                                lt = " ".join(w["text"] for w in lw).strip()
                                lines.append((lt, float(lw[0].get("size", 10))))
                            continue
                    except Exception:
                        pass
                    for line in (page.extract_text() or "").split("\n"):
                        lines.append((line.strip(), 10.0))
            return lines
        except ImportError:
            pass
        try:
            from pypdf import PdfReader
            for page in PdfReader(pdf_path).pages:
                for line in (page.extract_text() or "").split("\n"):
                    lines.append((line.strip(), 10.0))
            return lines
        except ImportError:
            pass
        try:
            with open(pdf_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    lines.append((line.strip(), 10.0))
        except Exception:
            pass
        return lines

    def _detect_headers(self, lines) -> List[str]:
        normalized, sizes = [], []
        for item in lines:
            if isinstance(item, tuple):
                normalized.append(item[0]); sizes.append(item[1])
            else:
                normalized.append(str(item)); sizes.append(10.0)

        sorted_s    = sorted(sizes)
        median_size = sorted_s[len(sorted_s) // 2] if sorted_s else 10.0
        headers, seen = [], set()

        for i, (line, size) in enumerate(zip(normalized, sizes)):
            if not line or len(line) < 3: continue
            score  = 0
            line_l = line.lower().strip().rstrip(":")

            if line_l in self._known_lower:
                score += 10
            else:
                for known in self._known_lower:
                    if known in line_l and len(line_l) < len(known) + 20:
                        score += 5; break

            for pattern in self._patterns:
                if pattern.match(line): score += 4; break

            if size > median_size * 1.1: score += 3
            if len(line) < 60:           score += 2
            if line.istitle():           score += 1
            if line.endswith(":"):       score += 1
            if i + 1 < len(normalized):
                nl = normalized[i + 1].strip()
                if not nl or len(nl) < 10: score += 1
            if len(line) > 80:                           score -= 5
            if line.endswith(".") or line.endswith(","): score -= 3
            if line.count(" ") > 12:                     score -= 2

            if score >= 5:
                clean = line.rstrip(":").strip()
                clean = re.sub(r"^\d+[\.\)]\s*", "", clean)
                clean = re.sub(r"^[A-Z][\.\)]\s*", "", clean)
                clean = re.sub(r"^(?:I{1,3}|IV|V|VI{0,3}|IX|X)\.\s*", "", clean)
                clean = clean.strip()
                if clean and clean.lower() not in seen:
                    seen.add(clean.lower()); headers.append(clean)
        return headers

    def map_to_section_type(self, header: str) -> Optional[str]:
        MAPPING = {
            "specific aims": "specific_aims", "aims": "specific_aims",
            "goals and objectives": "specific_aims", "proposed solution": "specific_aims",
            "significance": "significance", "need": "significance",
            "community need": "significance", "problem statement": "significance",
            "background and significance": "significance",
            "innovation": "innovation", "evidence base": "innovation",
            "approach": "approach", "research strategy": "approach",
            "methods": "methods", "implementation plan": "methods", "work plan": "methods",
            "background": "background", "literature review": "background",
            "preliminary studies": "background", "organizational background": "background",
            "organizational capacity": "background",
            "project summary": "project_summary", "abstract": "project_summary",
            "executive summary": "project_summary", "project description": "project_summary",
            "project narrative": "project_summary",
        }
        h = header.lower().strip()
        if h in MAPPING: return MAPPING[h]
        for key, val in MAPPING.items():
            if key in h or h in key: return val
        return None


# ============ APPLICATION ASSISTANT ============

class ApplicationAssistant:
    """
    User-facing grant writing assistant.

    describe_application() workflow:
      1. Parse intent (rule-based or LLM)
      2. Corpus discovery — pipeline.discover_grants() finds top relevant grants
      3. Merge: user_grants first (priority), corpus fills remaining slots
      4. Section retrieval with source_url on every chunk
    """

    CORPUS_DISCOVERY_SLOTS = 8

    def __init__(self, pipeline, candidates_per_section: int = 3):
        self.pipeline    = pipeline
        self.n           = candidates_per_section
        self.detector    = SectionDetector()
        self.user_grants: List[str] = []

    def set_user_grants(self, grant_ids: List[str]):
        self.user_grants = [str(g) for g in grant_ids]
        print(f"  \u2705 User corpus: {len(self.user_grants)} grants registered")
        for g in self.user_grants:
            print(f"     \u2022 {g}")

    def find_user_grants(self) -> List[str]:
        if not self.pipeline.weaviate.collection:
            print("  \u274c Pipeline not ready"); return []
        try:
            from weaviate.classes.query import Filter
            resp = self.pipeline.weaviate.collection.query.fetch_objects(
                limit=500,
                filters=Filter.by_property("chunkType").not_equal("abstract"),
                return_properties=["grantId", "chunkType"]
            )
            ids = {obj.properties.get("grantId", "")
                   for obj in resp.objects if obj.properties.get("grantId", "")}
            ids = {g for g in ids if not any(g.startswith(p) for p in ("FQHC_SYNTH_", "SYNTH_"))}
            self.user_grants = sorted(ids)
            print(f"  \u2705 Auto-detected {len(self.user_grants)} user grants")
            for g in self.user_grants:
                print(f"     \u2022 {g}")
            return self.user_grants
        except Exception as e:
            print(f"  \u26a0\ufe0f  Auto-detect failed: {e}"); return []

    def describe_application(self, description: str, use_llm: bool = False,
                              api_key: str = None, provider: str = "anthropic") -> Dict:
        """
        Parse description → discover grants → retrieve section chunks.
        """
        parser = ApplicationDescriptionParser()

        print(f"\n  \U0001f50e Parsing application description...")
        parsed = parser.parse_llm(description, api_key=api_key, provider=provider) \
                 if use_llm else parser.parse_rule_based(description)
        parser.print_parsed(parsed)

        topic       = parsed["enriched_topic"]
        sections    = parsed["sections"]
        funder_type = parsed.get("funder_type", "NIH")
        is_fqhc     = parsed.get("is_fqhc", False)

        # ── corpus discovery ──────────────────────────────────────────────
        print(f"  \U0001f50d Discovering relevant grants from corpus (hybrid + graph)...")
        discovered = self.pipeline.discover_grants(
            topic     = topic,
            top_k     = self.CORPUS_DISCOVERY_SLOTS + len(self.user_grants),
            fqhc_only = is_fqhc,
            parsed    = parsed,
        )

        # ── merge: user grants first, corpus fills remaining slots ────────
        user_set   = set(self.user_grants)
        corpus_new = [g for g in discovered if g not in user_set]
        merged     = self.user_grants + corpus_new[:self.CORPUS_DISCOVERY_SLOTS]

        if merged:
            # Retrieve source tags from pipeline if available
            discovery_sources = getattr(self.pipeline, "_last_discovery_sources", {})
            n_corpus = len(corpus_new[:self.CORPUS_DISCOVERY_SLOTS])
            print(f"  \U0001f4da Searching {len(merged)} grants "
                  f"({len(self.user_grants)} user + {n_corpus} discovered)")
            for g in merged[:6]:
                if g in user_set:
                    tag = " [user]"
                elif discovery_sources.get(g) == "graph":
                    tag = " [graph]"
                else:
                    tag = " [hybrid]"
                print(f"     \u2022 {g}{tag}")
            if len(merged) > 6:
                print(f"     ... and {len(merged) - 6} more")

        results = self.find_for_application(
            topic          = topic,
            sections       = sections,
            user_grant_ids = merged if merged else None,
            funder_type    = funder_type,
        )
        return {"parsed": parsed, "results": results}

    def find_for_application(
        self,
        topic: str,
        pdf_path: str             = None,
        sections: List[str]       = None,
        user_grant_ids: List[str] = None,
        funder_type: str          = "NIH",
        verbose: bool             = True,
    ) -> Dict[str, List[Dict]]:
        if not self.pipeline._ready:
            print("\u274c Pipeline not ready"); return {}

        grants = user_grant_ids or self.user_grants

        if pdf_path and os.path.exists(pdf_path):
            print(f"\n\U0001f4c4 Scanning {os.path.basename(pdf_path)} for sections...")
            detected = self.detector.detect_from_pdf(pdf_path)
        elif sections:
            detected = self.detector.detect_from_list(sections)
            print(f"\n\U0001f4cb Using {len(detected)} provided sections")
        else:
            print("\n\u26a0\ufe0f  No sections provided — using standard NIH sections")
            detected = ["Specific Aims", "Significance", "Innovation",
                        "Approach", "Methods", "Background"]

        if not detected:
            print("\u274c No sections detected"); return {}

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"\U0001f4dd APPLICATION ASSISTANT")
            print(f"{'=' * 70}")
            print(f"   Topic:    {topic[:120]}{'...' if len(topic) > 120 else ''}")
            print(f"   Sections: {len(detected)}  |  Funder: {funder_type}")
            print(f"   Corpus:   {len(grants)} grants" if grants
                  else "   Corpus:   full system (no grant filter)")
            print(f"   Per slot: {self.n} candidates")

        t, results, used_chunks = time.time(), {}, set()

        for section in detected:
            section_type = self.detector.map_to_section_type(section)
            candidates   = self._retrieve_for_section(
                topic        = topic,
                section_name = section,
                section_type = section_type,
                grant_ids    = grants,
                funder_type  = funder_type,
                used_chunks  = used_chunks,
            )
            results[section] = candidates

            for c in candidates:
                used_chunks.add((c.get("grant_id"),
                                 c.get("chunk_index", c.get("section_type", ""))))
            if verbose:
                self._print_section(section, candidates)

        elapsed = round(time.time() - t, 2)
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"\u2705 Done ({elapsed}s) — {len(detected)} sections, "
                  f"{sum(len(v) for v in results.values())} total candidates")
            print(f"{'=' * 70}")

        return results

    def interactive(self):
        print("\n" + "=" * 70)
        print("\U0001f4dd GRANT APPLICATION ASSISTANT")
        print("=" * 70)
        print("  /grants                    — show registered user grants")
        print("  /autodetect                — auto-detect user grants from Weaviate")
        print("  /setgrants <id1> <id2> ... — set user grants manually")
        print("  /pdf <path> <topic>        — run on a PDF application")
        print("  /sections <topic>          — standard NIH sections")
        print("  /custom <s1>|<s2> <topic>  — custom sections (pipe-separated)")
        print("  /describe <description>    — rule-based intent + corpus discovery + chunks")
        print("  /smart <description>       — LLM intent + corpus discovery + chunks")
        print("  /apikey <key>              — set API key for /smart")
        print("  /n <number>                — candidates per section (default 3)")
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
                    for g in self.user_grants: print(f"    \u2022 {g}")
                else:
                    print("  No user grants set. Use /autodetect or /setgrants")

            elif ui.lower() == "/autodetect":
                self.find_user_grants()

            elif ui.startswith("/setgrants "):
                self.set_user_grants(ui[11:].strip().split())

            elif ui.startswith("/apikey "):
                self._llm_api_key = ui[8:].strip()
                print(f"  \u2705 API key set ({self._llm_api_key[:8]}...)")

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
                        print(f"  \u274c File not found: {pdf_path}")
                else:
                    print("  Usage: /pdf <path> <topic description>")

            elif ui.startswith("/sections "):
                self.find_for_application(topic=ui[10:].strip())

            elif ui.startswith("/custom "):
                rest = ui[8:].strip()
                if "|" in rest:
                    parts = rest.split(" ", 1)
                    if len(parts) == 2:
                        self.find_for_application(
                            topic    = parts[1],
                            sections = [s.strip() for s in parts[0].split("|")]
                        )
                    else:
                        print("  Usage: /custom <s1>|<s2>|<s3> <topic>")
                else:
                    print("  Usage: /custom <s1>|<s2>|<s3> <topic>")

            elif ui.startswith("/describe "):
                description = ui[10:].strip()
                if description:
                    self.describe_application(description, use_llm=False)
                else:
                    print("  Usage: /describe <natural language description>")

            elif ui.startswith("/smart "):
                description = ui[7:].strip()
                if description:
                    key = self._llm_api_key or \
                          os.environ.get("ANTHROPIC_API_KEY") or \
                          os.environ.get("OPENAI_API_KEY")
                    if not key:
                        print("  \u26a0\ufe0f  No API key — falling back to rule-based")
                        self.describe_application(description, use_llm=False)
                    else:
                        provider = "anthropic" if (
                            self._llm_api_key or os.environ.get("ANTHROPIC_API_KEY")
                        ) else "openai"
                        self.describe_application(description, use_llm=True,
                                                   api_key=key, provider=provider)
                else:
                    print("  Usage: /smart <description>")

            else:
                print(f"  Running standard NIH sections for: {ui}")
                self.find_for_application(topic=ui)

    # ── Private helpers ───────────────────────────────────────────────────

    def _retrieve_for_section(
        self,
        topic:        str,
        section_name: str,
        section_type: Optional[str],
        grant_ids:    List[str],
        funder_type:  str = "NIH",
        used_chunks:  set = None,
    ) -> List[Dict]:
        model = self.pipeline.model
        if not model: return []

        sec_label   = section_type.replace("_", " ") if section_type else section_name.lower()
        encode_text = f"{sec_label}: {topic}"
        query_vec   = model.encode(encode_text).tolist()

        use_section_filter = (funder_type == "NIH") and (section_type is not None)

        results = self.pipeline.weaviate.hybrid_search(
            query           = f"{section_name}: {topic}",
            query_vector    = query_vec,
            alpha           = self.pipeline.alpha,
            top_k           = 60,
            section_filter  = section_type if use_section_filter else None,
            grant_id_filter = grant_ids if grant_ids else None,
        )

        if used_chunks:
            results = [r for r in results
                       if (r.get("grant_id"),
                           r.get("chunk_index", r.get("section_type", "")))
                       not in used_chunks]

        # User grants bubble to top within results
        if grant_ids and self.user_grants:
            user_set = set(self.user_grants)
            results  = ([r for r in results if r.get("grant_id") in user_set] +
                        [r for r in results if r.get("grant_id") not in user_set])

        # Fallback: broaden if too few results
        if len(results) < self.n and use_section_filter:
            broader = self.pipeline.weaviate.hybrid_search(
                query           = topic,
                query_vector    = model.encode(topic).tolist(),
                alpha           = self.pipeline.alpha,
                top_k           = 20,
                section_filter  = None,
                grant_id_filter = grant_ids if grant_ids else None,
            )
            seen_ids = {(r.get("grant_id"), r.get("chunk_index")) for r in results}
            results += [r for r in broader
                        if (r.get("grant_id"), r.get("chunk_index")) not in seen_ids]

        # Remove boilerplate / TOC chunks
        def _is_substantive(r: Dict) -> bool:
            text  = r.get("text", "")
            words = text.split()
            if len(words) < 40: return False
            numbered = sum(1 for l in text.splitlines()
                           if l.strip() and l.strip()[0].isdigit() and ". " in l[:10])
            if numbered >= 3: return False
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines and sum(len(l) for l in lines) / len(lines) < 30 and len(lines) > 5:
                return False
            return True

        results = [r for r in results if _is_substantive(r)]

        # Deduplicate by (grant_id, chunk_index)
        seen_chunks: Dict = {}
        for r in results:
            key = (r.get("grant_id"), r.get("chunk_index", r.get("section_type", "")))
            if key not in seen_chunks or r["score"] > seen_chunks[key]["score"]:
                seen_chunks[key] = r

        return sorted(seen_chunks.values(), key=lambda x: x["score"], reverse=True)[:self.n]

    def _print_section(self, section_name: str, candidates: List[Dict]):
        print(f"\n{chr(0x2500) * 70}")
        print(f"\U0001f4cc {section_name.upper()}")
        print(f"{chr(0x2500) * 70}")

        if not candidates:
            print("  \u26a0\ufe0f  No candidates found"); return

        for i, c in enumerate(candidates, 1):
            gid   = c.get("grant_id", "N/A")
            score = c.get("score", 0)
            sec   = c.get("section_type", "")
            title = c.get("title", "")[:70]
            text  = c.get("text", "")[:250].strip()
            year  = c.get("year", "")
            cidx  = c.get("chunk_index", "")
            url   = c.get("source_url", "")

            print(f"\n  [{i}] Score: {score:.4f}  |  {gid}  ({year})")
            if title: print(f"      {title}")
            if sec and cidx != "": print(f"      from: {sec} section, chunk #{cidx}")
            elif sec:              print(f"      from: {sec} section")
            if url:                print(f"      \U0001f517 {url}")
            if text:
                wrapped = "\n      ".join(text[j:j+80] for j in range(0, len(text), 80))
                print(f"\n      \"{wrapped}\"")


# ============ APPLICATION DESCRIPTION PARSER ============

class ApplicationDescriptionParser:
    """
    Extracts structured grant intent from a free-text application description.

    Funder detection fix:
    - "nofo" and "notice of funding opportunity" removed from CDC signals
      (these are used by all federal agencies, not CDC-specific)
    - NIH content-based fallback added: etiology, pathogenesis, clinical trial,
      disease mechanism, investigative team all point to NIH before defaulting
    """

    CONDITION_MAP = {
        "diabetes":           ["diabetes", "diabetic", "glycemic", "hba1c", "insulin",
                               "prediabetes", "type 2 diabetes", "type 1 diabetes"],
        "hypertension":       ["hypertension", "blood pressure", "cardiovascular",
                               "heart disease", "stroke", "cardiac", "hypertensive"],
        "cancer":             ["cancer", "oncology", "tumor", "mammogram", "colonoscopy",
                               "cervical", "breast cancer", "colorectal"],
        "HIV":                ["hiv", "aids", "prep", "antiretroviral", "art", "viral load",
                               "hiv prevention", "hiv treatment", "viral hepatitis",
                               "hepatitis c", "hepatitis b", "hcv", "hbv"],
        "substance_use":      ["substance use", "opioid", "addiction", "alcohol use",
                               "suds", "overdose", "fentanyl", "naloxone", "narcan",
                               "harm reduction", "syringe", "needle exchange",
                               "medication assisted treatment", "mat", "buprenorphine",
                               "methadone", "recovery", "drug use"],
        "behavioral_health":  ["mental health", "depression", "anxiety", "psychiatric",
                               "behavioral health", "trauma", "ptsd", "suicide",
                               "crisis intervention", "psychosis", "bipolar",
                               "co-occurring", "dual diagnosis"],
        "obesity":            ["obesity", "overweight", "bmi", "weight loss", "nutrition",
                               "physical activity", "obesity prevention"],
        "asthma":             ["asthma", "copd", "respiratory", "lung disease",
                               "pulmonary", "inhaler"],
        "maternal_health":    ["maternal", "prenatal", "pregnancy", "postpartum",
                               "birth outcomes", "infant mortality", "neonatal",
                               "obstetric", "perinatal"],
        "pediatric":          ["pediatric", "child health", "adolescent health",
                               "childhood", "youth health", "school health"],
        "infectious_disease": ["infectious disease", "sexually transmitted", "sti", "std",
                               "tuberculosis", "tb", "covid", "pandemic", "epidemic",
                               "syndemic", "communicable disease"],
        "chronic_disease":    ["chronic disease", "chronic condition", "multiple chronic",
                               "comorbidity", "disease management", "chronic care"],
        "oral_health":        ["oral health", "dental", "dentist", "tooth decay", "cavity"],
        "violence":           ["violence", "domestic violence", "gun violence", "homicide",
                               "trauma-informed", "adverse childhood", "aces",
                               "sexual assault", "intimate partner"],
    }

    POPULATION_MAP = {
        "Latino":             ["latino", "hispanic", "latinx", "spanish-speaking",
                               "promotora", "spanish language", "mexican", "puerto rican"],
        "African_American":   ["african american", "black", "african-american",
                               "black community", "racial equity", "life expectancy gap"],
        "low_income":         ["low income", "low-income", "poverty", "medicaid",
                               "uninsured", "underserved", "vulnerable population",
                               "economically disadvantaged", "200% fpl", "below poverty"],
        "rural":              ["rural", "frontier", "remote", "underserved rural"],
        "elderly":            ["elderly", "older adult", "aging", "geriatric",
                               "senior", "65 and older"],
        "pediatric":          ["children", "pediatric", "youth", "adolescent",
                               "teen", "young adult", "school-age"],
        "immigrant":          ["immigrant", "refugee", "undocumented", "limited english",
                               "lep", "language access", "newcomer", "asylum"],
        "LGBTQ":              ["lgbtq", "transgender", "trans", "sexual minority",
                               "msm", "gender nonconforming", "queer"],
        "homeless":           ["homeless", "unhoused", "housing instability",
                               "shelter resident", "transitional housing"],
        "justice_involved":   ["incarcerated", "reentry", "formerly incarcerated",
                               "justice-involved", "correctional", "probation", "parole"],
        "racial_ethnic_minority": ["racial", "ethnic minority", "minority population",
                                   "communities of color", "health equity", "health disparities"],
    }

    INTERVENTION_MAP = {
        "CHW":               ["community health worker", "chw", "promotora",
                              "lay health advisor", "patient navigator", "community advocate",
                              "outreach worker", "health educator", "peer navigator"],
        "telehealth":        ["telehealth", "telemedicine", "digital health",
                              "remote monitoring", "mhealth", "mobile app",
                              "text message", "sms", "virtual visit"],
        "harm_reduction":    ["harm reduction", "syringe services", "needle exchange",
                              "naloxone distribution", "overdose prevention"],
        "care_management":   ["care management", "care coordination", "case management",
                              "disease management", "care navigator", "integrated care"],
        "screening":         ["screening", "early detection", "preventive care",
                              "prevention program", "health screening", "testing"],
        "education":         ["health education", "health literacy", "training",
                              "curriculum", "workshop", "community education",
                              "awareness campaign", "outreach"],
        "peer_support":      ["peer support", "peer specialist", "peer counselor",
                              "lived experience", "recovery coach", "peer recovery"],
        "policy_advocacy":   ["policy", "advocacy", "systems change", "upstream",
                              "structural", "legislative", "coalition", "systemic"],
        "direct_services":   ["direct service", "direct care", "clinical services",
                              "primary care", "supportive services", "wraparound",
                              "comprehensive services", "social services"],
        "data_surveillance": ["surveillance", "data collection", "epidemiology",
                              "monitoring", "evaluation", "metrics", "dashboard"],
        "capacity_building": ["capacity building", "organizational capacity",
                              "workforce development", "technical assistance",
                              "coalition building", "partnership"],
    }

    SETTING_MAP = {
        "FQHC":               ["fqhc", "federally qualified", "community health center",
                               "chc", "look-alike", "section 330", "health center"],
        "hospital":           ["hospital", "inpatient", "medical center", "health system",
                               "clinical setting", "emergency department"],
        "school":             ["school", "classroom", "school-based", "after school",
                               "district", "k-12", "head start"],
        "community_org":      ["community organization", "nonprofit", "community-based",
                               "cbo", "faith-based", "church", "neighborhood org",
                               "grassroots", "community group"],
        "public_health_dept": ["health department", "public health", "cdph", "dph",
                               "city health", "county health", "state health",
                               "local health authority", "lhd"],
        "shelter_transitional":["shelter", "transitional housing", "housing program",
                                "supportive housing", "group home"],
        "home":               ["home visit", "home-based", "in-home", "home care"],
        "street_outreach":    ["street outreach", "mobile unit", "outreach team",
                               "field outreach", "van", "mobile health"],
    }

    FUNDER_SECTIONS = {
        "NIH": [
            "Specific Aims", "Significance", "Innovation",
            "Approach", "Methods", "Background",
        ],
        "HRSA": [
            "Need", "Proposed Solution", "Organizational Capacity",
            "Work Plan", "Evaluation Plan", "Budget Narrative",
        ],
        "SAMHSA": [
            "Project Description", "Goals and Objectives", "Evidence Base",
            "Implementation Plan", "Staff Qualifications", "Evaluation",
        ],
        "CDC": [
            "Background", "Statement of Need", "Approach",
            "Evaluation and Performance Measurement", "Organizational Capacity",
            "Work Plan",
        ],
        "city_public_health": [
            "Background and Community Need", "Program Description",
            "Target Population", "Implementation Plan", "Partnerships",
            "Evaluation Plan", "Budget Narrative", "Sustainability",
        ],
        "SAMHSA_block": [
            "Problem Statement", "Proposed Services", "Target Population",
            "Evidence-Based Practices", "Collaboration and Coordination",
            "Evaluation", "Budget",
        ],
        "foundation": [
            "Executive Summary", "Needs Statement", "Project Description",
            "Goals and Objectives", "Evaluation", "Budget Narrative",
            "Organizational Background", "Sustainability",
        ],
        "federal_other": [
            "Executive Summary", "Statement of Need", "Project Description",
            "Approach and Methods", "Evaluation", "Organizational Capacity",
            "Budget Narrative",
        ],
    }

    FUNDER_SIGNALS = {
        "NIH": [
            "nih", "national institutes", "nimh", "nida", "nhlbi",
            "nci", "nichd", "niaid", "niddk", "nibib", "ninds",
            "grant number", "study section",
            # NIH grant types need word-boundary matching (handled in _detect_funder)
            "r01", "r21", "k01", "k23", "r03", "r34", "u01",
        ],
        "HRSA": [
            "hrsa", "health resources and services", "fqhc",
            "bureau of primary care", "look-alike", "section 330",
            "bureau of health workforce", "ryan white",
        ],
        "SAMHSA": [
            "samhsa", "substance abuse and mental health",
            "mental health services administration",
            "block grant", "sow", "scope of work",
            "opioid response", "state opioid", "harm reduction grant",
            "recovery support", "treatment funding",
        ],
        "CDC": [
            # NOTE: "nofo" and "notice of funding opportunity" intentionally
            # removed — these are used by all federal agencies, not CDC-specific.
            # CDC-specific signals only:
            "cdc", "centers for disease control",
            "cooperative agreement", "ps ",
        ],
        "city_public_health": [
            "cdph", "chicago department of public health",
            "department of public health", "city health department",
            "county health department", "local health department",
            "healthy chicago", "rfp", "rfa",
            "life expectancy gap", "syndemic",
        ],
        "SAMHSA_block": [
            "block grant", "mhbg", "sabg", "prevention set-aside",
        ],
        "foundation": [
            "foundation", "charitable trust", "endowment",
            "robert wood johnson", "kellogg", "commonwealth fund",
            "mccormick", "macarthur", "open society", "united way",
            "community foundation", "philanthrop",
        ],
        "federal_other": [
            "department of justice", "hud", "doj", "ojjdp",
            "office of juvenile", "acf", "acl", "cms",
            "administration for children", "federal grant",
        ],
    }

    NONPROFIT_SIGNALS = [
        "501(c)(3)", "nonprofit", "non-profit", "community organization",
        "community-based organization", "cbo", "coalition", "collaborative",
        "partnership", "consortium", "memorandum of understanding", "mou",
        "subcontract", "subgrantee", "lead agency", "fiscal sponsor",
        "service area", "target population", "scope of services",
        "logic model", "theory of change", "racial equity", "health equity",
        "social justice", "community engagement", "community voice",
        "lived experience", "trusted messenger", "cultural competency",
        "linguistically appropriate", "culturally appropriate",
    ]

    def parse_rule_based(self, description: str) -> Dict:
        text = description.lower()
        conditions    = self._match(text, self.CONDITION_MAP)
        populations   = self._match(text, self.POPULATION_MAP)
        interventions = self._match(text, self.INTERVENTION_MAP)
        settings      = self._match(text, self.SETTING_MAP)
        funder_type   = self._detect_funder(text)
        grant_type    = self._detect_grant_type(text)
        is_fqhc       = "FQHC" in settings or any(
            kw in text for kw in ["fqhc", "federally qualified", "community health center", "chc"]
        )
        is_nonprofit  = any(kw in text for kw in self.NONPROFIT_SIGNALS)
        is_equity     = any(kw in text for kw in [
            "health equity", "health disparities", "racial equity", "life expectancy gap",
            "health fairness", "equitable", "disparity", "structural racism",
        ])

        enriched_parts = [description.strip()]
        extras = conditions + populations + interventions + settings
        if extras: enriched_parts.append(" ".join(extras))

        # ── Funder-aware vocabulary enrichment ────────────────────────────
        # Clinical specificity terms are prepended for NIH queries because
        # BM25 rewards rare, specific terms — generic words like "prevention"
        # are too common across the corpus to discriminate well.
        # Community/equity terms are prepended for HRSA/FQHC/nonprofit queries
        # because those grants don't use clinical terminology but do use
        # population health and program delivery language.
        # Both layers are included when context is mixed (e.g. FQHC + diabetes).

        # Clinical specificity layer — condition-specific vocabulary that
        # rarely appears outside genuinely relevant grants
        CLINICAL_VOCAB = {
            "diabetes":          "glycemic HbA1c prediabetes insulin type 2 diabetes glucose",
            "hypertension":      "blood pressure systolic diastolic antihypertensive cardiovascular",
            "cancer":            "oncology tumor screening mammogram colonoscopy chemotherapy",
            "HIV":               "antiretroviral viral load PrEP CD4 HIV transmission",
            "substance_use":     "opioid buprenorphine methadone naloxone overdose SUD MAT",
            "behavioral_health": "PHQ-9 GAD-7 psychiatric depression anxiety PTSD crisis",
            "obesity":           "BMI weight loss lifestyle intervention physical activity caloric",
            "asthma":            "inhaler spirometry bronchodilator pulmonary FEV1 COPD",
            "maternal_health":   "prenatal postpartum obstetric birth outcomes neonatal perinatal",
            "pediatric":         "child development growth adolescent school-age immunization",
            "infectious_disease":"pathogen surveillance transmission epidemiology incidence",
            "chronic_disease":   "disease management comorbidity care coordination self-management",
            "oral_health":       "dental caries periodontal fluoride oral hygiene",
            "violence":          "trauma-informed ACEs intimate partner safety screening",
        }

        # Community/program layer — vocabulary for FQHC, nonprofit, and
        # community-based grants that don't use clinical terminology
        COMMUNITY_VOCAB = {
            "CHW":             "community health worker promotora outreach lay health advisor",
            "telehealth":      "remote monitoring virtual visit mHealth digital health platform",
            "care_management": "care coordination case management integrated care navigator",
            "screening":       "early detection preventive care health screening testing outreach",
            "education":       "health literacy curriculum workshop community education awareness",
            "peer_support":    "peer specialist lived experience recovery coach peer counselor",
            "direct_services": "direct care wraparound comprehensive services social services",
            "capacity_building":"workforce development technical assistance coalition partnership",
        }

        # Add clinical specificity for matched conditions (always helpful)
        for cond in conditions:
            if cond in CLINICAL_VOCAB:
                enriched_parts.append(CLINICAL_VOCAB[cond])

        # Add community/program vocabulary for matched interventions
        for intervention in interventions:
            if intervention in COMMUNITY_VOCAB:
                enriched_parts.append(COMMUNITY_VOCAB[intervention])

        # Funder-specific framing
        if funder_type == "NIH":
            # NIH: emphasize research design and clinical outcomes
            enriched_parts.append("research design study protocol clinical outcomes measurement")
            if conditions:
                enriched_parts.append("disease mechanism pathophysiology biomarker evidence-based")
        elif funder_type in ("HRSA", "SAMHSA") or is_fqhc:
            # HRSA/FQHC: emphasize access, underserved populations, service delivery
            enriched_parts.append(
                "federally qualified health center community health center "
                "underserved primary care access safety-net medically underserved"
            )
            if populations:
                enriched_parts.append("health disparities health equity vulnerable population")
        elif funder_type == "CDC":
            # CDC: emphasize surveillance, population-level impact, public health
            enriched_parts.append(
                "surveillance epidemiology population health public health "
                "disease burden incidence prevalence community-level"
            )
        elif funder_type in ("foundation", "city_public_health"):
            # Foundation/city: emphasize community impact, equity, program outcomes
            enriched_parts.append(
                "community-based organization nonprofit direct services "
                "program outcomes logic model sustainability equity"
            )

        # Cross-cutting equity layer — added for any funder when equity is present
        if is_equity:
            enriched_parts.append(
                "health disparities racial equity underserved communities "
                "social determinants structural barriers"
            )

        # Nonprofit/CBO layer
        if is_nonprofit:
            enriched_parts.append(
                "community-based organization direct services target population "
                "program participants organizational capacity"
            )

        enriched = " ".join(enriched_parts)

        sections = self.FUNDER_SECTIONS.get(funder_type, self.FUNDER_SECTIONS["foundation"])
        summary  = self._build_summary(conditions, populations, interventions,
                                       settings, funder_type, is_fqhc, is_equity)
        return {
            "conditions": conditions, "populations": populations,
            "interventions": interventions, "settings": settings,
            "funder_type": funder_type, "grant_type": grant_type,
            "is_fqhc": is_fqhc, "is_nonprofit": is_nonprofit, "is_equity": is_equity,
            "enriched_topic": enriched, "sections": sections,
            "summary": summary, "method": "rule-based",
        }

    def parse_llm(self, description: str, api_key: str = None,
                  provider: str = "anthropic") -> Dict:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or \
                         os.environ.get("OPENAI_API_KEY")
        if not key:
            print("  \u26a0\ufe0f  No API key — falling back to rule-based")
            return self.parse_rule_based(description)

        prompt = f"""You are a grant writing analyst. Extract structured information from this grant application description.

Application description:
\"\"\"{description}\"\"\"

Return ONLY a JSON object (no markdown, no explanation):
{{
  "conditions": ["health conditions addressed"],
  "populations": ["target populations served"],
  "interventions": ["program interventions or methods"],
  "settings": ["care or program settings"],
  "funder_type": "NIH|HRSA|SAMHSA|CDC|city_public_health|foundation|federal_other|unknown",
  "grant_type": "R01|R21|U01|HRSA|RFP|cooperative agreement|city/county grant|unknown",
  "is_fqhc": true/false,
  "is_equity": true/false,
  "is_nonprofit": true/false,
  "sections": ["the actual application sections to WRITE — e.g. Specific Aims, Significance, Innovation, Approach for NIH; Need, Proposed Solution, Work Plan for HRSA; Background, Statement of Need, Approach for CDC — NOT document type names like NOFO or Notice"],
  "summary": "one sentence describing the grant application",
  "enriched_topic": "rich keyword string for semantic search — include conditions, populations, interventions, setting"
}}"""

        try:
            if provider == "anthropic" or os.environ.get("ANTHROPIC_API_KEY"):
                result = self._call_anthropic(prompt, key)
            else:
                result = self._call_openai(prompt, key)

            result["method"] = "llm"
            rb = self.parse_rule_based(description)
            for k in ["conditions", "populations", "interventions", "settings",
                      "funder_type", "grant_type", "is_fqhc",
                      "enriched_topic", "summary"]:
                if k not in result or not result[k]: result[k] = rb[k]

            # Always validate sections — fall back to rule-based if LLM returned
            # document type names instead of application section names
            BAD_SECTION_TOKENS = {"nofo", "notice", "funding opportunity", "announcement",
                                  "solicitation", "request for application", "rfa"}
            sections = result.get("sections", [])
            if not sections or any(
                any(bad in s.lower() for bad in BAD_SECTION_TOKENS) for s in sections
            ):
                result["sections"] = rb["sections"]

            return result
        except Exception as e:
            print(f"  \u26a0\ufe0f  LLM failed ({e}) — falling back to rule-based")
            return self.parse_rule_based(description)

    def print_parsed(self, parsed: Dict):
        method = parsed.get("method", "?")
        print(f"\n  {chr(0x2500)*62}")
        print(f"  \U0001f50d Application Analysis  [{method}]")
        print(f"  {chr(0x2500)*62}")
        if parsed.get("summary"):       print(f"  Summary:       {parsed['summary']}")
        if parsed.get("conditions"):    print(f"  Conditions:    {', '.join(parsed['conditions'])}")
        if parsed.get("populations"):   print(f"  Populations:   {', '.join(parsed['populations'])}")
        if parsed.get("interventions"): print(f"  Interventions: {', '.join(parsed['interventions'])}")
        if parsed.get("settings"):      print(f"  Settings:      {', '.join(parsed['settings'])}")
        flags = []
        if parsed.get("is_fqhc"):      flags.append("FQHC \u2705")
        if parsed.get("is_equity"):    flags.append("Health Equity \u2705")
        if parsed.get("is_nonprofit"): flags.append("Nonprofit/CBO \u2705")
        print(f"  Funder:        {parsed.get('funder_type','?')}  "
              f"| Grant type: {parsed.get('grant_type','?')}"
              + (f"  | {' '.join(flags)}" if flags else ""))
        print(f"  Sections:      {', '.join(parsed.get('sections',[]))}")
        et = parsed.get("enriched_topic", "")
        print(f"  Search query:  \"{et[:200]}{'...' if len(et) > 200 else ''}\"")
        print(f"  {chr(0x2500)*62}\n")

    def _match(self, text: str, vocab: Dict) -> List[str]:
        return [label for label, kws in vocab.items() if any(kw in text for kw in kws)]

    def _detect_funder(self, text: str) -> str:
        # Short acronyms need word boundaries to avoid false matches
        SHORT = {"r01","r21","r03","r34","u01","k01","k23","k99",
                 "nci","nida","nimh","nhlbi","nichd","niaid","niddk",
                 "cdc","hud","doj","acf","acl","cms","sow","mou","rfp","rfa","bmi",
                 "ps"}

        def _any(signals):
            for s in signals:
                if s in SHORT:
                    if re.search(r'\b' + re.escape(s) + r'\b', text): return True
                else:
                    if s in text: return True
            return False

        # Check explicit funder signals — order matters (most specific first)
        for funder in ["NIH", "HRSA", "SAMHSA", "CDC",
                       "city_public_health", "SAMHSA_block", "federal_other", "foundation"]:
            if _any(self.FUNDER_SIGNALS.get(funder, [])): return funder

        # ── Content-based fallbacks ───────────────────────────────────────
        # NIH-specific research vocabulary — check before generic fallbacks
        # "nofo" alone is not enough; look for research/mechanism language
        if any(s in text for s in [
            "etiology", "pathogenesis", "disease mechanism", "investigative team",
            "clinical trial", "randomized", "cohort study", "basic science",
            "translational", "r01", "r21", "study section", "specific aims",
        ]):
            return "NIH"

        if any(s in text for s in ["syringe", "naloxone", "buprenorphine", "methadone",
                                   "overdose prevention"]): return "SAMHSA"
        if any(s in text for s in ["mental health", "psychiatric",
                                   "crisis intervention"]): return "SAMHSA"
        if any(s in text for s in ["community health center", "fqhc",
                                   "primary care"]): return "HRSA"
        if any(s in text for s in ["request for proposal", "scope of services",
                                   "501(c)(3)", "nonprofit"]): return "foundation"

        # Final fallback: if "nofo" or "notice of funding opportunity" present
        # but no other signals, assume NIH (most common federal research funder)
        if any(s in text for s in ["nofo", "notice of funding opportunity",
                                   "funding opportunity"]):
            return "NIH"

        return "NIH"

    def _detect_grant_type(self, text: str) -> str:
        for g in ["r01", "r21", "r03", "r34", "u01", "k01", "k23", "k99"]:
            if g in text: return g.upper()
        if any(s in text for s in ["hrsa", "section 330"]):         return "HRSA"
        if any(s in text for s in ["rfp", "request for proposal"]): return "RFP"
        if any(s in text for s in ["cooperative agreement"]):        return "cooperative agreement"
        if any(s in text for s in ["nofo", "notice of funding"]):    return "NIH NOFO"
        if any(s in text for s in ["cdph", "city health", "county health"]): return "city/county grant"
        return "unknown"

    def _build_summary(self, conditions, populations, interventions,
                       settings, funder_type, is_fqhc, is_equity=False) -> str:
        parts = []
        if settings:      parts.append(f"{'/'.join(settings[:2])} setting")
        if conditions:    parts.append(f"{'/'.join(conditions[:2])} focused")
        if populations:   parts.append(f"serving {'/'.join(populations[:2])}")
        if interventions: parts.append(f"via {'/'.join(interventions[:2])}")
        if is_equity:     parts.append("health equity focus")
        if is_fqhc:       parts.append("FQHC-eligible")
        parts.append(f"[{funder_type}]")
        return ", ".join(parts) if parts else "General health grant"

    def _call_anthropic(self, prompt: str, api_key: str) -> Dict:
        import urllib.request
        data = json.dumps({
            "model": "claude-haiku-4-5-20251001", "max_tokens": 800,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=data,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        text = re.sub(r"^```(?:json)?\s*", "", body["content"][0]["text"].strip())
        return json.loads(re.sub(r"\s*```$", "", text))

    def _call_openai(self, prompt: str, api_key: str) -> Dict:
        import urllib.request
        data = json.dumps({
            "model": "gpt-4o-mini", "max_tokens": 800,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=data,
            headers={"Authorization": f"Bearer {api_key}",
                     "content-type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        text = re.sub(r"^```(?:json)?\s*", "", body["choices"][0]["message"]["content"].strip())
        return json.loads(re.sub(r"\s*```$", "", text))


# ============ STANDALONE RUNNER ============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic",      type=str)
    parser.add_argument("--pdf",        type=str)
    parser.add_argument("--sections",   type=str)
    parser.add_argument("--grants",     type=str)
    parser.add_argument("--n",          type=int, default=3)
    parser.add_argument("--autodetect", action="store_true")
    args = parser.parse_args()

    from query_pipeline import GrantQueryPipeline
    pipeline = GrantQueryPipeline()
    if not pipeline.setup():
        print("\u274c Pipeline setup failed"); return

    assistant = ApplicationAssistant(pipeline, candidates_per_section=args.n)
    if args.grants:       assistant.set_user_grants(args.grants.split(","))
    elif args.autodetect: assistant.find_user_grants()

    try:
        if args.topic:
            sections = args.sections.split(",") if args.sections else None
            assistant.find_for_application(topic=args.topic, pdf_path=args.pdf,
                                           sections=sections)
        else:
            assistant.interactive()
    except KeyboardInterrupt:
        print("\n\n\u26a0\ufe0f  Interrupted")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()