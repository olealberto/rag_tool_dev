# ============================================================================
# 📁 production_pipeline.py - PDF EXTRACTION + PRE-PROCESSING + CHUNKING
# ============================================================================

print("="*70)
print("🚀 PRODUCTION PIPELINE: PDF → CLEAN TEXT → SEMANTIC CHUNKS")
print("="*70)

import sys
import os
import re
import json
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

# ============ PDF EXTRACTION ============

class PDFExtractor:
    """Extract text from PDFs with semantic preservation"""
    
    def __init__(self):
        # Try multiple PDF libraries
        self.pdf_libraries = self._detect_pdf_libs()
    
    def extract_with_semantics(self, pdf_path: str) -> Dict:
        """
        Extract text while preserving semantic structure
        Returns: {'text': clean_text, 'metadata': {...}, 'sections': [...]}
        """
        print(f"📄 Extracting from: {os.path.basename(pdf_path)}")
        
        # Try libraries in order of preference
        for lib_name, extract_func in self.pdf_libraries:
            try:
                print(f"  Trying {lib_name}...")
                result = extract_func(pdf_path)
                
                if result and len(result.get('text', '')) > 100:
                    print(f"✅ Success with {lib_name}")
                    return result
            except Exception as e:
                print(f"  ❌ {lib_name} failed: {str(e)[:50]}")
                continue
        
        print("⚠️  All PDF libraries failed")
        return {'text': '', 'metadata': {}, 'sections': []}
    
    def _detect_pdf_libs(self) -> List[Tuple]:
        """Detect available PDF libraries"""
        libraries = []
        
        # 1. Try PyPDF2 (fast, common)
        try:
            import PyPDF2
            libraries.append(('PyPDF2', self._extract_with_pypdf2))
        except:
            pass
        
        # 2. Try pdfplumber (better for tables/formats)
        try:
            import pdfplumber
            libraries.append(('pdfplumber', self._extract_with_pdfplumber))
        except:
            pass
        
        # 3. Try pdfminer (most accurate)
        try:
            from pdfminer.high_level import extract_text
            libraries.append(('pdfminer', self._extract_with_pdfminer))
        except:
            pass
        
        # 4. Try pymupdf (very fast)
        try:
            import fitz  # PyMuPDF
            libraries.append(('PyMuPDF', self._extract_with_pymupdf))
        except:
            pass
        
        print(f"📚 Available PDF libraries: {[l[0] for l in libraries]}")
        return libraries
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Dict:
        """Extract using PyPDF2"""
        import PyPDF2
        
        text = ""
        metadata = {}
        
        with open(pdf_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            metadata = pdf.metadata
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        return self._post_process_text(text, metadata)
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> Dict:
        """Extract using pdfplumber (preserves layout)"""
        import pdfplumber
        
        text = ""
        sections = []
        
        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    # Extract text with coordinates (for semantic analysis)
                    chars = page.chars
                    
                    # Group by y-position to detect paragraphs
                    lines = page.extract_text_lines()
                    
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        return self._post_process_text(text, metadata)
    
    def _extract_with_pdfminer(self, pdf_path: str) -> Dict:
        """Extract using pdfminer (most accurate)"""
        from pdfminer.high_level import extract_text
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument
        
        # Extract text
        text = extract_text(pdf_path)
        
        # Extract metadata
        metadata = {}
        with open(pdf_path, 'rb') as file:
            parser = PDFParser(file)
            doc = PDFDocument(parser)
            metadata = doc.info[0] if doc.info else {}
        
        return self._post_process_text(text, metadata)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict:
        """Extract using PyMuPDF (very fast)"""
        import fitz
        
        text = ""
        metadata = {}
        
        with fitz.open(pdf_path) as doc:
            metadata = doc.metadata
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        return self._post_process_text(text, metadata)
    
    def _post_process_text(self, text: str, metadata: dict) -> Dict:
        """Clean and structure extracted text"""
        if not text:
            return {'text': '', 'metadata': metadata, 'sections': []}
        
        # 1. Basic cleaning
        cleaned = self._clean_text(text)
        
        # 2. Detect sections (for semantic chunking)
        sections = self._detect_sections(cleaned)
        
        # 3. Extract document type
        doc_type = self._classify_document(cleaned)
        
        return {
            'text': cleaned,
            'metadata': {**metadata, 'doc_type': doc_type},
            'sections': sections,
            'char_count': len(cleaned),
            'word_count': len(cleaned.split()),
            'extraction_time': datetime.now().isoformat()
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text while preserving semantics"""
        # Remove excessive whitespace but keep paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix common OCR/PDF issues
        text = re.sub(r'ﬁ', 'fi', text)
        text = re.sub(r'ﬂ', 'fl', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'-\s*\d+\s*-', '', text)  # Page numbers like "- 1 -"
        text = re.sub(r'\n\d+\n', '\n', text)    # Standalone page numbers
        
        # Standardize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _detect_sections(self, text: str) -> List[Dict]:
        """Detect document sections for semantic chunking"""
        sections = []
        
        # Common grant document sections
        section_patterns = {
            'title': r'^[A-Z][A-Z\s\-\:,]{5,100}$',
            'abstract': r'(?:PROJECT\s+)?(?:SUMMARY/)?ABSTRACT\s*[:]?',
            'specific_aims': r'SPECIFIC\s+AIMS?\s*[:]?',
            'background': r'BACKGROUND\s*(?:AND\s+SIGNIFICANCE)?\s*[:]?',
            'methods': r'(?:RESEARCH\s+)?(?:DESIGN\s+AND\s+)?METHODS?\s*[:]?',
            'results': r'RESULTS?\s*[:]?',
            'discussion': r'DISCUSSION\s*[:]?',
            'references': r'REFERENCES?\s*[:]?',
            'budget': r'BUDGET\s*(?:JUSTIFICATION)?\s*[:]?',
        }
        
        lines = text.split('\n')
        current_section = 'header'
        section_text = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if line starts a new section
            section_found = False
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # Save previous section
                    if section_text:
                        sections.append({
                            'name': current_section,
                            'text': '\n'.join(section_text),
                            'start_line': len(sections) + 1
                        })
                    
                    # Start new section
                    current_section = section_name
                    section_text = [line_stripped]
                    section_found = True
                    break
            
            if not section_found and line_stripped:
                section_text.append(line_stripped)
        
        # Add final section
        if section_text:
            sections.append({
                'name': current_section,
                'text': '\n'.join(section_text),
                'start_line': len(sections) + 1
            })
        
        return sections
    
    def _classify_document(self, text: str) -> str:
        """Classify document type for semantic chunking strategy"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['r01', 'r21', 'r34', 'nih', 'grant']):
            return 'nih_grant'
        elif any(word in text_lower for word in ['rfp', 'request for proposal', 'solicitation']):
            return 'rfp'
        elif any(word in text_lower for word in ['abstract', 'summary', 'background']):
            return 'research_abstract'
        elif any(word in text_lower for word in ['protocol', 'trial', 'clinical']):
            return 'study_protocol'
        else:
            return 'general_document'

# ============ INTELLIGENT PRE-PROCESSOR ============

class IntelligentPreprocessor:
    """
    PRE-PROCESS TEXT FOR BETTER SEMANTIC CHUNKING
    Cleans text while preserving semantic meaning
    """
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self.stopwords = self._load_stopwords()
    
    def preprocess_for_chunking(self, text: str, doc_type: str = None) -> str:
        """
        Clean text for optimal semantic chunking
        Returns: Cleaned text ready for chunking
        """
        print(f"🧹 Pre-processing {len(text):,} chars for {doc_type or 'general'} document")
        
        # 1. Document-specific cleaning
        cleaned = self._document_specific_clean(text, doc_type)
        
        # 2. Semantic preservation steps
        cleaned = self._preserve_semantic_structure(cleaned)
        
        # 3. Remove noise but keep meaning
        cleaned = self._remove_noise_preserve_meaning(cleaned)
        
        print(f"✅ Cleaned to {len(cleaned):,} chars ({len(cleaned)/max(len(text),1):.1%} of original)")
        return cleaned
    
    def _document_specific_clean(self, text: str, doc_type: str) -> str:
        """Apply document-specific cleaning"""
        if doc_type == 'nih_grant':
            return self._clean_nih_grant(text)
        elif doc_type == 'rfp':
            return self._clean_rfp(text)
        elif doc_type == 'research_abstract':
            return self._clean_abstract(text)
        else:
            return self._clean_general(text)
    
    def _clean_nih_grant(self, text: str) -> str:
        """Clean NIH grant documents"""
        # Remove common NIH boilerplate
        patterns_to_remove = [
            r'NOT-[A-Z]+-\d+-\d+',  # NIH notice numbers
            r'RFA-[A-Z]+-\d+-\d+',  # RFA numbers
            r'PA-\d+-\d+',          # Program announcement numbers
            r'Page \d+ of \d+',     # Page numbers
            r'OMB No\. \d+-\d+',    # OMB numbers
            r'Expiration Date:.*',  # Expiration dates
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def _clean_rfp(self, text: str) -> str:
        """Clean RFP documents"""
        # Extract requirements section
        requirements_section = self._extract_requirements_section(text)
        if requirements_section:
            return requirements_section
        
        # Fallback: keep entire text but clean
        return self._clean_general(text)
    
    def _clean_abstract(self, text: str) -> str:
        """Clean research abstracts"""
        # Remove section headers
        text = re.sub(r'^(ABSTRACT|SUMMARY|BACKGROUND)\s*[:]?\s*', '', text, flags=re.IGNORECASE)
        
        # Remove author lists and affiliations
        text = re.sub(r'^[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+,? (?:et al\.?,? )?', '', text)
        
        return text.strip()
    
    def _clean_general(self, text: str) -> str:
        """General text cleaning"""
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
        
        return text
    
    def _preserve_semantic_structure(self, text: str) -> str:
        """Preserve semantic structure for chunking"""
        # Preserve paragraph breaks (semantic boundaries)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Preserve section headers (capitalized lines)
        # Don't lowercase headers - they indicate semantic boundaries
        
        # Preserve list structure
        text = re.sub(r'^\s*[\-\*•]\s+', '• ', text, flags=re.MULTILINE)
        
        # Preserve numbered lists
        text = re.sub(r'^\s*\d+\.\s+', '1. ', text, flags=re.MULTILINE)
        
        return text
    
    def _remove_noise_preserve_meaning(self, text: str) -> str:
        """Remove noise while preserving semantic meaning"""
        # Remove excessive whitespace but keep sentence boundaries
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common OCR errors that affect meaning
        corrections = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            '’': "'",
            '``': '"',
            "''": '"'
        }
        
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        
        # Remove non-semantic characters but keep punctuation
        text = re.sub(r'[^\w\s.,;:!?\'"()\[\]{}<>-]', ' ', text)
        
        return text.strip()
    
    def _extract_requirements_section(self, text: str) -> Optional[str]:
        """Extract requirements section from RFP"""
        # Look for requirements section
        patterns = [
            r'REQUIREMENTS?.*?(?=DELIVERABLES|QUALIFICATIONS|SUBMISSION|$)',
            r'SCOPE OF WORK.*?(?=DELIVERABLES|QUALIFICATIONS|SUBMISSION|$)',
            r'STATEMENT OF WORK.*?(?=DELIVERABLES|QUALIFICATIONS|SUBMISSION|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0).strip()
        
        return None
    
    def _load_stopwords(self) -> set:
        """Load stopwords for language"""
        try:
            import nltk
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words(self.language))
        except:
            # Default English stopwords
            return {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

# ============ ENHANCED SEMANTIC CHUNKER ============

class EnhancedSemanticChunker:
    """
    SEMANTIC CHUNKER WITH PRE-PROCESSING
    Intelligently chunks pre-processed text
    """
    
    def __init__(self, 
                 chunk_size: int = 250,
                 overlap: int = 50,
                 strategy: str = 'semantic'):
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        
        # Semantic boundary detectors
        self.boundary_detectors = {
            'paragraph': self._detect_paragraph_boundaries,
            'sentence': self._detect_sentence_boundaries,
            'topic': self._detect_topic_boundaries,
            'section': self._detect_section_boundaries
        }
    
    def chunk_with_semantics(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text with semantic awareness
        """
        print(f"🔪 Chunking {len(text.split()):,} words with {self.strategy} strategy")
        
        # Choose chunking strategy
        if self.strategy == 'semantic':
            chunks = self._chunk_semantic(text, metadata)
        elif self.strategy == 'section':
            chunks = self._chunk_by_sections(text, metadata)
        elif self.strategy == 'paragraph':
            chunks = self._chunk_by_paragraphs(text, metadata)
        else:
            chunks = self._chunk_fixed(text, metadata)
        
        # Add semantic metadata
        chunks = self._add_semantic_metadata(chunks, text)
        
        print(f"✅ Created {len(chunks)} semantic chunks")
        return chunks
    
    def _chunk_semantic(self, text: str, metadata: Dict) -> List[Dict]:
        """Semantic chunking using multiple boundary detectors"""
        chunks = []
        
        # Detect all possible boundaries
        boundaries = self._find_semantic_boundaries(text)
        
        # Create chunks at semantic boundaries
        current_chunk = []
        current_words = 0
        
        for segment, boundary_type in boundaries:
            segment_words = segment.split()
            segment_word_count = len(segment_words)
            
            if current_words + segment_word_count <= self.chunk_size:
                current_chunk.append((segment, boundary_type))
                current_words += segment_word_count
            else:
                # Save current chunk
                if current_chunk:
                    chunk_text = self._combine_segments(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, metadata))
                
                # Start new chunk
                current_chunk = [(segment, boundary_type)]
                current_words = segment_word_count
        
        # Add final chunk
        if current_chunk:
            chunk_text = self._combine_segments(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata))
        
        return chunks
    
    def _find_semantic_boundaries(self, text: str) -> List[Tuple[str, str]]:
        """Find semantic boundaries in text"""
        segments = []
        
        # First, split by major sections
        sections = self._detect_section_boundaries(text)
        
        for section_text, section_type in sections:
            # Within each section, split by paragraphs
            paragraphs = self._detect_paragraph_boundaries(section_text)
            
            for para_text, para_type in paragraphs:
                # Within each paragraph, split by sentences if needed
                if len(para_text.split()) > self.chunk_size:
                    sentences = self._detect_sentence_boundaries(para_text)
                    segments.extend(sentences)
                else:
                    segments.append((para_text, 'paragraph'))
        
        return segments
    
    def _detect_section_boundaries(self, text: str) -> List[Tuple[str, str]]:
        """Detect document sections"""
        sections = []
        
        # Common section patterns
        patterns = [
            (r'^(ABSTRACT|SUMMARY)\b', 'abstract'),
            (r'^(BACKGROUND|INTRODUCTION)\b', 'background'),
            (r'^(METHODS?|APPROACH|DESIGN)\b', 'methods'),
            (r'^(RESULTS?|FINDINGS)\b', 'results'),
            (r'^(DISCUSSION|CONCLUSION)\b', 'discussion'),
            (r'^(REFERENCES|BIBLIOGRAPHY)\b', 'references')
        ]
        
        lines = text.split('\n')
        current_section = []
        current_type = 'header'
        
        for line in lines:
            line_stripped = line.strip()
            section_found = False
            
            for pattern, section_type in patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # Save previous section
                    if current_section:
                        sections.append(('\n'.join(current_section), current_type))
                    
                    # Start new section
                    current_section = [line_stripped]
                    current_type = section_type
                    section_found = True
                    break
            
            if not section_found and line_stripped:
                current_section.append(line_stripped)
        
        # Add final section
        if current_section:
            sections.append(('\n'.join(current_section), current_type))
        
        return sections
    
    def _detect_paragraph_boundaries(self, text: str) -> List[Tuple[str, str]]:
        """Detect paragraph boundaries"""
        paragraphs = text.split('\n\n')
        return [(p.strip(), 'paragraph') for p in paragraphs if p.strip()]
    
    def _detect_sentence_boundaries(self, text: str) -> List[Tuple[str, str]]:
        """Detect sentence boundaries"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [(s.strip(), 'sentence') for s in sentences if s.strip()]
    
    def _detect_topic_boundaries(self, text: str) -> List[Tuple[str, str]]:
        """Detect topic boundaries using keyword analysis"""
        # This is simplified - in production, use topic modeling
        sentences = self._detect_sentence_boundaries(text)
        
        # Group sentences by topic keywords
        chunks = []
        current_chunk = []
        
        for sentence, _ in sentences:
            current_chunk.append(sentence)
            
            # Check for topic shift keywords
            topic_shift_keywords = ['however', 'therefore', 'furthermore', 
                                   'in addition', 'on the other hand', 'in contrast']
            
            if any(keyword in sentence.lower() for keyword in topic_shift_keywords):
                if current_chunk:
                    chunks.append((' '.join(current_chunk), 'topic'))
                    current_chunk = []
        
        # Add remaining sentences
        if current_chunk:
            chunks.append((' '.join(current_chunk), 'topic'))
        
        return chunks
    
    def _combine_segments(self, segments: List[Tuple[str, str]]) -> str:
        """Combine segments into chunk text"""
        chunk_text = []
        
        for segment, boundary_type in segments:
            if boundary_type == 'paragraph':
                chunk_text.append(segment)
            elif boundary_type == 'sentence':
                # Add space for sentences
                chunk_text.append(segment + ' ')
            else:
                chunk_text.append(segment)
        
        return ' '.join(chunk_text).strip()
    
    def _create_chunk(self, text: str, metadata: Dict) -> Dict:
        """Create chunk dictionary"""
        words = text.split()
        
        return {
            'text': text,
            'word_count': len(words),
            'char_count': len(text),
            'chunk_type': 'semantic',
            'boundary_types': metadata.get('boundary_types', []),
            'semantic_score': self._calculate_semantic_score(text),
            'contains_key_terms': self._extract_key_terms(text),
            'coherence_score': self._calculate_coherence(text)
        }
    
    def _calculate_semantic_score(self, text: str) -> float:
        """Calculate semantic coherence score"""
        # Simple implementation - in production use ML model
        sentences = self._detect_sentence_boundaries(text)
        
        if len(sentences) < 2:
            return 0.5
        
        # Check for transition words indicating coherence
        transition_words = ['therefore', 'however', 'thus', 'consequently',
                           'furthermore', 'moreover', 'in addition']
        
        has_transitions = any(any(tw in sentence.lower() for tw in transition_words)
                            for sentence, _ in sentences)
        
        return 0.7 if has_transitions else 0.5
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from chunk"""
        # Remove stopwords and get most frequent terms
        words = text.lower().split()
        filtered = [w for w in words if len(w) > 3 and w not in self.stopwords]
        
        # Count frequencies
        from collections import Counter
        term_counts = Counter(filtered)
        
        # Return top 5 terms
        return [term for term, _ in term_counts.most_common(5)]
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence score"""
        # Simplified coherence measure
        sentences = self._detect_sentence_boundaries(text)
        
        if len(sentences) < 2:
            return 0.3
        
        # Check for pronoun references (indicates coherence)
        pronouns = ['it', 'they', 'this', 'that', 'these', 'those', 'he', 'she']
        pronoun_count = sum(1 for sentence, _ in sentences 
                          if any(pronoun in sentence.lower().split() 
                               for pronoun in pronouns))
        
        return min(pronoun_count / len(sentences), 1.0)
    
    def _add_semantic_metadata(self, chunks: List[Dict], original_text: str) -> List[Dict]:
        """Add semantic metadata to chunks"""
        for i, chunk in enumerate(chunks):
            # Add position context
            chunk['chunk_index'] = i
            chunk['total_chunks'] = len(chunks)
            
            # Add relative position
            chunk['position_pct'] = i / max(len(chunks) - 1, 1)
            
            # Add contextual info
            if i > 0:
                chunk['previous_chunk_summary'] = chunks[i-1]['contains_key_terms'][:3]
            if i < len(chunks) - 1:
                chunk['next_chunk_summary'] = chunks[i+1]['contains_key_terms'][:3]
        
        return chunks

# ============ COMPLETE PRODUCTION PIPELINE ============

class ProductionDocumentPipeline:
    """
    COMPLETE PIPELINE: PDF → CLEAN TEXT → SEMANTIC CHUNKS
    """
    
    def __init__(self, 
                 output_dir: str = "./processed_documents/",
                 cache_dir: str = "./cache/"):
        
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.preprocessor = IntelligentPreprocessor()
        self.chunker = EnhancedSemanticChunker(
            chunk_size=250,
            overlap=50,
            strategy='semantic'
        )
    
    def process_document(self, 
                        input_path: str,
                        document_type: str = None) -> Dict:
        """
        Complete processing pipeline for a single document
        """
        print(f"\n📊 PROCESSING: {os.path.basename(input_path)}")
        print("="*50)
        
        # Step 1: Extract from PDF
        if input_path.lower().endswith('.pdf'):
            extracted = self.pdf_extractor.extract_with_semantics(input_path)
            if not extracted['text']:
                print("❌ Failed to extract text from PDF")
                return None
        else:
            # Assume text file
            with open(input_path, 'r', encoding='utf-8') as f:
                extracted = {
                    'text': f.read(),
                    'metadata': {'source': input_path},
                    'sections': []
                }
        
        # Step 2: Determine document type
        if not document_type:
            document_type = extracted.get('metadata', {}).get('doc_type', 'general')
        
        # Step 3: Pre-process for semantic chunking
        clean_text = self.preprocessor.preprocess_for_chunking(
            extracted['text'], 
            document_type
        )
        
        # Step 4: Semantic chunking
        chunks = self.chunker.chunk_with_semantics(
            clean_text,
            metadata=extracted['metadata']
        )
        
        # Step 5: Save results
        result = self._save_processing_results(
            input_path, extracted, clean_text, chunks
        )
        
        print(f"\n✅ PROCESSING COMPLETE")
        print(f"   • Original: {len(extracted['text']):,} chars")
        print(f"   • Cleaned: {len(clean_text):,} chars")
        print(f"   • Chunks: {len(chunks)} semantic chunks")
        print(f"   • Saved to: {result['output_files']['chunks']}")
        
        return result
    
    def process_batch(self, 
                     input_dir: str,
                     file_pattern: str = "*.pdf") -> List[Dict]:
        """
        Process batch of documents
        """
        import glob
        
        # Find all matching files
        pattern = os.path.join(input_dir, file_pattern)
        files = glob.glob(pattern)
        
        print(f"📦 Processing batch of {len(files)} files")
        
        results = []
        
        for i, file_path in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] Processing: {os.path.basename(file_path)}")
            
            try:
                result = self.process_document(file_path)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        
        # Create batch summary
        self._create_batch_summary(results)
        
        return results
    
    def _save_processing_results(self, 
                                input_path: str,
                                extracted: Dict,
                                clean_text: str,
                                chunks: List[Dict]) -> Dict:
        """Save all processing results"""
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output files
        output_files = {
            'extracted': f"{self.output_dir}/{base_name}_extracted.json",
            'cleaned': f"{self.output_dir}/{base_name}_cleaned.txt",
            'chunks': f"{self.output_dir}/{base_name}_chunks.json",
            'metadata': f"{self.output_dir}/{base_name}_metadata.json"
        }
        
        # Save extracted data
        with open(output_files['extracted'], 'w') as f:
            json.dump(extracted, f, indent=2)
        
        # Save cleaned text
        with open(output_files['cleaned'], 'w', encoding='utf-8') as f:
            f.write(clean_text)
        
        # Save chunks
        with open(output_files['chunks'], 'w') as f:
            json.dump(chunks, f, indent=2)
        
        # Create metadata
        metadata = {
            'processing_timestamp': timestamp,
            'source_file': input_path,
            'document_type': extracted.get('metadata', {}).get('doc_type', 'unknown'),
            'stats': {
                'original_chars': len(extracted['text']),
                'cleaned_chars': len(clean_text),
                'chunk_count': len(chunks),
                'avg_chunk_size': np.mean([c['word_count'] for c in chunks]) if chunks else 0
            },
            'output_files': output_files
        }
        
        with open(output_files['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def _create_batch_summary(self, results: List[Dict]):
        """Create batch processing summary"""
        if not results:
            return
        
        summary = {
            'batch_timestamp': datetime.now().isoformat(),
            'total_documents': len(results),
            'total_chunks': sum(r['stats']['chunk_count'] for r in results),
            'avg_chunks_per_doc': np.mean([r['stats']['chunk_count'] for r in results]),
            'documents_by_type': {},
            'processing_stats': []
        }
        
        for result in results:
            doc_type = result.get('document_type', 'unknown')
            summary['documents_by_type'][doc_type] = summary['documents_by_type'].get(doc_type, 0) + 1
            summary['processing_stats'].append(result['stats'])
        
        # Save summary
        summary_file = f"{self.output_dir}/batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 BATCH SUMMARY saved to: {summary_file}")

# ============================================================================
# 🏃‍♂️ TEST THE PIPELINE
# ============================================================================

def test_production_pipeline():
    """Test the complete production pipeline"""
    print("="*70)
    print("🧪 TESTING PRODUCTION PIPELINE")
    print("="*70)
    
    # Initialize pipeline
    pipeline = ProductionDocumentPipeline()
    
    # Test with a sample PDF or text file
    test_file = input("\n📁 Enter path to test document (PDF or text): ").strip()
    
    if not os.path.exists(test_file):
        print("⚠️  File not found. Using sample text...")
        # Create sample text
        sample_text = """
        ABSTRACT: This study examines diabetes prevention in Federally Qualified Health Centers.
        
        BACKGROUND: Diabetes rates are increasing in underserved populations.
        
        METHODS: We conducted a randomized controlled trial with 500 participants.
        
        RESULTS: Intervention group showed 20% reduction in HbA1c levels.
        
        DISCUSSION: Community health workers are effective for diabetes prevention.
        """
        
        # Save sample to file
        test_file = "./sample_document.txt"
        with open(test_file, 'w') as f:
            f.write(sample_text)
    
    # Process the document
    result = pipeline.process_document(test_file)
    
    if result:
        # Show sample chunks
        chunks_file = result['output_files']['chunks']
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        
        print(f"\n📄 SAMPLE CHUNKS ({len(chunks)} total):")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n{i+1}. Chunk Type: {chunk.get('chunk_type', 'N/A')}")
            print(f"   Words: {chunk['word_count']}, Coherence: {chunk.get('coherence_score', 0):.2f}")
            print(f"   Key Terms: {', '.join(chunk.get('contains_key_terms', []))}")
            print(f"   Text: {chunk['text'][:150]}...")
    
    return pipeline, result

if __name__ == "__main__":
    # Install required packages
    print("📦 Installing required packages...")
    
    required = ['PyPDF2', 'pdfplumber', 'pdfminer.six', 'pymupdf', 'nltk']
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  📥 Installing {package}...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Run test
    pipeline, result = test_production_pipeline()
    
    print("\n" + "="*70)
    print("✅ PRODUCTION PIPELINE READY!")
    print("="*70)
    print("\n🎯 Use pipeline.process_document('your_file.pdf') for single documents")
    print("🎯 Use pipeline.process_batch('/your/folder/', '*.pdf') for batch processing")