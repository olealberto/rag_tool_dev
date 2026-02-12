# ============================================================================
# 📁 phase3_document_rag.py - FQHC RAG WITH DOCUMENT CHUNKING (EMBEDDING SAVE/LOAD)
# ============================================================================

"""
PHASE 3: FQHC-FOCUSED RAG BASELINE WITH DOCUMENT CHUNKING
NOW WITH EMBEDDING SAVE/LOAD - 90 mins first time, 5 seconds thereafter!
"""

print("="*70)
print("🎯 PHASE 3: FQHC-FOCUSED RAG WITH DOCUMENT CHUNKING")
print("="*70)

import sys
sys.path.append('.')

from config import RAG_CONFIG
from utils import logger, DataProcessor
import pandas as pd
import numpy as np
import time
import json
import os
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime

# EDIT HERE: Import FAISS for vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not available. Run: !pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️  Sentence transformers not available.")

# ============ DOCUMENT CHUNKING CLASS ============

class DocumentChunker:
    """
    CHUNK GRANT DOCUMENTS FOR RFP MATCHING
    Splits documents into meaningful sections
    """
    
    def __init__(self, chunk_size: int = 250, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.section_patterns = {
            'abstract': r'(?:PROJECT\s+)?(?:SUMMARY/)?ABSTRACT[:]?\s*',
            'specific_aims': r'SPECIFIC\s+AIMS?[:]?\s*',
            'background': r'BACKGROUND\s*(?:AND\s+SIGNIFICANCE)?[:]?\s*',
            'methods': r'(?:RESEARCH\s+)?(?:DESIGN\s+AND\s+)?METHODS?[:]?\s*',
            'results': r'RESULTS?[:]?\s*',
            'discussion': r'DISCUSSION[:]?\s*',
            'significance': r'SIGNIFICANCE\s*(?:AND\s+IMPACT)?[:]?\s*',
            'innovation': r'INNOVATION[:]?\s*',
            'approach': r'APPROACH[:]?\s*',
        }
    
    def chunk_full_documents(self, documents_df: pd.DataFrame) -> pd.DataFrame:
        """
        Chunk full grant documents into sections for RFP matching
        """
        print(f"\n🔪 Chunking {len(documents_df)} documents...")
        
        all_chunks = []
        
        for idx, document in documents_df.iterrows():
            grant_id = document.get('grant_id', f'doc_{idx}')
            title = document.get('title', 'Untitled')
            full_text = self._get_full_text(document)
            
            # Chunk by sections if available
            chunks = self._chunk_by_sections(full_text, grant_id, title)
            
            # If no sections found, chunk by semantic units
            if not chunks:
                chunks = self._chunk_semantic(full_text, grant_id, title)
            
            # Add document metadata to each chunk
            for chunk in chunks:
                chunk.update({
                    'grant_id': grant_id,
                    'document_title': title,
                    'year': document.get('year', 2024),
                    'institute': document.get('institute', 'Unknown'),
                    'is_fqhc_focused': document.get('is_fqhc_focused', False),
                    'fqhc_score': document.get('fqhc_score', 0.0),
                    'data_source': document.get('data_source', 'unknown'),
                    'primary_condition': document.get('primary_condition', 'general'),
                    'chunk_id': f"{grant_id}_chunk{len(all_chunks)}"
                })
                all_chunks.append(chunk)
            
            if (idx + 1) % 100 == 0:
                print(f"  Chunked {idx+1}/{len(documents_df)} documents...")
        
        print(f"✅ Created {len(all_chunks)} chunks from {len(documents_df)} documents")
        print(f"   Avg chunks per document: {len(all_chunks)/len(documents_df):.1f}")
        
        return pd.DataFrame(all_chunks)
    
    def _get_full_text(self, document: Dict) -> str:
        """Extract full text from document"""
        # Try different text fields
        text_fields = ['full_text', 'text', 'abstract', 'content', 'narrative']
        
        for field in text_fields:
            if field in document and isinstance(document[field], str):
                return document[field]
        
        # Fallback: combine available text fields
        text_parts = []
        for field in ['title', 'abstract', 'specific_aims', 'methods']:
            if field in document and isinstance(document[field], str):
                text_parts.append(document[field])
        
        return "\n\n".join(text_parts)
    
    def _chunk_by_sections(self, text: str, grant_id: str, title: str) -> List[Dict]:
        """Chunk document by detected sections"""
        chunks = []
        
        # Find all section boundaries
        section_matches = []
        for section_name, pattern in self.section_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                section_matches.append((match.start(), section_name))
        
        # Sort by position
        section_matches.sort(key=lambda x: x[0])
        
        if not section_matches:
            return chunks
        
        # Extract sections
        for i, (start_pos, section_name) in enumerate(section_matches):
            end_pos = section_matches[i+1][0] if i+1 < len(section_matches) else len(text)
            section_text = text[start_pos:end_pos].strip()
            
            if section_text and len(section_text.split()) >= 20:
                chunks.append({
                    'text': section_text,
                    'chunk_type': 'section',
                    'section_name': section_name,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'word_count': len(section_text.split())
                })
        
        return chunks
    
    def _chunk_semantic(self, text: str, grant_id: str, title: str) -> List[Dict]:
        """Chunk text by semantic units (paragraphs, sentences)"""
        chunks = []
        
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = []
        current_word_count = 0
        
        for para in paragraphs:
            para_words = para.split()
            para_word_count = len(para_words)
            
            if para_word_count > self.chunk_size:
                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    sent_words = sentence.split()
                    sent_word_count = len(sent_words)
                    
                    if current_word_count + sent_word_count <= self.chunk_size:
                        current_chunk.append(sentence)
                        current_word_count += sent_word_count
                    else:
                        if current_chunk:
                            chunks.append(self._create_semantic_chunk(
                                current_chunk, current_word_count, grant_id
                            ))
                        current_chunk = [sentence]
                        current_word_count = sent_word_count
            else:
                if current_word_count + para_word_count <= self.chunk_size:
                    current_chunk.append(para)
                    current_word_count += para_word_count
                else:
                    if current_chunk:
                        chunks.append(self._create_semantic_chunk(
                            current_chunk, current_word_count, grant_id
                        ))
                    current_chunk = [para]
                    current_word_count = para_word_count
        
        if current_chunk:
            chunks.append(self._create_semantic_chunk(
                current_chunk, current_word_count, grant_id
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_semantic_chunk(self, text_parts: List[str], word_count: int, grant_id: str) -> Dict:
        chunk_text = '\n\n'.join(text_parts)
        chunk_type = self._classify_chunk_type(chunk_text)
        
        return {
            'text': chunk_text,
            'chunk_type': chunk_type,
            'section_name': 'semantic_chunk',
            'word_count': word_count,
            'contains_methods': 'method' in chunk_text.lower() or 'approach' in chunk_text.lower(),
            'contains_outcomes': 'outcome' in chunk_text.lower() or 'result' in chunk_text.lower(),
            'contains_innovation': 'innovative' in chunk_text.lower() or 'novel' in chunk_text.lower(),
        }
    
    def _classify_chunk_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ['method', 'approach', 'design', 'procedure']):
            return 'methods'
        elif any(word in text_lower for word in ['result', 'finding', 'outcome', 'data']):
            return 'results'
        elif any(word in text_lower for word in ['background', 'significance', 'rationale']):
            return 'background'
        elif any(word in text_lower for word in ['aim', 'objective', 'goal', 'purpose']):
            return 'aims'
        elif any(word in text_lower for word in ['innovative', 'novel', 'unique', 'advance']):
            return 'innovation'
        else:
            return 'general'


# ============ ENHANCED FQHC DATASET ============

class EnhancedFQHCDataset:
    """
    LOAD AND PREPARE ENHANCED FQHC DATASET
    Combines Phase 2 data with synthetic FQHC examples
    """
    
    def __init__(self, phase2_data_path: str = None, force_rebuild: bool = False):
        self.data_processor = DataProcessor()
        self.phase2_data_path = phase2_data_path or "./phase2_output/nih_research_abstracts.csv"
        self.force_rebuild = force_rebuild
        
    def load_or_create_enhanced_dataset(self, phase2_path: str = None, force_rebuild: bool = None) -> pd.DataFrame:
        """
        Load enhanced FQHC dataset or create it if not exists
        force_rebuild=True ignores cached file and rebuilds from scratch
        """
        if phase2_path:
            self.phase2_data_path = phase2_path
        if force_rebuild is not None:
            self.force_rebuild = force_rebuild
            
        enhanced_path = "./phase3_data/enhanced_fqhc_dataset.csv"
        
        if self.force_rebuild:
            print("🗑️  Force rebuild enabled - deleting cached dataset")
            if os.path.exists(enhanced_path):
                os.remove(enhanced_path)
            data = self._create_enhanced_dataset()
            os.makedirs("./phase3_data", exist_ok=True)
            data.to_csv(enhanced_path, index=False)
            print(f"💾 Saved enhanced dataset to {enhanced_path}")
            print(f"✅ Created {len(data)} documents")
            return data
        
        if os.path.exists(enhanced_path):
            print(f"📂 Loading enhanced dataset from {enhanced_path}")
            data = pd.read_csv(enhanced_path)
            print(f"✅ Loaded {len(data)} documents")
            return data
        else:
            print("📝 Creating enhanced FQHC dataset...")
            data = self._create_enhanced_dataset()
            os.makedirs("./phase3_data", exist_ok=True)
            data.to_csv(enhanced_path, index=False)
            print(f"💾 Saved enhanced dataset to {enhanced_path}")
            return data
    
    def _create_enhanced_dataset(self) -> pd.DataFrame:
        """Create enhanced dataset by combining Phase 2 data + synthetic FQHC examples"""
        all_data = []
        
        # 1. Load Phase 2 data
        try:
            phase2_data = pd.read_csv(self.phase2_data_path)
            print(f"📊 Phase 2 data: {len(phase2_data)} abstracts from {self.phase2_data_path}")
            
            phase2_data['is_fqhc_focused'] = phase2_data['abstract'].apply(
                lambda x: self._detect_fqhc_focus(str(x))
            )
            phase2_data['fqhc_score'] = phase2_data['abstract'].apply(
                lambda x: self._calculate_fqhc_score(str(x))
            )
            phase2_data['data_source'] = 'phase2_nih'
            
            all_data.append(phase2_data)
        except Exception as e:
            print(f"⚠️  Could not load Phase 2 data: {e}")
        
        # 2. Add synthetic FQHC data
        synthetic_data = self._create_synthetic_fqhc_data(50)
        all_data.append(synthetic_data)
        
        if all_data:
            enhanced_data = pd.concat(all_data, ignore_index=True)
            
            if 'grant_id' not in enhanced_data.columns:
                enhanced_data['grant_id'] = [f"DOC_{i}" for i in range(len(enhanced_data))]
            if 'title' not in enhanced_data.columns:
                enhanced_data['title'] = enhanced_data.get('grant_id', 'Untitled')
            
            print(f"\n🎯 Enhanced Dataset Summary:")
            print(f"   • Total documents: {len(enhanced_data)}")
            print(f"   • FQHC-focused: {enhanced_data['is_fqhc_focused'].sum()}")
            print(f"   • Synthetic examples: {(enhanced_data['data_source'] == 'synthetic_fqhc').sum()}")
            
            return enhanced_data
        else:
            print("❌ No data available")
            return pd.DataFrame()
    
    def _create_synthetic_fqhc_data(self, n: int = 50) -> pd.DataFrame:
        """Create synthetic FQHC-focused abstracts"""
        print(f"🧪 Creating {n} synthetic FQHC abstracts...")
        
        templates = [{
            "title": "Community Health Worker Program for {condition} Management in {population} {setting}",
            "abstract": """This {study_type} evaluates a community health worker-led {condition} management program 
            for {population} patients at {setting}. The intervention includes {components} with focus on 
            {focus_area}. {design} with {participants} participants across {num_clinics} clinics. 
            Primary outcomes include {outcomes}. Results show {results}.""",
            "conditions": ["diabetes", "hypertension", "depression", "asthma", "HIV"],
            "populations": ["Latino", "African American", "low-income", "Medicaid", "rural", "older adult"],
            "settings": ["Federally Qualified Health Centers", "community health centers", "safety-net clinics"],
            "study_types": ["randomized controlled trial", "implementation study", "pragmatic trial"],
            "components": ["culturally-adapted education", "regular health screenings", "medication adherence support", "telehealth follow-ups"],
            "focus_areas": ["health disparities reduction", "chronic disease management", "preventive care", "behavioral health integration"],
            "designs": ["Mixed-methods design", "Stepped-wedge design", "Cluster randomized design"],
            "participants": ["200", "500", "1000", "1500"],
            "num_clinics": ["5", "10", "15", "20"],
            "outcomes": ["HbA1c levels", "blood pressure control", "depression scores", "healthcare utilization"],
            "results": ["significant improvements in clinical outcomes", "high patient satisfaction", "cost-effective intervention", "sustainable model for other clinics"]
        }]
        
        synthetic_docs = []
        for i in range(n):
            template = templates[0]
            title = template["title"].format(
                condition=np.random.choice(template["conditions"]),
                population=np.random.choice(template["populations"]),
                setting=np.random.choice(template["settings"])
            )
            abstract = template["abstract"].format(
                study_type=np.random.choice(template["study_types"]),
                condition=np.random.choice(template["conditions"]),
                population=np.random.choice(template["populations"]),
                setting=np.random.choice(template["settings"]),
                components=np.random.choice(template["components"]),
                focus_area=np.random.choice(template["focus_areas"]),
                design=np.random.choice(template["designs"]),
                participants=np.random.choice(template["participants"]),
                num_clinics=np.random.choice(template["num_clinics"]),
                outcomes=np.random.choice(template["outcomes"]),
                results=np.random.choice(template["results"])
            )
            abstract = ' '.join(abstract.split())
            
            synthetic_docs.append({
                'grant_id': f'FQHC_SYNTH_{i:04d}',
                'title': title,
                'abstract': abstract,
                'year': np.random.choice([2022, 2023, 2024]),
                'institute': 'NIMHD',
                'institution': 'SYNTHETIC_FQHC_RESEARCH',
                'abstract_length': len(abstract),
                'word_count': len(abstract.split()),
                'is_fqhc_focused': True,
                'fqhc_score': 0.8 + np.random.random() * 0.2,
                'data_source': 'synthetic_fqhc'
            })
        
        print(f"✅ Created {len(synthetic_docs)} synthetic FQHC abstracts")
        return pd.DataFrame(synthetic_docs)
    
    def _detect_fqhc_focus(self, text: str) -> bool:
        text_lower = text.lower()
        fqhc_keywords = ['federally qualified health center', 'fqhc', 'community health center', 
                        'safety-net clinic', 'medically underserved']
        return any(keyword in text_lower for keyword in fqhc_keywords)
    
    def _calculate_fqhc_score(self, text: str) -> float:
        text_lower = text.lower()
        fqhc_terms = {
            'federally qualified health center': 3.0, 'fqhc': 3.0, 'community health center': 2.5,
            'safety-net clinic': 2.5, 'medically underserved': 2.0, 'low-income': 1.5,
            'uninsured': 1.5, 'medicaid': 1.5, 'health disparities': 2.0, 'primary care access': 1.5
        }
        total_score = sum(weight for term, weight in fqhc_terms.items() if term in text_lower)
        max_possible = sum(fqhc_terms.values())
        return min(total_score / max_possible, 1.0)


# ============ CHUNK-BASED RAG SYSTEM WITH EMBEDDING SAVE/LOAD ============

class ChunkBasedRAG:
    """
    CHUNK-BASED RAG SYSTEM FOR RFP MATCHING
    NOW WITH EMBEDDING SAVE/LOAD - 90 mins first time, 5 seconds thereafter!
    """
    
    def __init__(self, model_name: str = None, load_existing_embeddings: bool = False, 
                 phase2_data_path: str = None, force_rebuild_dataset: bool = False):
        
        if model_name is None:
            model_name = RAG_CONFIG.get("phase3", {}).get("embedding_model", 
                                                         "pritamdeka/S-PubMedBert-MS-MARCO")
        
        print(f"🚀 Initializing Chunk-Based RAG for RFP Matching...")
        print(f"   Model: {model_name}")
        print(f"   Vector store: FAISS (for baseline comparison)")
        print(f"   Mode: {'🔄 LOAD existing embeddings' if load_existing_embeddings else '⚡ CREATE new embeddings'}")
        
        # Load enhanced dataset
        self.dataset_loader = EnhancedFQHCDataset(phase2_data_path, force_rebuild_dataset)
        self.data = self.dataset_loader.load_or_create_enhanced_dataset()
        
        if self.data.empty:
            raise ValueError("No data available for RAG system")
        
        # Chunk documents (always need to do this - it's fast)
        print("\n🔪 Chunking documents for RFP matching...")
        chunker = DocumentChunker(chunk_size=250, overlap=50)
        self.chunks_df = chunker.chunk_full_documents(self.data)
        
        # Load embedding model (needed for queries)
        if EMBEDDING_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            print(f"✅ Loaded embedding model: {model_name}")
        else:
            self.model = None
            print("⚠️  No embedding model available")
        
        # Either load existing embeddings or build new ones
        self.index = None
        self.embeddings = None
        
        if load_existing_embeddings:
            self._load_existing_index()
        else:
            if FAISS_AVAILABLE and self.model:
                self._build_faiss_index()
                self._save_embeddings()  # Save for next time!
    
    def _build_faiss_index(self):
        """Build FAISS vector index for chunks (90 minutes)"""
        print(f"\n🔨 Building FAISS index for {len(self.chunks_df)} chunks...")
        
        chunk_texts = self.chunks_df['text'].fillna('').tolist()
        
        print(f"📐 Embedding {len(chunk_texts)} chunks...")
        self.embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"✅ FAISS index built: {self.index.ntotal} chunks, {dimension} dimensions")
    
    def _save_embeddings(self):
        """Save embeddings and FAISS index for future use (Phase 4, Phase 5, future runs)"""
        print("\n💾 SAVING EMBEDDINGS FOR FUTURE USE...")
        
        os.makedirs('./phase3_results', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, './phase3_results/faiss_index.bin')
        print(f"✅ Saved FAISS index: {self.index.ntotal} vectors")
        
        # Save embeddings as numpy array
        np.save('./phase3_results/chunk_embeddings.npy', self.embeddings)
        print(f"✅ Saved embeddings: {self.embeddings.shape}")
        
        # Save chunks WITH embeddings (for Phase 4)
        self.chunks_df['embedding'] = self.embeddings.tolist()
        self.chunks_df.to_csv('./phase3_results/document_chunks_with_embeddings.csv', index=False)
        print(f"✅ Saved chunks with embeddings: {len(self.chunks_df)} rows")
        
        # Remove embedding column to free memory
        self.chunks_df = self.chunks_df.drop(columns=['embedding'])
        print("✅ Embeddings saved! Future runs can use load_existing_embeddings=True")
    
    def _load_existing_index(self):
        """Load pre-computed embeddings and FAISS index (5 seconds)"""
        print("\n📦 Loading pre-computed embeddings from disk...")
        
        try:
            # Load FAISS index
            self.index = faiss.read_index('./phase3_results/faiss_index.bin')
            print(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")
            
            # Load embeddings
            self.embeddings = np.load('./phase3_results/chunk_embeddings.npy')
            print(f"✅ Loaded embeddings: {self.embeddings.shape}")
            
            # Verify counts match
            if len(self.chunks_df) != self.index.ntotal:
                print(f"⚠️  Warning: Chunks ({len(self.chunks_df)}) vs Index ({self.index.ntotal}) mismatch")
            
        except FileNotFoundError as e:
            print(f"❌ Could not load embeddings: {e}")
            print("⚠️  Run with load_existing_embeddings=False first to generate embeddings")
            raise
    
    # ============ SEARCH METHODS ============
    
    def search(self, query: str, top_k: int = 5, fqhc_boost: bool = True) -> List[Dict]:
        """Search for chunks matching query"""
        if self.index is None or self.model is None:
            print("❌ FAISS index or model not available")
            return []
        
        print(f"\n🔍 Chunk Search: '{query[:50]}...'")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, top_k * 2)
        search_time = time.time() - start_time
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks_df):
                continue
            
            chunk_data = self.chunks_df.iloc[idx].to_dict()
            similarity = float(distances[0][i])
            
            boosted_similarity = similarity
            if fqhc_boost and chunk_data.get('is_fqhc_focused', False):
                boost_factor = 1.0 + (chunk_data.get('fqhc_score', 0.5) * 0.5)
                boosted_similarity = similarity * boost_factor
            
            results.append({
                'rank': len(results) + 1,
                'chunk_id': chunk_data.get('chunk_id', f'chunk_{idx}'),
                'text': chunk_data.get('text', ''),
                'chunk_type': chunk_data.get('chunk_type', 'unknown'),
                'section_name': chunk_data.get('section_name', ''),
                'grant_id': chunk_data.get('grant_id', ''),
                'document_title': chunk_data.get('document_title', 'Untitled'),
                'year': chunk_data.get('year', 'Unknown'),
                'institute': chunk_data.get('institute', 'Unknown'),
                'similarity': similarity,
                'boosted_similarity': boosted_similarity,
                'fqhc_score': chunk_data.get('fqhc_score', 0.0),
                'is_fqhc_focused': chunk_data.get('is_fqhc_focused', False),
                'data_source': chunk_data.get('data_source', 'unknown'),
                'word_count': chunk_data.get('word_count', 0),
                'contains_methods': chunk_data.get('contains_methods', False),
                'contains_outcomes': chunk_data.get('contains_outcomes', False),
                'search_time': search_time,
                'retrieval_method': 'faiss_chunk_vector',
                'relevance_explanation': self._explain_relevance(query, chunk_data['text'])
            })
        
        results.sort(key=lambda x: x['boosted_similarity'], reverse=True)
        final_results = results[:top_k]
        
        print(f"📊 Found {len(final_results)} chunks in {search_time:.3f}s")
        if final_results:
            print(f"   Top similarity: {final_results[0]['similarity']:.3f}")
            print(f"   Top chunk type: {final_results[0]['chunk_type']}")
        
        return final_results
    
    def _explain_relevance(self, query: str, chunk_text: str) -> str:
        query_words = set(query.lower().split())
        chunk_words = set(chunk_text.lower().split())
        overlap = query_words.intersection(chunk_words)
        if len(overlap) > 3:
            return f"Contains key terms: {', '.join(list(overlap)[:5])}"
        else:
            return "Semantic similarity to requirement"
    
    def _calculate_fqhc_score(self, text: str) -> float:
        if not isinstance(text, str):
            return 0.0
        return self.dataset_loader._calculate_fqhc_score(text)
    
    # ============ RFP MATCHING ============
    
    def match_rfp_requirements(self, rfp_text: str, requirements: List[str] = None, top_k: int = 5) -> Dict:
        """Match RFP requirements to document chunks"""
        print(f"\n📋 Matching RFP requirements...")
        
        if requirements is None:
            requirements = self._extract_requirements(rfp_text)
        
        results = {}
        for i, requirement in enumerate(requirements):
            print(f"\n🔍 Requirement {i+1}: {requirement[:80]}...")
            chunks = self.search(requirement, top_k=top_k, fqhc_boost=True)
            categorized = self._categorize_chunks_for_rfp(chunks, requirement)
            results[f"requirement_{i+1}"] = {
                "requirement": requirement,
                "total_matches": len(chunks),
                "best_matches": chunks[:3],
                "categorized": categorized,
                "suggested_sections": self._suggest_rfp_sections(chunks)
            }
        return results
    
    def _extract_requirements(self, rfp_text: str) -> List[str]:
        requirements = []
        patterns = [
            r'Requirements?:?\s*(.+?)(?=Requirements?|Qualifications|Deliverables|$)',
            r'Applicants? must\s*(.+?)(?=\.|Applicants? must|$)',
            r'Proposals? should\s*(.+?)(?=\.|Proposals? should|$)',
            r'Projects? will\s*(.+?)(?=\.|Projects? will|$)',
            r'Key\s+(?:Components|Elements):?\s*(.+?)(?=Key|$)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, rfp_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match.strip()) > 10:
                    requirements.append(match.strip())
        if not requirements:
            sentences = re.split(r'(?<=[.!?])\s+', rfp_text)
            requirements = [s.strip() for s in sentences if len(s.split()) > 5]
        return requirements[:10]
    
    def _categorize_chunks_for_rfp(self, chunks: List[Dict], requirement: str) -> Dict:
        categories = {'methods_approach': [], 'background_significance': [], 'innovation': [],
                     'evaluation_outcomes': [], 'implementation_plan': [], 'budget_justification': []}
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', '').lower()
            text = chunk.get('text', '').lower()
            if 'method' in chunk_type or chunk.get('contains_methods', False):
                categories['methods_approach'].append(chunk)
            elif 'background' in chunk_type:
                categories['background_significance'].append(chunk)
            elif 'innovation' in chunk_type or 'innovative' in text:
                categories['innovation'].append(chunk)
            elif 'result' in chunk_type or chunk.get('contains_outcomes', False):
                categories['evaluation_outcomes'].append(chunk)
            elif any(word in text for word in ['implement', 'timeline', 'plan', 'schedule']):
                categories['implementation_plan'].append(chunk)
            elif any(word in text for word in ['budget', 'cost', 'funding', 'resource']):
                categories['budget_justification'].append(chunk)
            else:
                categories['methods_approach'].append(chunk)
        return {k: v for k, v in categories.items() if v}
    
    def _suggest_rfp_sections(self, chunks: List[Dict]) -> List[str]:
        sections = set()
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', '')
            if chunk_type in ['methods', 'approach']:
                sections.add('Methods and Approach')
            elif chunk_type == 'background':
                sections.add('Background and Significance')
            elif chunk_type == 'innovation':
                sections.add('Innovation')
            elif chunk_type == 'results':
                sections.add('Evaluation and Outcomes')
            elif 'implementation' in chunk.get('text', '').lower():
                sections.add('Implementation Plan')
            elif 'budget' in chunk.get('text', '').lower():
                sections.add('Budget Justification')
        return list(sections)
    
    # ============ EVALUATION METHODS ============
    
    def evaluate(self, test_queries: List[Dict] = None) -> Dict:
        """Evaluate chunk-based RAG performance"""
        print("\n🧪 Evaluating Chunk-Based RAG...")
        
        if test_queries is None:
            try:
                with open('./phase2_output/evaluation_set.json', 'r') as f:
                    test_queries = json.load(f)
                print(f"📋 Using {len(test_queries)} test queries from Phase 2")
            except:
                print("⚠️  Creating default FQHC test queries")
                test_queries = self._create_fqhc_test_queries()
        
        metrics = {'precision_at_1': [], 'precision_at_3': [], 'precision_at_5': [],
                  'fqhc_alignment': [], 'retrieval_time': [], 'avg_similarity': []}
        
        for i, query_data in enumerate(test_queries):
            query = query_data.get('query', '')
            relevant_ids = set(query_data.get('relevant_grant_ids', []))
            results = self.search(query, top_k=5, fqhc_boost=True)
            retrieved_ids = [r['grant_id'] for r in results]
            
            for k in [1, 3, 5]:
                top_k_ids = retrieved_ids[:k]
                relevant_in_top_k = len([id for id in top_k_ids if id in relevant_ids])
                precision = relevant_in_top_k / k if k > 0 else 0
                metrics[f'precision_at_{k}'].append(precision)
            
            if results:
                metrics['fqhc_alignment'].append(np.mean([r['fqhc_score'] for r in results[:3]]))
                metrics['retrieval_time'].append(results[0]['search_time'])
                metrics['avg_similarity'].append(np.mean([r['similarity'] for r in results[:3]]))
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(test_queries)} queries...")
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
        
        print(f"\n📊 Chunk-Based RAG Results:")
        print(f"   • Precision@1: {avg_metrics.get('precision_at_1', 0):.3f}")
        print(f"   • Precision@3: {avg_metrics.get('precision_at_3', 0):.3f}")
        print(f"   • Precision@5: {avg_metrics.get('precision_at_5', 0):.3f}")
        print(f"   • FQHC Alignment: {avg_metrics.get('fqhc_alignment', 0):.3f}")
        print(f"   • Avg Retrieval Time: {avg_metrics.get('retrieval_time', 0):.3f}s")
        
        return avg_metrics
    
    def _create_fqhc_test_queries(self) -> List[Dict]:
        return [
            {'query': 'diabetes prevention in Federally Qualified Health Centers', 'relevant_grant_ids': ['FQHC_SYNTH_0001', 'FQHC_SYNTH_0010', 'FQHC_SYNTH_0020'], 'type': 'fqhc_chronic_disease'},
            {'query': 'community health worker programs for underserved populations', 'relevant_grant_ids': ['FQHC_SYNTH_0005', 'FQHC_SYNTH_0015', 'FQHC_SYNTH_0025'], 'type': 'fqhc_intervention'},
            {'query': 'telehealth implementation in rural community health centers', 'relevant_grant_ids': ['FQHC_SYNTH_0003', 'FQHC_SYNTH_0013', 'FQHC_SYNTH_0023'], 'type': 'fqhc_technology'},
            {'query': 'health disparities reduction in safety-net clinics', 'relevant_grant_ids': ['FQHC_SYNTH_0007', 'FQHC_SYNTH_0017', 'FQHC_SYNTH_0027'], 'type': 'fqhc_equity'},
            {'query': 'behavioral health integration in primary care FQHCs', 'relevant_grant_ids': ['FQHC_SYNTH_0009', 'FQHC_SYNTH_0019', 'FQHC_SYNTH_0029'], 'type': 'fqhc_behavioral_health'}
        ]
    
    def create_proper_evaluation(self, num_queries: int = 20) -> List[Dict]:
        """Create proper evaluation queries that match actual documents"""
        print(f"\n🎯 Creating proper evaluation from {len(self.data)} documents...")
        fqhc_docs = self.data[self.data['is_fqhc_focused'] == True]
        if len(fqhc_docs) < num_queries:
            num_queries = len(fqhc_docs)
        
        eval_set = []
        for i, (_, doc) in enumerate(fqhc_docs.head(num_queries).iterrows()):
            full_text = (doc.get('title', '') + ' ' + doc.get('abstract', '')).lower()
            if 'diabetes' in full_text:
                query = "diabetes management in community health settings"
            elif 'hypertension' in full_text or 'blood pressure' in full_text:
                query = "hypertension control in underserved populations"
            elif 'depression' in full_text or 'mental health' in full_text:
                query = "behavioral health integration in primary care"
            elif 'asthma' in full_text:
                query = "asthma management in pediatric populations"
            elif 'cancer' in full_text:
                query = "cancer screening in community health centers"
            elif 'hiv' in full_text:
                query = "HIV prevention and care in safety-net settings"
            else:
                words = doc.get('title', '').split()[:4]
                query = f"{' '.join(words)} in Federally Qualified Health Centers"
            
            eval_set.append({
                "query_id": f"Q{i+1:03d}_PROPER",
                "query": query,
                "relevant_grant_ids": [doc['grant_id']],
                "query_type": "document_based",
                "condition": doc.get('primary_condition', 'general'),
                "source_document": doc['grant_id']
            })
        
        os.makedirs('./phase3_results', exist_ok=True)
        with open('./phase3_results/proper_evaluation.json', 'w') as f:
            json.dump(eval_set, f, indent=2)
        print(f"✅ Created {len(eval_set)} proper evaluation queries")
        return eval_set
    
    def evaluate_with_proper_queries(self, num_queries: int = 20) -> Dict:
        """Evaluate using proper queries (recommended for accurate metrics)"""
        print("\n🧪 EVALUATION WITH PROPER GROUND TRUTH")
        eval_path = './phase3_results/proper_evaluation.json'
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                test_queries = json.load(f)
            print(f"📋 Loaded existing proper evaluation: {len(test_queries)} queries")
        else:
            test_queries = self.create_proper_evaluation(num_queries)
        return self._run_evaluation_with_queries(test_queries)
    
    def _run_evaluation_with_queries(self, test_queries: List[Dict]) -> Dict:
        metrics = {'precision_at_1': [], 'precision_at_3': [], 'precision_at_5': [],
                  'fqhc_alignment': [], 'retrieval_time': [], 'avg_similarity': []}
        
        for query_data in test_queries:
            query = query_data.get('query', '')
            relevant_ids = set(query_data.get('relevant_grant_ids', []))
            results = self.search(query, top_k=5, fqhc_boost=True)
            retrieved_ids = [r['grant_id'] for r in results]
            
            for k in [1, 3, 5]:
                top_k_ids = retrieved_ids[:k]
                relevant_in_top_k = len([id for id in top_k_ids if id in relevant_ids])
                metrics[f'precision_at_{k}'].append(relevant_in_top_k / k if k > 0 else 0)
            
            if results:
                metrics['fqhc_alignment'].append(np.mean([r['fqhc_score'] for r in results[:3]]))
                metrics['retrieval_time'].append(results[0]['search_time'])
                metrics['avg_similarity'].append(np.mean([r['similarity'] for r in results[:3]]))
        
        return {k: np.mean(v) for k, v in metrics.items() if v}
    
    # ============ INTERACTIVE DEMOS ============
    
    def interactive_rfp_matching(self):
        """Interactive RFP matching demo"""
        print("\n" + "="*70)
        print("💼 INTERACTIVE RFP MATCHING DEMO")
        print("="*70)
        print("Enter RFP requirements to find matching grant sections")
        print("Type 'quit' to exit, 'sample' for sample RFP")
        
        while True:
            user_input = input("\n📋 Enter RFP requirement (or 'sample'): ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input.lower() == 'sample':
                requirement = "Develop a community health worker program for diabetes prevention in underserved populations with evaluation metrics including HbA1c reduction and cost-effectiveness analysis"
                print(f"\n📋 Sample RFP: {requirement}")
            else:
                requirement = user_input
            
            print(f"\n🔍 Searching for: '{requirement[:100]}...'")
            chunks = self.search(requirement, top_k=3, fqhc_boost=True)
            
            if not chunks:
                print("No matching chunks found. Try more specific requirement.")
                continue
            
            print(f"✅ Found {len(chunks)} relevant chunks:")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n{i}. 📄 From: {chunk['document_title']}")
                print(f"   📅 Year: {chunk['year']} | Grant: {chunk['grant_id']}")
                print(f"   📊 Similarity: {chunk['similarity']:.3f}")
                print(f"   🏷️  Section: {chunk['section_name']} ({chunk['chunk_type']})")
                print(f"   📝 Words: {chunk['word_count']} | FQHC: {chunk['is_fqhc_focused']}")
                print(f"   💡 Use for: {chunk.get('relevance_explanation', 'General reference')}")
                print(f"   📋 Text: {chunk['text'][:200]}...")
            
            suggested = self._suggest_rfp_sections(chunks)
            if suggested:
                print(f"\n🎯 Suggested RFP sections to include:")
                for section in suggested:
                    print(f"   • {section}")
    
    def interactive_demo(self):
        """Interactive demo of chunk-based RAG"""
        print("\n" + "="*70)
        print("💬 CHUNK-BASED RAG DEMO")
        print("="*70)
        print("Test queries against the chunk-based RAG")
        print("Type 'quit' to exit, 'rfp' for RFP matching mode")
        
        while True:
            query = input("\n🔍 Your question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query.lower() == 'rfp':
                self.interactive_rfp_matching()
                continue
            if not query:
                continue
            
            print(f"\n📚 Chunk-Based RAG Searching for: '{query}'")
            results = self.search(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['document_title']}")
                print(f"   📅 Year: {result['year']} | Institute: {result['institute']}")
                print(f"   📊 Similarity: {result['similarity']:.3f} | Boosted: {result['boosted_similarity']:.3f}")
                print(f"   🏷️  Section: {result['section_name']} ({result['chunk_type']})")
                print(f"   🎯 FQHC Score: {result['fqhc_score']:.2f} | FQHC-focused: {result['is_fqhc_focused']}")
                print(f"   📝 Source: {result['data_source']}")
                print(f"   📝 Text: {result['text'][:200]}...")
            
            if not results:
                print("No results found. Try a different query.")


# ============ VISUALIZATION FUNCTIONS ============

def visualize_phase3_results(rag_system, evaluation_metrics: Dict):
    """Visualize Phase 3 results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase 3: Chunk-Based RAG Results', fontsize=16, fontweight='bold')
        
        data = rag_system.data
        chunks_df = rag_system.chunks_df
        
        # 1. Dataset composition
        ax = axes[0, 0]
        categories = ['FQHC-focused', 'Non-FQHC', 'Synthetic', 'Phase 2 NIH']
        counts = [
            data['is_fqhc_focused'].sum(),
            (~data['is_fqhc_focused']).sum(),
            (data['data_source'] == 'synthetic_fqhc').sum(),
            (data['data_source'] == 'phase2_nih').sum()
        ]
        bars = ax.bar(categories, counts, color=['green', 'lightblue', 'orange', 'blue'])
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_title('Enhanced Dataset Composition')
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{count}', ha='center', va='bottom')
        
        # 2. Chunk type distribution
        ax = axes[0, 1]
        if 'chunk_type' in chunks_df.columns:
            chunk_counts = chunks_df['chunk_type'].value_counts()
            ax.bar(range(len(chunk_counts)), chunk_counts.values, color='skyblue')
            ax.set_xlabel('Chunk Type')
            ax.set_ylabel('Count')
            ax.set_title('Chunk Type Distribution')
            ax.set_xticks(range(len(chunk_counts)))
            ax.set_xticklabels(chunk_counts.index, rotation=45, ha='right')
        
        # 3. Evaluation metrics
        ax = axes[0, 2]
        if evaluation_metrics:
            eval_metrics = {
                'P@1': evaluation_metrics.get('precision_at_1', 0),
                'P@3': evaluation_metrics.get('precision_at_3', 0),
                'P@5': evaluation_metrics.get('precision_at_5', 0),
                'FQHC Align': evaluation_metrics.get('fqhc_alignment', 0)
            }
            bars = ax.bar(range(len(eval_metrics)), list(eval_metrics.values()), 
                        color=['skyblue', 'lightgreen', 'salmon', 'gold'])
            ax.set_xticks(range(len(eval_metrics)))
            ax.set_xticklabels(list(eval_metrics.keys()))
            ax.set_ylabel('Score')
            ax.set_title('Evaluation Metrics')
            ax.set_ylim(0, 1)
            for bar, value in zip(bars, eval_metrics.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Chunk size distribution
        ax = axes[1, 0]
        if 'word_count' in chunks_df.columns:
            ax.hist(chunks_df['word_count'], bins=30, alpha=0.7, color='lightgreen')
            ax.set_xlabel('Chunk Size (words)')
            ax.set_ylabel('Frequency')
            ax.set_title('Chunk Size Distribution')
            ax.axvline(x=chunks_df['word_count'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {chunks_df["word_count"].mean():.0f} words')
            ax.legend()
        
        # 5. System info
        ax = axes[1, 1]
        ax.axis('off')
        system_info = [
            ['Model', str(rag_system.model).split('/')[-1][:20] if rag_system.model else 'N/A'],
            ['Documents', str(len(data))],
            ['Chunks', str(len(chunks_df))],
            ['FQHC-focused', str(data['is_fqhc_focused'].sum())],
            ['Vector Dims', str(rag_system.embeddings.shape[1] if rag_system.embeddings is not None else 'N/A')],
            ['FAISS Index', str(rag_system.index.ntotal if rag_system.index is not None else 'N/A')],
            ['Mode', 'LOAD' if hasattr(rag_system, '_load_existing_index') and rag_system.embeddings is not None else 'CREATE']
        ]
        table = ax.table(cellText=system_info, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title('System Configuration')
        
        # 6. RFP capabilities
        ax = axes[1, 2]
        ax.axis('off')
        rfp_capabilities = [
            "📊 CHUNK-BASED RAG FOR RFP MATCHING",
            "",
            "Capabilities:",
            "• Document chunking into sections",
            "• RFP requirement matching",
            "• Chunk categorization",
            "• FQHC relevance boosting",
            f"• {len(chunks_df)} chunks available",
            f"• {data['is_fqhc_focused'].sum()} FQHC-focused docs"
        ]
        ax.text(0.5, 0.5, '\n'.join(rfp_capabilities), ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        os.makedirs('./phase3_results', exist_ok=True)
        plt.savefig('./phase3_results/phase3_chunk_based_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("⚠️  Visualization libraries not available")


# ============ MAIN EXECUTION ============

def run_phase3_chunking(load_existing_embeddings: bool = False, 
                       phase2_data_path: str = None,
                       force_rebuild_dataset: bool = False):
    """
    RUN CHUNK-BASED PHASE 3
    
    Args:
        load_existing_embeddings: If True, load saved embeddings (5 seconds)
                                 If False, create new embeddings (90 minutes)
        phase2_data_path: Path to Phase 2 NIH abstracts CSV
        force_rebuild_dataset: If True, rebuild enhanced dataset from scratch
    """
    print("\n" + "="*70)
    print("🚀 STARTING PHASE 3: CHUNK-BASED RAG")
    print("="*70)
    
    # Step 1: Initialize chunk-based RAG system
    print("\n🤖 STEP 1: INITIALIZING CHUNK-BASED RAG")
    print("-" * 50)
    
    try:
        rag_system = ChunkBasedRAG(
            load_existing_embeddings=load_existing_embeddings,
            phase2_data_path=phase2_data_path,
            force_rebuild_dataset=force_rebuild_dataset
        )
        print("✅ Chunk-based RAG initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return None, None
    
    # Step 2: Evaluate performance
    print("\n🧪 STEP 2: EVALUATING PERFORMANCE")
    print("-" * 50)
    evaluation_metrics = rag_system.evaluate()
    
    # Step 3: Visualization
    print("\n📊 STEP 3: GENERATING VISUALIZATIONS")
    print("-" * 50)
    visualize_phase3_results(rag_system, evaluation_metrics)
    
    # Step 4: Save results
    print("\n💾 STEP 4: SAVING RESULTS")
    print("-" * 50)
    
    results = {
        "phase": "phase3_chunk_based_rag",
        "timestamp": datetime.now().isoformat(),
        "mode": "load_existing" if load_existing_embeddings else "create_new",
        "model": str(rag_system.model).split('/')[-1] if rag_system.model else "unknown",
        "dataset_stats": {
            "total_documents": len(rag_system.data),
            "total_chunks": len(rag_system.chunks_df),
            "avg_chunks_per_doc": len(rag_system.chunks_df) / len(rag_system.data),
            "fqhc_focused": int(rag_system.data['is_fqhc_focused'].sum()),
            "synthetic": int((rag_system.data['data_source'] == 'synthetic_fqhc').sum()),
            "phase2_nih": int((rag_system.data['data_source'] == 'phase2_nih').sum())
        },
        "chunk_stats": {
            "chunk_types": rag_system.chunks_df['chunk_type'].value_counts().to_dict() if 'chunk_type' in rag_system.chunks_df.columns else {},
            "avg_chunk_size": rag_system.chunks_df['word_count'].mean() if 'word_count' in rag_system.chunks_df.columns else 0
        },
        "evaluation_metrics": evaluation_metrics,
        "system_info": {
            "vector_dimensions": rag_system.embeddings.shape[1] if rag_system.embeddings is not None else None,
            "faiss_index_size": rag_system.index.ntotal if rag_system.index is not None else None,
            "embeddings_loaded": load_existing_embeddings,
            "embeddings_saved": os.path.exists('./phase3_results/chunk_embeddings.npy')
        }
    }
    
    os.makedirs('./phase3_results', exist_ok=True)
    with open('./phase3_results/phase3_chunk_based_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save chunk database (without embeddings)
    rag_system.chunks_df.to_csv('./phase3_results/document_chunks.csv', index=False)
    
    print("\n" + "="*70)
    print("✅ PHASE 3 CHUNK-BASED RAG COMPLETE!")
    print("="*70)
    print("\n📁 Results saved to ./phase3_results/:")
    print("  • phase3_chunk_based_results.json")
    print("  • phase3_chunk_based_results.png")
    print("  • document_chunks.csv (all text chunks)")
    
    if load_existing_embeddings:
        print("  • ✅ Used pre-computed embeddings (5 seconds)")
    else:
        print("  • ✅ Created new embeddings (90 minutes) and saved them for future use!")
        print("  • 💾 FAISS index saved to faiss_index.bin")
        print("  • 💾 Embeddings saved to chunk_embeddings.npy")
        print("  • 💾 Chunks+embeddings saved to document_chunks_with_embeddings.csv")
    
    print(f"\n🎯 CHUNK-BASED RAG READY FOR RFP MATCHING:")
    print(f"   1. Enhanced dataset: {len(rag_system.data)} documents")
    print(f"   2. Created {len(rag_system.chunks_df)} text chunks")
    print(f"   3. FAISS index with {rag_system.index.ntotal if rag_system.index else 0} chunk embeddings")
    
    return rag_system, results


# ============================================================================
# 🏃‍♂️ MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Install required packages
    print("📦 Checking/installing required packages...")
    required_packages = []
    if not FAISS_AVAILABLE:
        required_packages.append("faiss-cpu")
    try:
        import matplotlib
    except ImportError:
        required_packages.extend(["matplotlib", "seaborn"])
    if required_packages:
        print(f"Installing: {', '.join(required_packages)}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + required_packages)
    
    print("\n" + "="*70)
    print("🚀 PHASE 3: CHUNK-BASED RAG FOR RFP MATCHING")
    print("="*70)
    print("\n📌 Choose mode:")
    print("   1. FIRST RUN: Create new embeddings (90 minutes)")
    print("   2. SUBSEQUENT RUNS: Load existing embeddings (5 seconds)")
    print("-" * 50)
    
    # Auto-detect if embeddings exist
    embeddings_exist = os.path.exists('./phase3_results/chunk_embeddings.npy')
    
    if embeddings_exist:
        print(f"✅ Found existing embeddings! Loading in 5 seconds...")
        rag_system, results = run_phase3_chunking(load_existing_embeddings=True)
    else:
        print(f"⚠️  No existing embeddings found. Creating new ones (90 minutes)...")
        print(f"   This will only happen once. Future runs will be instant!")
        rag_system, results = run_phase3_chunking(load_existing_embeddings=False)
    
    if rag_system:
        print("\n" + "="*70)
        print("🎮 INTERACTIVE CHUNK-BASED RAG DEMO")
        print("="*70)
        rag_system.interactive_demo()
        
        print("\n" + "="*70)
        print("✅ PHASE 3 READY FOR RFP MATCHING!")
        print("="*70)
        print(f"\n📊 Your RFP Matching Capabilities:")
        print(f"   • {len(rag_system.chunks_df)} text chunks available")
        print(f"   • Embeddings: {'LOADED' if rag_system.embeddings is not None else 'N/A'}")
        print(f"   • FAISS index: {rag_system.index.ntotal if rag_system.index else 0} vectors")
        print("\n🚀 Try: rag_system.interactive_rfp_matching() for RFP requirements!")
        print("="*70)