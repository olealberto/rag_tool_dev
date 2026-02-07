# ============================================================================
# 📁 phase3_document_rag.py - FQHC RAG BASELINE
# ============================================================================

"""
PHASE 3: FQHC-FOCUSED RAG BASELINE (Using Enhanced Dataset)
This serves as baseline comparison for Phase 4 (Weaviate) and Phase 5 (Knowledge Graph)
"""

print("="*70)
print("🎯 PHASE 3: FQHC-FOCUSED RAG BASELINE")
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

class EnhancedFQHCDataset:
    """
    LOAD AND PREPARE ENHANCED FQHC DATASET
    Combines Phase 2 data with synthetic FQHC examples
    """
    
    def __init__(self):
        self.data_processor = DataProcessor()
        
    def load_or_create_enhanced_dataset(self) -> pd.DataFrame:
        """
        Load enhanced FQHC dataset or create it if not exists
        """
        enhanced_path = "./phase3_data/enhanced_fqhc_dataset.csv"
        
        if os.path.exists(enhanced_path):
            print(f"📂 Loading enhanced dataset from {enhanced_path}")
            data = pd.read_csv(enhanced_path)
            print(f"✅ Loaded {len(data)} documents")
        else:
            print("📝 Creating enhanced FQHC dataset...")
            data = self._create_enhanced_dataset()
            os.makedirs("./phase3_data", exist_ok=True)
            data.to_csv(enhanced_path, index=False)
            print(f"💾 Saved enhanced dataset to {enhanced_path}")
        
        # Ensure required columns
        if 'abstract' not in data.columns:
            print("⚠️  'abstract' column not found. Using 'text' if available.")
            if 'text' in data.columns:
                data['abstract'] = data['text']
            else:
                print("❌ No text content available")
                return pd.DataFrame()
        
        return data
    
    def _create_enhanced_dataset(self) -> pd.DataFrame:
        """
        Create enhanced dataset by combining:
        1. Phase 2 NIH abstracts
        2. Synthetic FQHC examples
        3. FQHC-focused metadata
        """
        all_data = []
        
        # 1. Load Phase 2 data
        try:
            phase2_data = pd.read_csv("./phase2_output/nih_research_abstracts.csv")
            print(f"📊 Phase 2 data: {len(phase2_data)} abstracts")
            
            # Add FQHC detection to Phase 2 data
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
        
        # 3. Combine all data
        if all_data:
            enhanced_data = pd.concat(all_data, ignore_index=True)
            
            # Fill missing columns
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
        """
        Create synthetic FQHC-focused abstracts
        """
        print(f"🧪 Creating {n} synthetic FQHC abstracts...")
        
        templates = [
            {
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
            }
        ]
        
        synthetic_docs = []
        
        for i in range(n):
            template = templates[i % len(templates)]
            
            # Fill template
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
            
            # Clean up whitespace
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
        """Detect if text is FQHC-focused"""
        text_lower = text.lower()
        fqhc_keywords = [
            'federally qualified health center',
            'fqhc',
            'community health center',
            'safety-net clinic',
            'medically underserved'
        ]
        return any(keyword in text_lower for keyword in fqhc_keywords)
    
    def _calculate_fqhc_score(self, text: str) -> float:
        """Calculate FQHC relevance score"""
        text_lower = text.lower()
        
        fqhc_terms = {
            'federally qualified health center': 3.0,
            'fqhc': 3.0,
            'community health center': 2.5,
            'safety-net clinic': 2.5,
            'medically underserved': 2.0,
            'low-income': 1.5,
            'uninsured': 1.5,
            'medicaid': 1.5,
            'health disparities': 2.0,
            'primary care access': 1.5
        }
        
        total_score = 0
        for term, weight in fqhc_terms.items():
            if term in text_lower:
                total_score += weight
        
        max_possible = sum(fqhc_terms.values())
        return min(total_score / max_possible, 1.0)

class FQHCRAGBaseline:
    """
    FQHC RAG BASELINE USING FAISS
    This serves as baseline comparison for Phase 4 (Weaviate)
    """
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = RAG_CONFIG.get("phase3", {}).get("embedding_model", 
                                                         "pritamdeka/S-PubMedBert-MS-MARCO")
        
        print(f"🚀 Initializing FQHC RAG Baseline...")
        print(f"   Model: {model_name}")
        print(f"   Vector store: FAISS (for baseline comparison)")
        
        # Load enhanced dataset
        self.dataset_loader = EnhancedFQHCDataset()
        self.data = self.dataset_loader.load_or_create_enhanced_dataset()
        
        if self.data.empty:
            raise ValueError("No data available for RAG system")
        
        # Load embedding model
        if EMBEDDING_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            print(f"✅ Loaded embedding model: {model_name}")
        else:
            self.model = None
            print("⚠️  No embedding model available")
        
        # Build FAISS index
        self.index = None
        self.embeddings = None
        
        if FAISS_AVAILABLE and self.model:
            self._build_faiss_index()
    
    def _build_faiss_index(self):
        """Build FAISS vector index"""
        print("\n🔨 Building FAISS vector index...")
        
        # Extract texts for embedding
        texts = self.data['abstract'].fillna('').tolist()
        
        # Create embeddings
        print(f"📐 Embedding {len(texts)} documents...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine for normalized vectors
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"✅ FAISS index built: {self.index.ntotal} vectors, {dimension} dimensions")
    
    def search(self, query: str, top_k: int = 5, fqhc_boost: bool = True) -> List[Dict]:
        """
        Search using FAISS vector similarity
        """
        if self.index is None or self.model is None:
            print("❌ FAISS index or model not available")
            return []
        
        print(f"\n🔍 FAISS Search: '{query}'")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, top_k * 2)  # Get extra for potential filtering
        search_time = time.time() - start_time
        
        results = []
        query_fqhc_score = self._calculate_fqhc_score(query)
        
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.data):
                continue
            
            document = self.data.iloc[idx].to_dict()
            similarity = float(distances[0][i])
            
            # Calculate FQHC relevance
            doc_fqhc_score = document.get('fqhc_score', 
                                         self._calculate_fqhc_score(
                                             str(document.get('abstract', '')) + ' ' + 
                                             str(document.get('title', ''))
                                         ))
            
            # Apply FQHC boost if requested and query is FQHC-related
            boosted_similarity = similarity
            if fqhc_boost and query_fqhc_score > 0.3:
                boost_factor = 1.0 + (doc_fqhc_score * 0.5)  # Up to 50% boost
                boosted_similarity = similarity * boost_factor
            
            results.append({
                'rank': len(results) + 1,
                'grant_id': document.get('grant_id', f'DOC_{idx}'),
                'title': document.get('title', 'Untitled'),
                'abstract_preview': self._truncate_text(document.get('abstract', ''), 200),
                'year': document.get('year', 'Unknown'),
                'institute': document.get('institute', 'Unknown'),
                'similarity': similarity,
                'boosted_similarity': boosted_similarity,
                'fqhc_score': doc_fqhc_score,
                'is_fqhc_focused': bool(document.get('is_fqhc_focused', False)),
                'data_source': document.get('data_source', 'unknown'),
                'search_time': search_time,
                'retrieval_method': 'faiss_vector'
            })
        
        # Sort by boosted similarity
        results.sort(key=lambda x: x['boosted_similarity'], reverse=True)
        
        # Return top_k results
        final_results = results[:top_k]
        
        print(f"📊 Found {len(final_results)} documents in {search_time:.3f}s")
        if final_results:
            print(f"   Top similarity: {final_results[0]['similarity']:.3f}")
            print(f"   Top FQHC score: {final_results[0]['fqhc_score']:.2f}")
        
        return final_results
    
    def _calculate_fqhc_score(self, text: str) -> float:
        """Calculate FQHC relevance score"""
        if not isinstance(text, str):
            return 0.0
        
        return self.dataset_loader._calculate_fqhc_score(text)
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to specified length"""
        if not isinstance(text, str):
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def evaluate(self, test_queries: List[Dict] = None) -> Dict:
        """
        Evaluate FAISS-based RAG performance
        """
        print("\n🧪 Evaluating FAISS RAG Baseline...")
        
        # Load test queries from Phase 2 or create default
        if test_queries is None:
            try:
                with open('./phase2_output/evaluation_set.json', 'r') as f:
                    test_queries = json.load(f)
                print(f"📋 Using {len(test_queries)} test queries from Phase 2")
            except:
                print("⚠️  Creating default FQHC test queries")
                test_queries = self._create_fqhc_test_queries()
        
        metrics = {
            'precision_at_1': [],
            'precision_at_3': [],
            'precision_at_5': [],
            'fqhc_alignment': [],
            'retrieval_time': [],
            'avg_similarity': []
        }
        
        for i, query_data in enumerate(test_queries):
            query = query_data.get('query', '')
            relevant_ids = set(query_data.get('relevant_grant_ids', []))
            
            results = self.search(query, top_k=5, fqhc_boost=True)
            
            retrieved_ids = [r['grant_id'] for r in results]
            
            # Calculate precision@k
            for k in [1, 3, 5]:
                top_k_ids = retrieved_ids[:k]
                relevant_in_top_k = len([id for id in top_k_ids if id in relevant_ids])
                precision = relevant_in_top_k / k if k > 0 else 0
                metrics[f'precision_at_{k}'].append(precision)
            
            # Calculate FQHC alignment
            if results:
                fqhc_scores = [r['fqhc_score'] for r in results[:3]]
                metrics['fqhc_alignment'].append(np.mean(fqhc_scores))
            
            # Record metrics
            if results:
                metrics['retrieval_time'].append(results[0]['search_time'])
                metrics['avg_similarity'].append(np.mean([r['similarity'] for r in results[:3]]))
            
            # Progress update
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(test_queries)} queries...")
        
        # Calculate average metrics
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
        
        print(f"\n📊 FAISS Baseline Results:")
        print(f"   • Precision@1: {avg_metrics.get('precision_at_1', 0):.3f}")
        print(f"   • Precision@3: {avg_metrics.get('precision_at_3', 0):.3f}")
        print(f"   • Precision@5: {avg_metrics.get('precision_at_5', 0):.3f}")
        print(f"   • FQHC Alignment: {avg_metrics.get('fqhc_alignment', 0):.3f}")
        print(f"   • Avg Retrieval Time: {avg_metrics.get('retrieval_time', 0):.3f}s")
        
        return avg_metrics
    
    def _create_fqhc_test_queries(self) -> List[Dict]:
        """Create FQHC-focused test queries"""
        fqhc_queries = [
            {
                'query': 'diabetes prevention in Federally Qualified Health Centers',
                'relevant_grant_ids': ['FQHC_SYNTH_0001', 'FQHC_SYNTH_0010', 'FQHC_SYNTH_0020'],
                'type': 'fqhc_chronic_disease'
            },
            {
                'query': 'community health worker programs for underserved populations',
                'relevant_grant_ids': ['FQHC_SYNTH_0005', 'FQHC_SYNTH_0015', 'FQHC_SYNTH_0025'],
                'type': 'fqhc_intervention'
            },
            {
                'query': 'telehealth implementation in rural community health centers',
                'relevant_grant_ids': ['FQHC_SYNTH_0003', 'FQHC_SYNTH_0013', 'FQHC_SYNTH_0023'],
                'type': 'fqhc_technology'
            },
            {
                'query': 'health disparities reduction in safety-net clinics',
                'relevant_grant_ids': ['FQHC_SYNTH_0007', 'FQHC_SYNTH_0017', 'FQHC_SYNTH_0027'],
                'type': 'fqhc_equity'
            },
            {
                'query': 'behavioral health integration in primary care FQHCs',
                'relevant_grant_ids': ['FQHC_SYNTH_0009', 'FQHC_SYNTH_0019', 'FQHC_SYNTH_0029'],
                'type': 'fqhc_behavioral_health'
            }
        ]
        return fqhc_queries
    
    def interactive_demo(self):
        """Interactive demo of FAISS RAG baseline"""
        print("\n" + "="*70)
        print("💬 FAISS RAG BASELINE DEMO")
        print("="*70)
        print("Test queries against the FAISS baseline")
        print("Type 'quit' to exit, 'eval' to run evaluation")
        print("-" * 70)
        
        while True:
            query = input("\n🔍 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query.lower() == 'eval':
                self.evaluate()
                continue
            
            if not query:
                continue
            
            print(f"\n📚 FAISS Searching for: '{query}'")
            print("-" * 50)
            
            results = self.search(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']}")
                print(f"   📅 Year: {result['year']} | Institute: {result['institute']}")
                print(f"   📊 Similarity: {result['similarity']:.3f} | Boosted: {result['boosted_similarity']:.3f}")
                print(f"   🎯 FQHC Score: {result['fqhc_score']:.2f} | FQHC-focused: {result['is_fqhc_focused']}")
                print(f"   📝 Source: {result['data_source']}")
                print(f"   📝 Abstract: {result['abstract_preview']}")
            
            if not results:
                print("No results found. Try a different query.")
            
            print("\n" + "-" * 50)
    def create_proper_evaluation(self, num_queries: int = 20) -> List[Dict]:
        """
        Create proper evaluation queries that match actual documents
        This ensures ground truth IDs exist in the dataset
        """
        print(f"\n🎯 Creating proper evaluation from {len(self.data)} documents...")
        
        # Get FQHC documents
        fqhc_docs = self.data[self.data['is_fqhc_focused'] == True]
        
        if len(fqhc_docs) < num_queries:
            print(f"⚠️  Only {len(fqhc_docs)} FQHC documents available")
            num_queries = len(fqhc_docs)
        
        eval_set = []
        conditions_used = set()
        
        for i, (_, doc) in enumerate(fqhc_docs.head(num_queries).iterrows()):
            # Create query based on document content
            title = doc.get('title', 'Untitled')
            abstract = doc.get('abstract', '')
            full_text = (title + ' ' + abstract).lower()
            
            # Determine query based on content
            query = self._generate_query_from_document(doc, full_text)
            
            # Track conditions for diversity
            primary_condition = doc.get('primary_condition', 'general')
            conditions_used.add(primary_condition)
            
            eval_set.append({
                "query_id": f"Q{i+1:03d}_PROPER",
                "query": query,
                "relevant_grant_ids": [doc['grant_id']],  # CRITICAL: Use actual document ID
                "query_type": "document_based",
                "condition": primary_condition,
                "source_document": doc['grant_id'],
                "notes": f"Created from document: {doc['grant_id']}"
            })
        
        # Save evaluation
        eval_path = './phase3_results/proper_evaluation.json'
        os.makedirs('./phase3_results', exist_ok=True)
        with open(eval_path, 'w') as f:
            json.dump(eval_set, f, indent=2)
        
        print(f"✅ Created {len(eval_set)} proper evaluation queries")
        print(f"   Conditions represented: {list(conditions_used)}")
        print(f"   Saved to: {eval_path}")
        
        return eval_set
    
    def _generate_query_from_document(self, doc: Dict, full_text: str) -> str:
        """Generate a query from document content"""
        title = doc.get('title', '')
        
        # Check for specific conditions
        if 'diabetes' in full_text:
            return "diabetes management in community health settings"
        elif 'hypertension' in full_text or 'blood pressure' in full_text:
            return "hypertension control in underserved populations"
        elif 'depression' in full_text or 'mental health' in full_text:
            return "behavioral health integration in primary care"
        elif 'asthma' in full_text:
            return "asthma management in pediatric populations"
        elif 'cancer' in full_text:
            return "cancer screening in community health centers"
        elif 'hiv' in full_text:
            return "HIV prevention and care in safety-net settings"
        else:
            # Create generic query from title
            words = title.split()[:4]
            return f"{' '.join(words)} in Federally Qualified Health Centers"
    
    def evaluate_with_proper_queries(self, num_queries: int = 20) -> Dict:
        """
        Evaluate using proper queries (recommended for accurate metrics)
        """
        print("\n🧪 EVALUATION WITH PROPER GROUND TRUTH")
        print("=" * 50)
        
        # Create or load proper evaluation
        eval_path = './phase3_results/proper_evaluation.json'
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                test_queries = json.load(f)
            print(f"📋 Loaded existing proper evaluation: {len(test_queries)} queries")
        else:
            test_queries = self.create_proper_evaluation(num_queries)
        
        # Run evaluation
        return self._run_evaluation_with_queries(test_queries)
    
    def _run_evaluation_with_queries(self, test_queries: List[Dict]) -> Dict:
        """Internal method to run evaluation with given queries"""
        metrics = {
            'precision_at_1': [],
            'precision_at_3': [],
            'precision_at_5': [],
            'fqhc_alignment': [],
            'retrieval_time': [],
            'avg_similarity': []
        }
        
        for i, query_data in enumerate(test_queries):
            query = query_data.get('query', '')
            relevant_ids = set(query_data.get('relevant_grant_ids', []))
            
            results = self.search(query, top_k=5, fqhc_boost=True)
            
            retrieved_ids = [r['grant_id'] for r in results]
            
            # Calculate precision@k
            for k in [1, 3, 5]:
                top_k_ids = retrieved_ids[:k]
                relevant_in_top_k = len([id for id in top_k_ids if id in relevant_ids])
                precision = relevant_in_top_k / k if k > 0 else 0
                metrics[f'precision_at_{k}'].append(precision)
            
            # Calculate FQHC alignment
            if results:
                fqhc_scores = [r['fqhc_score'] for r in results[:3]]
                metrics['fqhc_alignment'].append(np.mean(fqhc_scores))
            
            # Record metrics
            if results:
                metrics['retrieval_time'].append(results[0]['search_time'])
                metrics['avg_similarity'].append(np.mean([r['similarity'] for r in results[:3]]))
            
            # Progress update
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(test_queries)} queries...")
        
        # Calculate averages
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
        
        print(f"\n📊 Evaluation Results:")
        print(f"   • Precision@1: {avg_metrics.get('precision_at_1', 0):.3f}")
        print(f"   • Precision@3: {avg_metrics.get('precision_at_3', 0):.3f}")
        print(f"   • Precision@5: {avg_metrics.get('precision_at_5', 0):.3f}")
        print(f"   • FQHC Alignment: {avg_metrics.get('fqhc_alignment', 0):.3f}")
        print(f"   • Avg Retrieval Time: {avg_metrics.get('retrieval_time', 0):.3f}s")
        
        return avg_metrics
    def create_proper_evaluation(self, num_queries: int = 20) -> List[Dict]:
        """
        Create proper evaluation queries that match actual documents
        This ensures ground truth IDs exist in the dataset
        """
        print(f"\n🎯 Creating proper evaluation from {len(self.data)} documents...")
        
        # Get FQHC documents
        fqhc_docs = self.data[self.data['is_fqhc_focused'] == True]
        
        if len(fqhc_docs) < num_queries:
            print(f"⚠️  Only {len(fqhc_docs)} FQHC documents available")
            num_queries = len(fqhc_docs)
        
        eval_set = []
        conditions_used = set()
        
        for i, (_, doc) in enumerate(fqhc_docs.head(num_queries).iterrows()):
            # Create query based on document content
            title = doc.get('title', 'Untitled')
            abstract = doc.get('abstract', '')
            full_text = (title + ' ' + abstract).lower()
            
            # Determine query based on content
            query = self._generate_query_from_document(doc, full_text)
            
            # Track conditions for diversity
            primary_condition = doc.get('primary_condition', 'general')
            conditions_used.add(primary_condition)
            
            eval_set.append({
                "query_id": f"Q{i+1:03d}_PROPER",
                "query": query,
                "relevant_grant_ids": [doc['grant_id']],  # CRITICAL: Use actual document ID
                "query_type": "document_based",
                "condition": primary_condition,
                "source_document": doc['grant_id'],
                "notes": f"Created from document: {doc['grant_id']}"
            })
        
        # Save evaluation
        eval_path = './phase3_results/proper_evaluation.json'
        os.makedirs('./phase3_results', exist_ok=True)
        with open(eval_path, 'w') as f:
            json.dump(eval_set, f, indent=2)
        
        print(f"✅ Created {len(eval_set)} proper evaluation queries")
        print(f"   Conditions represented: {list(conditions_used)}")
        print(f"   Saved to: {eval_path}")
        
        return eval_set
    
    def _generate_query_from_document(self, doc: Dict, full_text: str) -> str:
        """Generate a query from document content"""
        title = doc.get('title', '')
        
        # Check for specific conditions
        if 'diabetes' in full_text:
            return "diabetes management in community health settings"
        elif 'hypertension' in full_text or 'blood pressure' in full_text:
            return "hypertension control in underserved populations"
        elif 'depression' in full_text or 'mental health' in full_text:
            return "behavioral health integration in primary care"
        elif 'asthma' in full_text:
            return "asthma management in pediatric populations"
        elif 'cancer' in full_text:
            return "cancer screening in community health centers"
        elif 'hiv' in full_text:
            return "HIV prevention and care in safety-net settings"
        else:
            # Create generic query from title
            words = title.split()[:4]
            return f"{' '.join(words)} in Federally Qualified Health Centers"
    
    def evaluate_with_proper_queries(self, num_queries: int = 20) -> Dict:
        """
        Evaluate using proper queries (recommended for accurate metrics)
        """
        print("\n🧪 EVALUATION WITH PROPER GROUND TRUTH")
        print("=" * 50)
        
        # Create or load proper evaluation
        eval_path = './phase3_results/proper_evaluation.json'
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                test_queries = json.load(f)
            print(f"📋 Loaded existing proper evaluation: {len(test_queries)} queries")
        else:
            test_queries = self.create_proper_evaluation(num_queries)
        
        # Run evaluation
        return self._run_evaluation_with_queries(test_queries)
    
    def _run_evaluation_with_queries(self, test_queries: List[Dict]) -> Dict:
        """Internal method to run evaluation with given queries"""
        metrics = {
            'precision_at_1': [],
            'precision_at_3': [],
            'precision_at_5': [],
            'fqhc_alignment': [],
            'retrieval_time': [],
            'avg_similarity': []
        }
        
        for i, query_data in enumerate(test_queries):
            query = query_data.get('query', '')
            relevant_ids = set(query_data.get('relevant_grant_ids', []))
            
            results = self.search(query, top_k=5, fqhc_boost=True)
            
            retrieved_ids = [r['grant_id'] for r in results]
            
            # Calculate precision@k
            for k in [1, 3, 5]:
                top_k_ids = retrieved_ids[:k]
                relevant_in_top_k = len([id for id in top_k_ids if id in relevant_ids])
                precision = relevant_in_top_k / k if k > 0 else 0
                metrics[f'precision_at_{k}'].append(precision)
            
            # Calculate FQHC alignment
            if results:
                fqhc_scores = [r['fqhc_score'] for r in results[:3]]
                metrics['fqhc_alignment'].append(np.mean(fqhc_scores))
            
            # Record metrics
            if results:
                metrics['retrieval_time'].append(results[0]['search_time'])
                metrics['avg_similarity'].append(np.mean([r['similarity'] for r in results[:3]]))
            
            # Progress update
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(test_queries)} queries...")
        
        # Calculate averages
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
        
        print(f"\n📊 Evaluation Results:")
        print(f"   • Precision@1: {avg_metrics.get('precision_at_1', 0):.3f}")
        print(f"   • Precision@3: {avg_metrics.get('precision_at_3', 0):.3f}")
        print(f"   • Precision@5: {avg_metrics.get('precision_at_5', 0):.3f}")
        print(f"   • FQHC Alignment: {avg_metrics.get('fqhc_alignment', 0):.3f}")
        print(f"   • Avg Retrieval Time: {avg_metrics.get('retrieval_time', 0):.3f}s")
        
        return avg_metrics
def update_phase3_with_enhanced_data():
    """Update Phase 3 to use enhanced dataset"""
    
    # Copy enhanced dataset to Phase 3 location
    import shutil
    
    source = './enhanced_phase3_data/fqhc_enhanced_dataset.csv'
    destination = './phase3_data/enhanced_fqhc_dataset.csv'
    
    if os.path.exists(source):
        shutil.copy2(source, destination)
        print(f"✅ Updated Phase 3 dataset with enhanced data")
        print(f"   From: {source}")
        print(f"   To: {destination}")
        
        # Verify
        data = pd.read_csv(destination)
        print(f"   New dataset: {len(data)} documents")
        print(f"   FQHC-focused: {data['is_fqhc_focused'].sum()}")
        
        # Check for primary_condition field
        if 'primary_condition' in data.columns:
            conditions = data['primary_condition'].dropna().unique()
            print(f"   Conditions covered: {list(conditions)}")
    else:
        print(f"❌ Enhanced dataset not found at {source}")

# Run the update
update_phase3_with_enhanced_data()

def create_proper_evaluation(self, num_queries: int = 20) -> List[Dict]:
        """
        Create proper evaluation queries that match actual documents
        This ensures ground truth IDs exist in the dataset
        """
        print(f"\n🎯 Creating proper evaluation from {len(self.data)} documents...")
        
        # Get FQHC documents
        fqhc_docs = self.data[self.data['is_fqhc_focused'] == True]
        
        if len(fqhc_docs) < num_queries:
            print(f"⚠️  Only {len(fqhc_docs)} FQHC documents available")
            num_queries = len(fqhc_docs)
        
        eval_set = []
        conditions_used = set()
        
        for i, (_, doc) in enumerate(fqhc_docs.head(num_queries).iterrows()):
            # Create query based on document content
            title = doc.get('title', 'Untitled')
            abstract = doc.get('abstract', '')
            full_text = (title + ' ' + abstract).lower()
            
            # Determine query based on content
            query = self._generate_query_from_document(doc, full_text)
            
            # Track conditions for diversity
            primary_condition = doc.get('primary_condition', 'general')
            conditions_used.add(primary_condition)
            
            eval_set.append({
                "query_id": f"Q{i+1:03d}_PROPER",
                "query": query,
                "relevant_grant_ids": [doc['grant_id']],  # CRITICAL: Use actual document ID
                "query_type": "document_based",
                "condition": primary_condition,
                "source_document": doc['grant_id'],
                "notes": f"Created from document: {doc['grant_id']}"
            })
        
        # Save evaluation
        eval_path = './phase3_results/proper_evaluation.json'
        os.makedirs('./phase3_results', exist_ok=True)
        with open(eval_path, 'w') as f:
            json.dump(eval_set, f, indent=2)
        
        print(f"✅ Created {len(eval_set)} proper evaluation queries")
        print(f"   Conditions represented: {list(conditions_used)}")
        print(f"   Saved to: {eval_path}")
        
        return eval_set
    
def _generate_query_from_document(self, doc: Dict, full_text: str) -> str:
        """Generate a query from document content"""
        title = doc.get('title', '')
        
        # Check for specific conditions
        if 'diabetes' in full_text:
            return "diabetes management in community health settings"
        elif 'hypertension' in full_text or 'blood pressure' in full_text:
            return "hypertension control in underserved populations"
        elif 'depression' in full_text or 'mental health' in full_text:
            return "behavioral health integration in primary care"
        elif 'asthma' in full_text:
            return "asthma management in pediatric populations"
        elif 'cancer' in full_text:
            return "cancer screening in community health centers"
        elif 'hiv' in full_text:
            return "HIV prevention and care in safety-net settings"
        else:
            # Create generic query from title
            words = title.split()[:4]
            return f"{' '.join(words)} in Federally Qualified Health Centers"
    
def evaluate_with_proper_queries(self, num_queries: int = 20) -> Dict:
        """
        Evaluate using proper queries (recommended for accurate metrics)
        """
        print("\n🧪 EVALUATION WITH PROPER GROUND TRUTH")
        print("=" * 50)
        
        # Create or load proper evaluation
        eval_path = './phase3_results/proper_evaluation.json'
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                test_queries = json.load(f)
            print(f"📋 Loaded existing proper evaluation: {len(test_queries)} queries")
        else:
            test_queries = self.create_proper_evaluation(num_queries)
        
        # Run evaluation
        return self._run_evaluation_with_queries(test_queries)
    
def _run_evaluation_with_queries(self, test_queries: List[Dict]) -> Dict:
        """Internal method to run evaluation with given queries"""
        metrics = {
            'precision_at_1': [],
            'precision_at_3': [],
            'precision_at_5': [],
            'fqhc_alignment': [],
            'retrieval_time': [],
            'avg_similarity': []
        }
        
        for i, query_data in enumerate(test_queries):
            query = query_data.get('query', '')
            relevant_ids = set(query_data.get('relevant_grant_ids', []))
            
            results = self.search(query, top_k=5, fqhc_boost=True)
            
            retrieved_ids = [r['grant_id'] for r in results]
            
            # Calculate precision@k
            for k in [1, 3, 5]:
                top_k_ids = retrieved_ids[:k]
                relevant_in_top_k = len([id for id in top_k_ids if id in relevant_ids])
                precision = relevant_in_top_k / k if k > 0 else 0
                metrics[f'precision_at_{k}'].append(precision)
            
            # Calculate FQHC alignment
            if results:
                fqhc_scores = [r['fqhc_score'] for r in results[:3]]
                metrics['fqhc_alignment'].append(np.mean(fqhc_scores))
            
            # Record metrics
            if results:
                metrics['retrieval_time'].append(results[0]['search_time'])
                metrics['avg_similarity'].append(np.mean([r['similarity'] for r in results[:3]]))
            
            # Progress update
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(test_queries)} queries...")
        
        # Calculate averages
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
        
        print(f"\n📊 Evaluation Results:")
        print(f"   • Precision@1: {avg_metrics.get('precision_at_1', 0):.3f}")
        print(f"   • Precision@3: {avg_metrics.get('precision_at_3', 0):.3f}")
        print(f"   • Precision@5: {avg_metrics.get('precision_at_5', 0):.3f}")
        print(f"   • FQHC Alignment: {avg_metrics.get('fqhc_alignment', 0):.3f}")
        print(f"   • Avg Retrieval Time: {avg_metrics.get('retrieval_time', 0):.3f}s")
        
        return avg_metrics

def visualize_phase3_results(rag_system, evaluation_metrics: Dict):
    """Visualize Phase 3 results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase 3: FAISS RAG Baseline Results', fontsize=16, fontweight='bold')
        
        data = rag_system.data
        
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
        ax.set_ylim(0, max(counts) * 1.1)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{count}', ha='center', va='bottom')
        
        # 2. FQHC score distribution
        ax = axes[0, 1]
        if 'fqhc_score' in data.columns:
            ax.hist(data['fqhc_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('FQHC Relevance Score')
            ax.set_ylabel('Frequency')
            ax.set_title('FQHC Relevance Distribution')
            ax.axvline(x=data['fqhc_score'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {data["fqhc_score"].mean():.2f}')
            ax.legend()
        
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
            ax.set_title('FAISS RAG Evaluation Metrics')
            ax.set_ylim(0, 1)
            
            for bar, value in zip(bars, eval_metrics.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Retrieval performance
        ax = axes[1, 0]
        if evaluation_metrics and 'avg_similarity' in evaluation_metrics:
            perf_metrics = {
                'Avg Similarity': evaluation_metrics['avg_similarity'],
                'Retrieval Time': evaluation_metrics.get('retrieval_time', 0)
            }
            
            x = range(len(perf_metrics))
            width = 0.35
            
            ax.bar(x, [perf_metrics['Avg Similarity']], width, label='Similarity', color='lightblue')
            ax.set_xlabel('Metric')
            ax.set_ylabel('Similarity Score')
            ax.set_title('Retrieval Performance')
            ax.set_xticks([0])
            ax.set_xticklabels(['Avg Similarity'])
            ax.set_ylim(0, 1)
            
            # Add retrieval time as text
            ax.text(0.5, 0.8, f"Avg Time: {perf_metrics['Retrieval Time']:.3f}s", 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        # 5. Model and system info
        ax = axes[1, 1]
        ax.axis('off')
        
        system_info = [
            ['Model', str(rag_system.model).split('/')[-1][:20] if rag_system.model else 'N/A'],
            ['Documents', str(len(data))],
            ['FQHC-focused', str(data['is_fqhc_focused'].sum())],
            ['Vector Dimensions', str(rag_system.embeddings.shape[1] if rag_system.embeddings is not None else 'N/A')],
            ['FAISS Index Size', str(rag_system.index.ntotal if rag_system.index is not None else 'N/A')],
            ['Retrieval Method', 'FAISS + FQHC Boost']
        ]
        
        table = ax.table(cellText=system_info, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title('System Configuration')
        
        # 6. Baseline comparison preview
        ax = axes[1, 2]
        ax.axis('off')
        
        comparison_text = [
            "📊 FAISS BASELINE (Phase 3)",
            "",
            "Strengths:",
            "• Fast vector similarity",
            "• Good semantic matching",
            "• Simple implementation",
            "",
            "Limitations (for Phase 4 comparison):",
            "• No keyword search",
            "• Limited metadata filtering",
            "• No hybrid search (α tuning)",
            "",
            "👉 Compare with Phase 4 (Weaviate)"
        ]
        
        ax.text(0.5, 0.5, '\n'.join(comparison_text), 
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs('./phase3_results', exist_ok=True)
        plt.savefig('./phase3_results/phase3_baseline_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("⚠️  Visualization libraries not available")

def run_phase3_enhanced():
    """
    RUN ENHANCED PHASE 3: FAISS RAG BASELINE
    """
    print("\n" + "="*70)
    print("🚀 STARTING PHASE 3: FAISS RAG BASELINE")
    print("="*70)
    
    # Step 1: Initialize enhanced RAG system
    print("\n🤖 STEP 1: INITIALIZING FQHC RAG BASELINE")
    print("-" * 50)
    
    try:
        rag_system = FQHCRAGBaseline()
        print("✅ FAISS RAG Baseline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return None
    
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
        "phase": "phase3_faiss_rag_baseline",
        "timestamp": datetime.now().isoformat(),
        "model": str(rag_system.model).split('/')[-1] if rag_system.model else "unknown",
        "dataset_stats": {
            "total_documents": len(rag_system.data),
            "fqhc_focused": int(rag_system.data['is_fqhc_focused'].sum()),
            "synthetic": int((rag_system.data['data_source'] == 'synthetic_fqhc').sum()),
            "phase2_nih": int((rag_system.data['data_source'] == 'phase2_nih').sum())
        },
        "evaluation_metrics": evaluation_metrics,
        "system_info": {
            "vector_dimensions": rag_system.embeddings.shape[1] if rag_system.embeddings is not None else None,
            "faiss_index_size": rag_system.index.ntotal if rag_system.index is not None else None,
            "fqhc_boosting_enabled": True
        },
        "baseline_notes": [
            "This serves as baseline for Phase 4 (Weaviate) comparison",
            "FAISS provides pure vector search without keyword capabilities",
            "FQHC boosting implemented to prioritize FQHC-relevant documents",
            "Compare Precision@K and FQHC alignment with Phase 4 results"
        ]
    }
    
    os.makedirs('./phase3_results', exist_ok=True)
    with open('./phase3_results/phase3_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ PHASE 3 BASELINE COMPLETE!")
    print("="*70)
    print("\n📁 Results saved to ./phase3_results/:")
    print("  • phase3_baseline_results.json")
    print("  • phase3_baseline_results.png")
    
    print("\n🎯 BASELINE ESTABLISHED:")
    print(f"   1. Enhanced dataset: {len(rag_system.data)} documents")
    print(f"   2. {rag_system.data['is_fqhc_focused'].sum()} FQHC-focused documents")
    print(f"   3. FAISS index with {rag_system.index.ntotal if rag_system.index else 0} vectors")
    
    if evaluation_metrics:
        print(f"   4. Precision@3: {evaluation_metrics.get('precision_at_3', 0):.3f}")
        print(f"   5. FQHC Alignment: {evaluation_metrics.get('fqhc_alignment', 0):.3f}")
        print(f"   6. Avg Retrieval Time: {evaluation_metrics.get('retrieval_time', 0):.3f}s")
    
    print("\n🚀 READY FOR PHASE 4 COMPARISON!")
    print("   Next: Run Phase 4 (Weaviate) with same enhanced dataset")
    
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
    
    # Run enhanced Phase 3
    print("\n" + "="*70)
    print("🚀 PHASE 3: FAISS RAG BASELINE WITH PROPER EVALUATION")
    print("="*70)
    
    # Option A: Run with proper evaluation (recommended)
    results = create_and_run_proper_evaluation()
    
    # Option B: Or run the original flow
    # rag_system, results = run_phase3_enhanced()
    
    if rag_system:
        # Interactive demo
        print("\n" + "="*70)
        print("🎮 INTERACTIVE FAISS BASELINE DEMO")
        print("="*70)
        
        rag_system.interactive_demo()
        
        print("\n" + "="*70)
        print("🎯 BASELINE ESTABLISHED FOR PHASE 4 COMPARISON")
        print("="*70)
        print(f"\n📊 Your FAISS Baseline Metrics:")
        print(f"   • Precision@3: {results.get('precision_at_3', 0):.3f}")
        print(f"   • FQHC Alignment: {results.get('fqhc_alignment', 0):.3f}")
        print(f"   • Avg Retrieval Time: {results.get('retrieval_time', 0):.3f}s")
        print("\n🚀 Ready for Phase 4 (Weaviate) comparison!")
        print("="*70)