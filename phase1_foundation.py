# ============================================================================
# 📁 phase1_foundation.py - WEEKS 1-3: COMPONENT TESTS
# ============================================================================

"""
PHASE 1: TEST BASIC RAG COMPONENTS
EDIT EACH TEST FUNCTION TO ADD CUSTOM LOGIC
"""

print("="*70)
print("🎯 PHASE 1: FOUNDATION TESTS (Weeks 1-3)")
print("="*70)

import sys
sys.path.append('.')  # For importing config and utils

from config import RAG_CONFIG
from utils import logger, DataProcessor
import pandas as pd
import numpy as np
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple  
from collections import Counter

# EDIT HERE: Import your visualization library preference
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Visualization libraries not installed. Run: !pip install matplotlib seaborn")

# ============ TEST DATA GENERATION ============

class TestDataGenerator:
    """Generate synthetic test data for Phase 1"""
    
    @staticmethod
    def create_sample_grants(num_grants: int = 5) -> pd.DataFrame:
        """
        CREATE SAMPLE GRANT DATA FOR TESTING
        Now includes relevance labels for evaluation
        """
        grants = []
        
        # Sample grant 1: FQHC-focused diabetes prevention
        grants.append({
            "grant_id": "R01MD123456",
            "title": "Community Health Worker Diabetes Prevention in Urban FQHCs",
            "abstract": """This R01 evaluates a CHW-led diabetes prevention program for Latino populations 
            in Federally Qualified Health Centers. Intervention includes culturally-adapted nutrition 
            education and regular health screenings in underserved communities. Randomized controlled trial 
            with 500 participants across 5 urban FQHCs.""",
            "full_text": """SPECIFIC AIMS: Implement diabetes prevention in 5 urban FQHCs serving 
            low-income Latino communities. Train 20 community health workers to deliver the intervention.
            
            BACKGROUND: Diabetes disproportionately affects Latino populations with rates 50% higher 
            than non-Hispanic whites. FQHCs serve as critical access points but lack prevention resources.
            
            METHODS: Randomized controlled trial with 500 participants across 5 FQHCs. Intervention 
            group receives weekly CHW sessions focusing on nutrition and physical activity.
            
            SIGNIFICANCE: Addresses health disparities through community-based implementation in 
            safety-net settings with potential for broad dissemination.""",
            "year": 2024,
            "institute": "NIMHD",
            "is_fqhc_focused": True,
            "primary_topic": "diabetes prevention",
            "population": "Latino, urban",
            "intervention": "CHW, culturally-adapted"
        })
        
        # Sample grant 2: FQHC behavioral health integration
        grants.append({
            "grant_id": "R34MH123457",
            "title": "Integrated Behavioral Health in Primary Care for Medicaid Patients",
            "abstract": "Implementation study of depression and anxiety screening in FQHC primary care clinics serving Medicaid populations. Stepped-wedge design across 10 clinics with measurement of screening rates and treatment initiation.",
            "full_text": """SPECIFIC AIMS: Implement PHQ-9 screening in 10 urban FQHCs. Train primary care providers in depression management.
            
            BACKGROUND: Mental health disparities persist in Medicaid populations with limited access to specialty care.
            
            METHODS: Stepped-wedge design across 10 clinics. Measure screening rates and treatment initiation.
            
            SIGNIFICANCE: First large-scale implementation of integrated behavioral health in Medicaid-focused FQHCs.""",
            "year": 2023,
            "institute": "NIMH",
            "is_fqhc_focused": True,
            "primary_topic": "behavioral health integration",
            "population": "Medicaid",
            "intervention": "screening, brief intervention"
        })
        
        # Sample grant 3: FQHC social determinants
        grants.append({
            "grant_id": "R01HD123458", 
            "title": "Social Determinants Screening in Pediatric FQHCs",
            "abstract": "Study of food insecurity and housing screening in pediatric visits at community health centers. Cluster randomized trial with 2000 pediatric patients comparing intervention vs usual care.",
            "full_text": """SPECIFIC AIMS: Implement screening protocol in 8 pediatric FQHC clinics. Develop referral network.
            
            BACKGROUND: Social determinants account for 60% of health outcomes but are rarely screened in pediatric care.
            
            METHODS: Cluster randomized trial with 2000 pediatric patients. Compare intervention vs usual care.
            
            SIGNIFICANCE: First pediatric-focused social determinants intervention in FQHC settings.""",
            "year": 2024,
            "institute": "NICHD",
            "is_fqhc_focused": True,
            "primary_topic": "social determinants",
            "population": "pediatric",
            "intervention": "screening, referral"
        })
        
        # Sample grant 4: NON-FQHC basic science (control)
        grants.append({
            "grant_id": "R01AG123459",
            "title": "Molecular Mechanisms of Cellular Senescence",
            "abstract": "Basic science research on genetic pathways in aging neurons using single-cell RNA sequencing. Identifies novel therapeutic targets for age-related cognitive decline through analysis of 50,000 neurons.",
            "full_text": """SPECIFIC AIMS: Identify senescence factors in hippocampal neurons. Validate with CRISPR screening.
            
            BACKGROUND: Cellular senescence contributes to neurodegeneration but mechanisms are poorly understood.
            
            METHODS: Single-cell RNA sequencing of 50,000 neurons from young and old mice.
            
            SIGNIFICANCE: Identifies novel therapeutic targets for age-related cognitive decline.""",
            "year": 2024,
            "institute": "NIA",
            "is_fqhc_focused": False,
            "primary_topic": "basic neuroscience",
            "population": "animal models",
            "intervention": "molecular analysis"
        })
        
        # Sample grant 5: FQHC cancer screening
        grants.append({
            "grant_id": "R01CA123460",
            "title": "Cancer Screening Navigation in Rural FQHCs",
            "abstract": "Patient navigation program for breast and cervical cancer screening in Appalachian FQHCs. Pragmatic randomized trial with 1200 women overdue for screenings, training lay health advisors as patient navigators.",
            "full_text": """SPECIFIC AIMS: Train lay health advisors as patient navigators. Implement across 12 rural FQHCs.
            
            BACKGROUND: Rural populations face unique barriers to cancer screening including transportation and cost.
            
            METHODS: Pragmatic randomized trial with 1200 women overdue for screenings.
            
            SIGNIFICANCE: Addresses critical gaps in cancer prevention for rural underserved women.""",
            "year": 2023,
            "institute": "NCI",
            "is_fqhc_focused": True,
            "primary_topic": "cancer screening",
            "population": "rural women",
            "intervention": "patient navigation"
        })
        
        return pd.DataFrame(grants[:num_grants])
    
    @staticmethod
    def create_labeled_test_queries() -> List[Dict]:
        """
        CREATE LABELED TEST QUERIES WITH RELEVANCE JUDGMENTS
        Critical for proper evaluation
        """
        queries = [
            {
                "query_id": "Q1_FQHC_DIABETES",
                "text": """Diabetes prevention strategies using community health workers in Federally Qualified Health Centers serving Latino populations""",
                "relevant_grants": ["R01MD123456"],  # Highly relevant
                "somewhat_relevant": ["R01CA123460"],  # Somewhat (different focus)
                "not_relevant": ["R01AG123459"],  # Not relevant (basic science)
                "expected_themes": ["diabetes", "chw", "fqhc", "latino", "prevention"],
                "query_type": "specific_fqhc_intervention"
            },
            {
                "query_id": "Q2_BEHAVIORAL_HEALTH",
                "text": """Integrated behavioral health services in primary care settings with depression screening for Medicaid patients""",
                "relevant_grants": ["R34MH123457"],  # Highly relevant
                "somewhat_relevant": ["R01HD123458"],  # Somewhat (different focus)
                "not_relevant": ["R01AG123459"],  # Not relevant
                "expected_themes": ["behavioral health", "screening", "medicaid", "primary care"],
                "query_type": "implementation_science"
            },
            {
                "query_id": "Q3_SOCIAL_DETERMINANTS",
                "text": """Screening for social determinants of health like food insecurity in pediatric community health settings""",
                "relevant_grants": ["R01HD123458"],  # Highly relevant
                "somewhat_relevant": ["R01MD123456", "R34MH123457"],  # Somewhat (health disparities)
                "not_relevant": ["R01AG123459"],  # Not relevant
                "expected_themes": ["social determinants", "pediatric", "screening", "fqhc"],
                "query_type": "preventive_services"
            },
            {
                "query_id": "Q4_RURAL_HEALTH",
                "text": """Health interventions addressing barriers to care in rural underserved populations""",
                "relevant_grants": ["R01CA123460"],  # Highly relevant
                "somewhat_relevant": ["R01MD123456"],  # Somewhat (underserved)
                "not_relevant": ["R01AG123459"],  # Not relevant
                "expected_themes": ["rural", "underserved", "barriers", "navigation"],
                "query_type": "geographic_focus"
            },
            {
                "query_id": "Q5_GENERAL_FQHC",
                "text": """Federally Qualified Health Center interventions for health disparities""",
                "relevant_grants": ["R01MD123456", "R34MH123457", "R01HD123458", "R01CA123460"],  # All FQHC
                "somewhat_relevant": [],  # None
                "not_relevant": ["R01AG123459"],  # Not FQHC
                "expected_themes": ["fqhc", "health disparities", "community health"],
                "query_type": "general_fqhc"
            }
        ]
        return queries

# ============ SEMANTIC CHUNKING STRATEGIES ============

class SemanticChunker:
    """
    SEMANTIC CHUNKING STRATEGIES FOR GRANT TEXT
    No fixed word counts - chunk by meaning
    """
    
    # FQHC keywords for relevance detection
    FQHC_KEYWORDS = [
        'fqhc', 'federally qualified health center', 'community health center',
        'safety-net', 'underserved', 'health disparities', 'medically underserved',
        'low-income', 'uninsured', 'medicaid', 'primary care', 'preventive care'
    ]
    
    @staticmethod
    def chunk_abstract_whole(abstract: str) -> List[Dict]:
        """
        KEEP ABSTRACT WHOLE - Most coherent for grants
        Abstracts are already concise summaries (150-300 words)
        """
        return [{
            "text": abstract.strip(),
            "chunk_type": "whole_abstract",
            "is_complete": True,
            "coherence_score": 1.0,
            "word_count": len(abstract.split())
        }]
    
    @staticmethod
    def chunk_by_sections(full_text: str) -> List[Dict]:
        """
        CHUNK BY GRANT SECTIONS
        Detect section headers and keep sections intact
        """
        sections = []
        current_section = {"name": "header", "text": ""}
        
        lines = full_text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect section headers (case-insensitive)
            section_patterns = {
                'specific_aims': r'^SPECIFIC AIMS|^AIMS|^RESEARCH AIMS',
                'background': r'^BACKGROUND|^SIGNIFICANCE',
                'methods': r'^METHODS|^RESEARCH DESIGN|^APPROACH',
                'results': r'^RESULTS|^FINDINGS',
                'discussion': r'^DISCUSSION|^CONCLUSIONS',
                'significance': r'^SIGNIFICANCE|^IMPACT'
            }
            
            section_found = False
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # Save previous section if exists
                    if current_section["text"].strip():
                        sections.append({
                            "text": current_section["text"].strip(),
                            "chunk_type": f"section_{current_section['name']}",
                            "section_name": current_section["name"],
                            "is_complete": True,
                            "word_count": len(current_section["text"].split())
                        })
                    
                    # Start new section
                    current_section = {"name": section_name, "text": line_stripped + "\n"}
                    section_found = True
                    break
            
            if not section_found:
                current_section["text"] += line + "\n"
        
        # Add final section
        if current_section["text"].strip():
            sections.append({
                "text": current_section["text"].strip(),
                "chunk_type": f"section_{current_section['name']}",
                "section_name": current_section["name"],
                "is_complete": True,
                "word_count": len(current_section["text"].split())
            })
        
        return sections
    
    @staticmethod
    def chunk_by_topic_shift(text: str, min_chunk_words: int = 50) -> List[Dict]:
        """
        CHUNK BY TOPIC SHIFTS USING SIMPLE HEURISTICS
        Look for topic changes in the text
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            return [{
                "text": text,
                "chunk_type": "single_topic",
                "is_complete": True,
                "word_count": len(text.split())
            }]
        
        chunks = []
        current_chunk = []
        current_topics = set()
        
        for para in paragraphs:
            para_words = para.split()
            
            # Extract topics from paragraph
            para_topics = SemanticChunker._extract_paragraph_topics(para)
            
            # Check if topic shift or chunk getting too large
            if (current_topics and not para_topics.intersection(current_topics) and 
                sum(len(p.split()) for p in current_chunk) >= min_chunk_words):
                
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "chunk_type": "topic_chunk",
                    "topics": list(current_topics),
                    "is_complete": len(current_topics) > 0,
                    "word_count": len(chunk_text.split())
                })
                
                # Start new chunk
                current_chunk = [para]
                current_topics = para_topics
            else:
                current_chunk.append(para)
                current_topics.update(para_topics)
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "chunk_type": "topic_chunk",
                "topics": list(current_topics),
                "is_complete": len(current_topics) > 0,
                "word_count": len(chunk_text.split())
            })
        
        return chunks
    
    @staticmethod
    def _extract_paragraph_topics(paragraph: str) -> Set[str]:
        """Extract key topics from a paragraph"""
        words = paragraph.lower().split()
        
        # FQHC-specific topics
        fqhc_topics = {
            'diabetes', 'prevention', 'screening', 'intervention',
            'community', 'health', 'disparities', 'underserved',
            'primary care', 'behavioral', 'mental', 'chronic',
            'implementation', 'randomized', 'trial', 'qualitative',
            'culturally', 'adapted', 'latino', 'medicaid', 'rural',
            'pediatric', 'geriatric', 'navigation', 'chw', 'telehealth'
        }
        
        # Find topics present in paragraph
        found_topics = set()
        for topic in fqhc_topics:
            if topic in paragraph.lower():
                found_topics.add(topic)
        
        return found_topics

# ============ WEEK 1: SEMANTIC CHUNKING TESTS ============

def test_semantic_chunking_strategies(grant_texts: List[str], grant_metadata: List[Dict] = None):
    """
    TEST SEMANTIC CHUNKING STRATEGIES (NO FIXED WORD COUNTS)
    """
    logger.log_experiment_start("Week1_SemanticChunking", RAG_CONFIG["phase1"])
    
    chunker = SemanticChunker()
    results = []
    
    strategies = [
        ("abstract_whole", chunker.chunk_abstract_whole),
        ("section_based", chunker.chunk_by_sections),
        ("topic_based", chunker.chunk_by_topic_shift)
    ]
    
    for strategy_name, chunk_func in strategies:
        print(f"\n🔧 Testing chunking strategy: {strategy_name}")
        
        all_chunks = []
        chunk_metrics = {
            "coherence_scores": [],
            "completeness_scores": [],
            "word_counts": [],
            "fqhc_relevance_scores": []
        }
        
        start_time = time.time()
        
        for i, text in enumerate(grant_texts):
            # Apply chunking strategy
            chunks = chunk_func(text)
            all_chunks.extend(chunks)
            
            # Calculate chunk quality metrics
            for chunk in chunks:
                # Coherence: does chunk make sense alone?
                coherence = SemanticChunker._calculate_chunk_coherence(chunk["text"])
                
                # Completeness: is thought complete?
                completeness = SemanticChunker._calculate_chunk_completeness(chunk["text"])
                
                # FQHC relevance
                fqhc_relevance = SemanticChunker._calculate_fqhc_relevance(chunk["text"])
                
                chunk_metrics["coherence_scores"].append(coherence)
                chunk_metrics["completeness_scores"].append(completeness)
                chunk_metrics["word_counts"].append(chunk["word_count"])
                chunk_metrics["fqhc_relevance_scores"].append(fqhc_relevance)
        
        processing_time = time.time() - start_time
        
        # Store results
        results.append({
            "strategy_name": strategy_name,
            "total_chunks": len(all_chunks),
            "avg_coherence": np.mean(chunk_metrics["coherence_scores"]),
            "avg_completeness": np.mean(chunk_metrics["completeness_scores"]),
            "avg_word_count": np.mean(chunk_metrics["word_counts"]),
            "std_word_count": np.std(chunk_metrics["word_counts"]),
            "avg_fqhc_relevance": np.mean(chunk_metrics["fqhc_relevance_scores"]),
            "processing_time_seconds": processing_time,
            "sample_chunks": all_chunks[:2] if all_chunks else []
        })
        
        print(f"  Created {len(all_chunks)} chunks")
        print(f"  Avg coherence: {results[-1]['avg_coherence']:.2f}")
        print(f"  Avg completeness: {results[-1]['avg_completeness']:.2f}")
        print(f"  Avg FQHC relevance: {results[-1]['avg_fqhc_relevance']:.2f}")
    
    return results

# Helper methods for SemanticChunker
@staticmethod
def _calculate_chunk_coherence(chunk_text: str) -> float:
    """Calculate if chunk makes sense alone"""
    sentences = [s.strip() for s in chunk_text.split('.') if s.strip()]
    
    if len(sentences) < 2:
        return 0.5  # Single sentence chunks are somewhat coherent
    
    # Check if sentences are related (simple heuristic)
    words_per_sentence = [len(s.split()) for s in sentences]
    if min(words_per_sentence) < 5:
        return 0.4  # Very short sentences may lack coherence
    
    # Check for narrative flow (simple)
    transition_words = {'however', 'therefore', 'furthermore', 'additionally', 'conversely'}
    has_transitions = any(word in chunk_text.lower() for word in transition_words)
    
    return 0.7 if has_transitions else 0.6

@staticmethod
def _calculate_chunk_completeness(chunk_text: str) -> float:
    """Calculate if chunk expresses complete thought"""
    # Check if chunk ends with proper punctuation
    if chunk_text.strip().endswith(('.', '!', '?')):
        punctuation_score = 1.0
    elif chunk_text.strip().endswith(','):
        punctuation_score = 0.5
    else:
        punctuation_score = 0.3
    
    # Check for incomplete sentences
    incomplete_indicators = ['e.g.', 'i.e.', 'etc.', '...', '•', '- ']
    has_incomplete = any(indicator in chunk_text for indicator in incomplete_indicators)
    
    if has_incomplete:
        completeness_score = 0.5
    else:
        completeness_score = 0.8
    
    return (punctuation_score + completeness_score) / 2

@staticmethod
def _calculate_fqhc_relevance(chunk_text: str) -> float:
    """Calculate FQHC relevance score"""
    chunk_lower = chunk_text.lower()
    
    # Count FQHC keyword matches
    fqhc_keywords = SemanticChunker.FQHC_KEYWORDS
    matches = sum(1 for keyword in fqhc_keywords if keyword in chunk_lower)
    
    # Normalize to 0-1 scale (cap at 3 matches)
    return min(matches / 3, 1.0)

# Add these methods to SemanticChunker class
SemanticChunker._calculate_chunk_coherence = _calculate_chunk_coherence
SemanticChunker._calculate_chunk_completeness = _calculate_chunk_completeness
SemanticChunker._calculate_fqhc_relevance = _calculate_fqhc_relevance

# ============ WEEK 2: BIOMEDICAL EMBEDDING TESTS ============

def test_biomedical_embeddings(grant_chunks: List[str], test_queries: List[Dict]):
    """
    TEST BIOMEDICAL EMBEDDING MODELS WITH FQHC CONTENT
    """
    logger.log_experiment_start("Week2_BiomedicalEmbeddings", RAG_CONFIG["phase1"])
    
    # Define biomedical models to test
    biomedical_models = [
        {
            "name": "all-mpnet-base-v2",
            "type": "general",
            "dimensions": 768,
            "description": "General purpose (baseline)"
        },
        {
            "name": "BAAI/bge-large-en",
            "type": "retrieval",
            "dimensions": 1024, 
            "description": "SOTA for retrieval"
        },
        {
            "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "type": "biomedical",
            "dimensions": 768,
            "description": "PubMed abstracts"
        },
        {
            "name": "emilyalsentzer/Bio_ClinicalBERT",
            "type": "clinical", 
            "dimensions": 768,
            "description": "Clinical notes"
        }
    ]
    
    results = []
    
    for model_info in biomedical_models:
        print(f"\n🔬 Testing embedding model: {model_info['name']}")
        print(f"   Type: {model_info['type']} - {model_info['description']}")
        
        try:
            # Load model
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_info["name"])
            
            # Test encoding speed
            encode_times = []
            test_texts = [q["text"] for q in test_queries[:3]] + grant_chunks[:3]
            
            for text in test_texts:
                start_time = time.time()
                _ = model.encode(text)
                encode_times.append(time.time() - start_time)
            
            # Test semantic preservation with FQHC content
            fqhc_queries = [q["text"] for q in test_queries]
            fqhc_chunks = grant_chunks[:20]  # Sample chunks
            
            query_embeddings = model.encode(fqhc_queries)
            chunk_embeddings = model.encode(fqhc_chunks)
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embeddings, chunk_embeddings)
            
            # Calculate FQHC-specific similarity
            fqhc_similarity_score = SemanticChunker._calculate_fqhc_similarity(
                model, fqhc_queries, fqhc_chunks
            )
            
            results.append({
                "model_name": model_info["name"],
                "model_type": model_info["type"],
                "dimensions": model_info["dimensions"],
                "avg_encode_time": np.mean(encode_times),
                "max_similarity": similarities.max(),
                "mean_similarity": similarities.mean(),
                "fqhc_similarity_score": fqhc_similarity_score,
                "status": "SUCCESS"
            })
            
            print(f"  ✓ Encode time: {results[-1]['avg_encode_time']:.4f}s")
            print(f"  ✓ FQHC similarity: {fqhc_similarity_score:.3f}")
            
        except Exception as e:
            print(f"❌ Error with {model_info['name']}: {str(e)[:100]}")
            results.append({
                "model_name": model_info["name"],
                "status": f"FAILED: {str(e)[:50]}",
                "avg_encode_time": None,
                "fqhc_similarity_score": None
            })
    
    return results

@staticmethod
def _calculate_fqhc_similarity(model, queries: List[str], chunks: List[str]) -> float:
    """Calculate similarity specifically for FQHC content"""
    # Extract FQHC terms
    fqhc_terms = SemanticChunker.FQHC_KEYWORDS
    
    # Find chunks and queries with FQHC content
    fqhc_queries = []
    fqhc_chunks = []
    
    for query in queries:
        if any(term in query.lower() for term in fqhc_terms):
            fqhc_queries.append(query)
    
    for chunk in chunks:
        if any(term in chunk.lower() for term in fqhc_terms):
            fqhc_chunks.append(chunk)
    
    if not fqhc_queries or not fqhc_chunks:
        return 0.0
    
    # Calculate embeddings and similarity
    query_embeddings = model.encode(fqhc_queries)
    chunk_embeddings = model.encode(fqhc_chunks)
    
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embeddings, chunk_embeddings)
    
    return float(similarities.mean())

SemanticChunker._calculate_fqhc_similarity = _calculate_fqhc_similarity

# ============ WEEK 3: RETRIEVAL WITH EVALUATION ============

def test_retrieval_with_evaluation(grant_chunks: List[Dict], 
                                  test_queries: List[Dict],
                                  embedding_model_name: str = "all-mpnet-base-v2"):
    """
    TEST RETRIEVAL STRATEGIES WITH PROPER EVALUATION
    Uses labeled test data
    """
    logger.log_experiment_start("Week3_RetrievalEvaluation", RAG_CONFIG["phase1"])
    
    # Load embedding model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_model_name)
    
    # Extract chunk texts and metadata
    chunk_texts = [chunk.get("text", "") for chunk in grant_chunks]
    chunk_metadata = grant_chunks  # Keep metadata for evaluation
    
    # Encode once
    query_texts = [q["text"] for q in test_queries]
    query_embeddings = model.encode(query_texts)
    chunk_embeddings = model.encode(chunk_texts)
    
    results = []
    
    # Test different retrieval strategies
    strategies = [
        {
            "name": "semantic_similarity",
            "function": retrieve_with_fqhc_boost,
            "params": {"fqhc_boost": 1.5}
        },
        {
            "name": "max_marginal_relevance",
            "function": retrieve_mmr_with_fqhb,
            "params": {"lambda_param": 0.7, "fqhc_boost": 1.3}
        },
        {
            "name": "hybrid_semantic_keyword",
            "function": retrieve_hybrid_fqhc_focused,
            "params": {"alpha": 0.7, "fqhc_keyword_weight": 2.0}
        }
    ]
    
    for strategy in strategies:
        print(f"\n🔍 Testing retrieval: {strategy['name']}")
        
        all_metrics = []
        retrieval_times = []
        
        for i, (query, query_embedding) in enumerate(zip(test_queries, query_embeddings)):
            start_time = time.time()
            
            # Retrieve using strategy
            retrieved = strategy["function"](
                query_embedding=query_embedding,
                query_text=query["text"],
                chunk_embeddings=chunk_embeddings,
                chunk_texts=chunk_texts,
                chunk_metadata=chunk_metadata,
                top_k=RAG_CONFIG["phase1"]["top_k_results"],
                **strategy["params"]
            )
            
            retrieval_times.append(time.time() - start_time)
            
            # Evaluate against ground truth
            metrics = evaluate_retrieval_against_labels(
                retrieved_chunks=retrieved,
                query=query,
                chunk_metadata=chunk_metadata
            )
            all_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = aggregate_evaluation_metrics(all_metrics)
        
        results.append({
            "strategy_name": strategy["name"],
            "params": strategy["params"],
            "avg_retrieval_time": np.mean(retrieval_times),
            "evaluation_metrics": avg_metrics
        })
        
        print(f"  Precision@3: {avg_metrics['precision_at_3']:.3f}")
        print(f"  FQHC Alignment: {avg_metrics['fqhc_alignment_score']:.3f}")
        print(f"  Avg time: {results[-1]['avg_retrieval_time']:.3f}s")
    
    return results

def retrieve_with_fqhc_boost(query_embedding, query_text, chunk_embeddings, 
                            chunk_texts, chunk_metadata, top_k=5, fqhc_boost=1.5):
    """Semantic similarity with FQHC relevance boost - UPDATED"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Basic semantic similarity
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # Apply FQHC boost
    boosted_similarities = []
    for i, (similarity, metadata) in enumerate(zip(similarities, chunk_metadata)):
        # Check FQHC relevance
        fqhc_score = SemanticChunker._calculate_fqhc_relevance(chunk_texts[i])
        
        # Boost FQHC-relevant chunks
        if fqhc_score > 0.5:
            boosted_score = similarity * fqhc_boost
        else:
            boosted_score = similarity
        
        boosted_similarities.append((i, boosted_score, similarity, fqhc_score))
    
    # Sort by boosted score
    boosted_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k with metadata
    results = []
    for idx, boosted_score, original_score, fqhc_score in boosted_similarities[:top_k]:
        results.append({
            "text": chunk_texts[idx],
            "metadata": chunk_metadata[idx],
            "similarity_score": float(original_score),
            "boosted_score": float(boosted_score),
            "fqhc_score": float(fqhc_score),
            "retrieval_method": "semantic_fqhc_boost"
        })
    
    return results    
# Add this class to phase1_foundation.py (around line 754)
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)

def retrieve_mmr_with_fqhb(query_embedding, query_text, chunk_embeddings, chunk_texts,
    chunk_metadata, top_k=5, lambda_param=0.7, fqhc_boost=1.3):
    """MMR with FQHC consideration"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    selected = []
    remaining = list(range(len(chunk_texts)))
    
    while len(selected) < top_k and remaining:
        scores = []
        
        for idx in remaining:
            # Relevance to query
            relevance = similarities[idx]
            
            # FQHC boost
            fqhc_score = SemanticChunker._calculate_fqhc_relevance(chunk_texts[idx])
            if fqhc_score > 0.5:
                relevance *= fqhc_boost
            
            # Diversity penalty
            if selected:
                selected_embs = chunk_embeddings[selected]
                sim_to_selected = cosine_similarity([chunk_embeddings[idx]], selected_embs)[0]
                max_sim = sim_to_selected.max()
            else:
                max_sim = 0
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            scores.append((idx, mmr_score, relevance, fqhc_score))
        
        # Select best
        next_idx = max(scores, key=lambda x: x[1])[0]
        selected.append(next_idx)
        remaining.remove(next_idx)
    
    # Prepare results
    results = []
    for idx in selected:
        fqhc_score = SemanticChunker._calculate_fqhc_relevance(chunk_texts[idx])
        results.append({
            "text": chunk_texts[idx],
            "metadata": chunk_metadata[idx],
            "similarity_score": float(similarities[idx]),
            "fqhc_score": float(fqhc_score),
            "retrieval_method": "mmr_fqhc"
        })
    
    return results

def retrieve_hybrid_fqhc_focused(query_embedding, query_text, chunk_embeddings,
    chunk_texts, chunk_metadata, top_k=5,
    alpha=0.7, fqhc_keyword_weight=2.0):
    """Hybrid semantic + keyword with FQHC focus"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Semantic component
    semantic_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # Keyword component (with FQHC emphasis)
    query_words = set(query_text.lower().split())
    fqhc_keywords = set(SemanticChunker.FQHC_KEYWORDS)
    
    keyword_scores = []
    for chunk_text in chunk_texts:
        chunk_words = set(chunk_text.lower().split())
        
        # Regular keyword overlap
        regular_overlap = len(query_words.intersection(chunk_words))
        
        # FQHC keyword overlap (weighted more)
        fqhc_overlap = len(fqhc_keywords.intersection(chunk_words))
        
        # Combined keyword score
        keyword_score = (regular_overlap + fqhc_keyword_weight * fqhc_overlap)
        keyword_score = keyword_score / max(len(query_words) + fqhc_keyword_weight * len(fqhc_keywords), 1)
        
        keyword_scores.append(keyword_score)
    
    # Combine scores
    combined_scores = alpha * semantic_scores + (1 - alpha) * np.array(keyword_scores)
    
    # Get top indices
    top_indices = combined_scores.argsort()[-top_k:][::-1]
    
    # Prepare results
    results = []
    for idx in top_indices:
        fqhc_score = SemanticChunker._calculate_fqhc_relevance(chunk_texts[idx])
        results.append({
            "text": chunk_texts[idx],
            "metadata": chunk_metadata[idx],
            "semantic_score": float(semantic_scores[idx]),
            "keyword_score": float(keyword_scores[idx]),
            "combined_score": float(combined_scores[idx]),
            "fqhc_score": float(fqhc_score),
            "retrieval_method": "hybrid_fqhc"
        })
    
    return results

def evaluate_retrieval_against_labels(retrieved_chunks, query, chunk_metadata):
    """Evaluate retrieval against ground truth labels"""
    # Extract grant IDs from retrieved chunks
    retrieved_grant_ids = []
    for chunk in retrieved_chunks:
        grant_id = chunk.get("metadata", {}).get("grant_id")
        if grant_id:
            retrieved_grant_ids.append(grant_id)
    
    # Get ground truth
    relevant_grants = set(query.get("relevant_grants", []))
    somewhat_relevant = set(query.get("somewhat_relevant", []))
    not_relevant = set(query.get("not_relevant", []))
    
    # Calculate metrics
    metrics = {
        "precision_at_1": 0,
        "precision_at_3": 0,
        "precision_at_5": 0,
        "recall": 0,
        "fqhc_alignment_score": 0,
        "relevance_breakdown": {"high": 0, "medium": 0, "low": 0}
    }
    
    # Precision@K
    for k in [1, 3, 5]:
        if len(retrieved_grant_ids) >= k:
            top_k = retrieved_grant_ids[:k]
            relevant_in_top_k = len([g for g in top_k if g in relevant_grants])
            metrics[f"precision_at_{k}"] = relevant_in_top_k / k
    
    # Recall (how many relevant grants were retrieved)
    total_relevant = len(relevant_grants)
    if total_relevant > 0:
        retrieved_relevant = len([g for g in retrieved_grant_ids if g in relevant_grants])
        metrics["recall"] = retrieved_relevant / total_relevant  

    # FQHC alignment score
    fqhc_scores = []
    for chunk in retrieved_chunks:
        # Check if chunk is FQHC-focused
        metadata = chunk.get("metadata", {})
        if metadata.get("is_fqhc_focused"):
            fqhc_scores.append(1.0)
        else:
            # Calculate from text
            fqhc_score = SemanticChunker._calculate_fqhc_relevance(chunk.get("text", ""))
            fqhc_scores.append(fqhc_score)
    
    if fqhc_scores:
        metrics["fqhc_alignment_score"] = np.mean(fqhc_scores)
    
    # Relevance breakdown
    for grant_id in retrieved_grant_ids[:5]:  # Top 5 only
        if grant_id in relevant_grants:
            metrics["relevance_breakdown"]["high"] += 1
        elif grant_id in somewhat_relevant:
            metrics["relevance_breakdown"]["medium"] += 1
        elif grant_id in not_relevant:
            metrics["relevance_breakdown"]["low"] += 1
    
    return metrics

def aggregate_evaluation_metrics(all_metrics):
    """Aggregate metrics across all queries"""
    aggregated = {
        "precision_at_1": np.mean([m["precision_at_1"] for m in all_metrics]),
        "precision_at_3": np.mean([m["precision_at_3"] for m in all_metrics]),
        "precision_at_5": np.mean([m["precision_at_5"] for m in all_metrics]),
        "recall": np.mean([m["recall"] for m in all_metrics]),
        "fqhc_alignment_score": np.mean([m["fqhc_alignment_score"] for m in all_metrics]),
        "relevance_breakdown": {
            "high": np.mean([m["relevance_breakdown"]["high"] for m in all_metrics]),
            "medium": np.mean([m["relevance_breakdown"]["medium"] for m in all_metrics]),
            "low": np.mean([m["relevance_breakdown"]["low"] for m in all_metrics])
        }
    }
    return aggregated

# ============ VISUALIZATION FUNCTIONS ============

def visualize_semantic_chunking_results(chunking_results):
    """Visualize semantic chunking results"""
    if not VISUALIZATION_AVAILABLE or not chunking_results:
        return
    
    strategies = [r["strategy_name"] for r in chunking_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Phase 1: Semantic Chunking Analysis', fontsize=16)
    
    # 1. Coherence vs Completeness
    ax = axes[0, 0]
    coherence = [r["avg_coherence"] for r in chunking_results]
    completeness = [r["avg_completeness"] for r in chunking_results]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax.bar(x - width/2, coherence, width, label='Coherence', color='skyblue')
    ax.bar(x + width/2, completeness, width, label='Completeness', color='lightgreen')
    
    ax.set_xlabel('Chunking Strategy')
    ax.set_ylabel('Score (0-1)')
    ax.set_title('Chunk Quality: Coherence vs Completeness')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 2. FQHC Relevance
    ax = axes[0, 1]
    fqhc_scores = [r["avg_fqhc_relevance"] for r in chunking_results]
    
    bars = ax.bar(strategies, fqhc_scores, color='salmon')
    ax.set_xlabel('Chunking Strategy')
    ax.set_ylabel('FQHC Relevance Score')
    ax.set_title('FQHC Relevance by Chunking Strategy')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, score in zip(bars, fqhc_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.2f}', ha='center', va='bottom')
    
    # 3. Chunk Size Distribution
    ax = axes[1, 0]
    avg_words = [r["avg_word_count"] for r in chunking_results]
    std_words = [r["std_word_count"] for r in chunking_results]
    
    x = np.arange(len(strategies))
    ax.bar(x, avg_words, yerr=std_words, capsize=5, color='lightblue', alpha=0.7)
    ax.set_xlabel('Chunking Strategy')
    ax.set_ylabel('Average Word Count')
    ax.set_title('Chunk Size Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    
    # 4. Processing Time
    ax = axes[1, 1]
    times = [r["processing_time_seconds"] for r in chunking_results]
    
    bars = ax.bar(strategies, times, color='lightgreen')
    ax.set_xlabel('Chunking Strategy')
    ax.set_ylabel('Processing Time (seconds)')
    ax.set_title('Processing Time by Strategy')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    
    # Add value labels
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('phase1_semantic_chunking.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_embedding_comparison(embedding_results):
    """Visualize biomedical embedding comparison"""
    if not VISUALIZATION_AVAILABLE or not embedding_results:
        return
    
    # Filter successful models
    successful = [r for r in embedding_results if r["status"] == "SUCCESS"]
    if not successful:
        return
    
    model_names = [r["model_name"].split('/')[-1][:15] for r in successful]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Phase 1: Biomedical Embedding Comparison', fontsize=16)
    
    # 1. FQHC Similarity Score
    ax = axes[0, 0]
    fqhc_scores = [r.get("fqhc_similarity_score", 0) for r in successful]
    
    bars = ax.bar(model_names, fqhc_scores, color='royalblue')
    ax.set_xlabel('Model')
    ax.set_ylabel('FQHC Similarity Score')
    ax.set_title('FQHC Content Understanding')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, score in zip(bars, fqhc_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.3f}', ha='center', va='bottom')
    
    # 2. Encoding Speed
    ax = axes[0, 1]
    encode_times = [r["avg_encode_time"] for r in successful]
    
    bars = ax.bar(model_names, encode_times, color='salmon')
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Encode Time (seconds)')
    ax.set_title('Encoding Speed Comparison')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels
    for bar, time_val in zip(bars, encode_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{time_val:.4f}s', ha='center', va='bottom')
    
    # 3. Model Dimensions
    ax = axes[1, 0]
    dimensions = [r["dimensions"] for r in successful]
    
    bars = ax.bar(model_names, dimensions, color='lightgreen')
    ax.set_xlabel('Model')
    ax.set_ylabel('Embedding Dimensions')
    ax.set_title('Model Complexity (Dimensions)')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels
    for bar, dim in zip(bars, dimensions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{dim}', ha='center', va='bottom')
    
    # 4. Model Type Distribution
    ax = axes[1, 1]
    model_types = [r.get("model_type", "unknown") for r in successful]
    type_counts = Counter(model_types)
    
    ax.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',
           startangle=90, colors=['lightblue', 'lightgreen', 'salmon', 'gold'])
    ax.set_title('Distribution of Model Types')
    
    plt.tight_layout()
    plt.savefig('phase1_embedding_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_retrieval_evaluation(retrieval_results):
    """Visualize retrieval evaluation results"""
    if not VISUALIZATION_AVAILABLE or not retrieval_results:
        return
    
    strategies = [r["strategy_name"] for r in retrieval_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Phase 1: Retrieval Strategy Evaluation', fontsize=16)
    
    # 1. Precision@K Comparison
    ax = axes[0, 0]
    x = np.arange(len(strategies))
    width = 0.25
    
    p1 = [r["evaluation_metrics"]["precision_at_1"] for r in retrieval_results]
    p3 = [r["evaluation_metrics"]["precision_at_3"] for r in retrieval_results]
    p5 = [r["evaluation_metrics"]["precision_at_5"] for r in retrieval_results]
    
    ax.bar(x - width, p1, width, label='P@1', color='skyblue')
    ax.bar(x, p3, width, label='P@3', color='lightgreen')
    ax.bar(x + width, p5, width, label='P@5', color='salmon')
    
    ax.set_xlabel('Retrieval Strategy')
    ax.set_ylabel('Precision Score')
    ax.set_title('Precision@K Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 2. FQHC Alignment Scores
    ax = axes[0, 1]
    fqhc_scores = [r["evaluation_metrics"]["fqhc_alignment_score"] for r in retrieval_results]
    
    bars = ax.bar(strategies, fqhc_scores, color='royalblue')
    ax.set_xlabel('Retrieval Strategy')
    ax.set_ylabel('FQHC Alignment Score')
    ax.set_title('FQHC Mission Alignment')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, score in zip(bars, fqhc_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.3f}', ha='center', va='bottom')
    
    # 3. Retrieval Time
    ax = axes[1, 0]
    times = [r["avg_retrieval_time"] for r in retrieval_results]
    
    bars = ax.bar(strategies, times, color='lightgreen')
    ax.set_xlabel('Retrieval Strategy')
    ax.set_ylabel('Average Time (seconds)')
    ax.set_title('Retrieval Speed Comparison')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    
    # Add value labels
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{time_val:.3f}s', ha='center', va='bottom')
    
    # 4. Relevance Breakdown
    ax = axes[1, 1]
    x = np.arange(len(strategies))
    width = 0.25
    
    high = [r["evaluation_metrics"]["relevance_breakdown"]["high"] for r in retrieval_results]
    medium = [r["evaluation_metrics"]["relevance_breakdown"]["medium"] for r in retrieval_results]
    low = [r["evaluation_metrics"]["relevance_breakdown"]["low"] for r in retrieval_results]
    
    ax.bar(x - width, high, width, label='High Relevance', color='green')
    ax.bar(x, medium, width, label='Medium Relevance', color='orange')
    ax.bar(x + width, low, width, label='Low Relevance', color='red')
    
    ax.set_xlabel('Retrieval Strategy')
    ax.set_ylabel('Average Count (top 5)')
    ax.set_title('Relevance Distribution in Top Results')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('phase1_retrieval_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============ MAIN EXECUTION ============

def run_phase1_tests():
    """MAIN FUNCTION TO RUN ALL PHASE 1 TESTS"""
    print("\n" + "="*70)
    print("🚀 STARTING PHASE 1 TESTS")
    print("="*70)
    
    # 1. Generate test data with labels
    print("\n📊 GENERATING TEST DATA WITH LABELS...")
    test_data = TestDataGenerator()
    grants_df = test_data.create_sample_grants(RAG_CONFIG["phase1"]["test_sample_size"])
    test_queries = test_data.create_labeled_test_queries()
    
    print(f"✅ Created {len(grants_df)} sample grants")
    print(f"✅ Created {len(test_queries)} labeled test queries")
    
    # Display test query info
    print("\n📋 Test Queries Created:")
    for i, query in enumerate(test_queries, 1):
        print(f"  {i}. {query['query_id']}: {query['text'][:60]}...")
        print(f"     Relevant grants: {len(query['relevant_grants'])}")
    
    # 2. Week 1: Semantic chunking tests
    print("\n" + "="*70)
    print("📅 WEEK 1: SEMANTIC CHUNKING TESTS")
    print("="*70)
    
    chunking_results = test_semantic_chunking_strategies(
        grants_df["abstract"].tolist()
    )
    
    # Select best chunking strategy
    best_chunking = max(chunking_results, key=lambda x: 
                       (x["avg_coherence"] + x["avg_completeness"]) / 2)
    print(f"\n🎯 Recommended chunking: {best_chunking['strategy_name']}")
    print(f"   Coherence: {best_chunking['avg_coherence']:.2f}")
    print(f"   Completeness: {best_chunking['avg_completeness']:.2f}")
    print(f"   FQHC Relevance: {best_chunking['avg_fqhc_relevance']:.2f}")
    
    # Create chunks for subsequent tests using best strategy
    chunker = SemanticChunker()
    all_chunks = []
    
    if best_chunking['strategy_name'] == "abstract_whole":
        chunk_func = chunker.chunk_abstract_whole
    elif best_chunking['strategy_name'] == "section_based":
        chunk_func = chunker.chunk_by_sections
    else:
        chunk_func = chunker.chunk_by_topic_shift
    
    for i, text in enumerate(grants_df["abstract"].tolist()):
        chunks = chunk_func(text)
        # Add grant metadata to chunks
        for chunk in chunks:
            chunk.update({
                "grant_id": grants_df.iloc[i]["grant_id"],
                "is_fqhc_focused": grants_df.iloc[i]["is_fqhc_focused"],
                "institute": grants_df.iloc[i]["institute"],
                "year": grants_df.iloc[i]["year"]
            })
        all_chunks.extend(chunks)
    
    print(f"📦 Created {len(all_chunks)} chunks using {best_chunking['strategy_name']}")
    
    # 3. Week 2: Biomedical embedding tests
    print("\n" + "="*70)
    print("📅 WEEK 2: BIOMEDICAL EMBEDDING TESTS")
    print("="*70)
    
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    embedding_results = test_biomedical_embeddings(chunk_texts, test_queries)
    
    # Select best model
    successful_models = [r for r in embedding_results if r["status"] == "SUCCESS"]
    if successful_models:
        # Prioritize FQHC understanding
        best_model = max(successful_models, key=lambda x: 
                        (x.get("fqhc_similarity_score", 0) * 0.7 + 
                         (1 / x["avg_encode_time"]) * 0.3))
        print(f"\n🎯 Recommended embedding model: {best_model['model_name']}")
        print(f"   FQHC Similarity: {best_model.get('fqhc_similarity_score', 0):.3f}")
        print(f"   Encode Time: {best_model['avg_encode_time']:.4f}s")
        print(f"   Dimensions: {best_model['dimensions']}")
    
    # 4. Week 3: Retrieval with evaluation
    print("\n" + "="*70)
    print("📅 WEEK 3: RETRIEVAL WITH EVALUATION")
    print("="*70)
    
    # Use best model for retrieval tests
    best_model_name = best_model['model_name'] if successful_models else "all-mpnet-base-v2"
    
    retrieval_results = test_retrieval_with_evaluation(
        all_chunks, 
        test_queries,
        embedding_model_name=best_model_name
    )
    
    # Select best retrieval strategy
    best_retrieval = max(retrieval_results, key=lambda x: 
                        x["evaluation_metrics"]["precision_at_3"])
    print(f"\n🎯 Recommended retrieval: {best_retrieval['strategy_name']}")
    print(f"   Precision@3: {best_retrieval['evaluation_metrics']['precision_at_3']:.3f}")
    print(f"   FQHC Alignment: {best_retrieval['evaluation_metrics']['fqhc_alignment_score']:.3f}")
    print(f"   Retrieval Time: {best_retrieval['avg_retrieval_time']:.3f}s")
    
    # 5. Visualization
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70)
    
    visualize_semantic_chunking_results(chunking_results)
    visualize_embedding_comparison(embedding_results)
    visualize_retrieval_evaluation(retrieval_results)
    
    # 6. Save results
    results = {
        "phase": "phase1_foundation",
        "timestamp": datetime.now().isoformat(),
        "recommendations": {
            "chunking_strategy": best_chunking["strategy_name"],
            "embedding_model": best_model["model_name"] if successful_models else None,
            "retrieval_strategy": best_retrieval["strategy_name"]
        },
        "detailed_results": {
            "chunking": chunking_results,
            "embeddings": embedding_results,
            "retrieval": retrieval_results
        },
        "test_data_info": {
            "grants_count": len(grants_df),
            "queries_count": len(test_queries),
            "fqhc_grants": sum(grants_df["is_fqhc_focused"]),
            "non_fqhc_grants": len(grants_df) - sum(grants_df["is_fqhc_focused"])
        },
        "config": RAG_CONFIG["phase1"]
    }
    
    with open("phase1_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print("\n" + "="*70)
    print("✅ PHASE 1 COMPLETE!")
    print("="*70)
    print("\n📁 Results saved to:")
    print("  • phase1_results.json")
    print("  • phase1_semantic_chunking.png")
    print("  • phase1_embedding_comparison.png")
    print("  • phase1_retrieval_evaluation.png")
    print("\n🎯 RECOMMENDATIONS:")
    print(f"  1. Chunking: {results['recommendations']['chunking_strategy']}")
    print(f"  2. Embedding: {results['recommendations']['embedding_model']}")
    print(f"  3. Retrieval: {results['recommendations']['retrieval_strategy']}")
    print("\n🚀 Next: Run Phase 2 (API Integration)")
    
    return results

# ============================================================================
# 🏃‍♂️ RUN PHASE 1 TESTS
# ============================================================================

if __name__ == "__main__":
    # Install required packages
    print("📦 Checking/installing required packages...")
    try:
        import sentence_transformers
    except ImportError:
        print("Installing sentence-transformers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    
    try:
        import sklearn
    except ImportError:
        print("Installing scikit-learn...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    
    # Run tests
    results = run_phase1_tests()