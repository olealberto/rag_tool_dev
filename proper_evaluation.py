# ============================================================================
# 📁 proper_evaluation.py - MEASURES ACTUAL RAG PERFORMANCE
# ============================================================================

"""
PROPER EVALUATION FOR RAG SYSTEMS
Measures topical relevance, not just exact grant ID matches
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import json
import time
import re
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# ============ TOPIC-BASED RELEVANCE EVALUATOR ============

class TopicBasedEvaluator:
    """
    Evaluates RAG performance by checking if retrieved chunks
    actually contain the topics the query is asking about.
    """
    
    # Comprehensive topic keywords for FQHC/health domain
    TOPIC_KEYWORDS = {
        # Chronic conditions
        "diabetes": [
            "diabetes", "diabetic", "type 2 diabetes", "type ii diabetes",
            "hba1c", "a1c", "glycemic control", "blood sugar", "insulin",
            "glucose", "metformin", "diabetes prevention", "dpp"
        ],
        "hypertension": [
            "hypertension", "high blood pressure", "blood pressure",
            "cardiovascular", "heart disease", "stroke", "systolic",
            "diastolic", "antihypertensive", "bp control"
        ],
        "cancer": [
            "cancer", "oncology", "tumor", "carcinoma", "malignancy",
            "screening", "mammography", "colonoscopy", "pap smear",
            "breast cancer", "colorectal cancer", "lung cancer"
        ],
        "depression": [
            "depression", "depressive", "mental health", "phq-9",
            "antidepressant", "mood disorder", "anxiety", "cbt",
            "cognitive behavioral", "behavioral health"
        ],
        "hiv": [
            "hiv", "aids", "prep", "antiretroviral", "art",
            "hiv prevention", "hiv care", "viral load"
        ],
        "asthma": [
            "asthma", "respiratory", "inhaler", "bronchial",
            "pulmonary", "lung function", "peak flow"
        ],
        "substance_use": [
            "substance use", "opioid", "addiction", "naloxone",
            "buprenorphine", "mat", "medication assisted treatment",
            "alcohol use", "drug use", "overdose"
        ],
        
        # Interventions
        "chw": [
            "community health worker", "chw", "promotora",
            "lay health advisor", "community health educator",
            "patient navigator", "health worker"
        ],
        "telehealth": [
            "telehealth", "telemedicine", "telehealth", "virtual visit",
            "remote monitoring", "mhealth", "mobile health",
            "video visit", "remote care", "digital health"
        ],
        "navigation": [
            "patient navigation", "navigator", "care coordination",
            "care manager", "care management", "patient support"
        ],
        "screening": [
            "screening", "early detection", "preventive screening",
            "health screening", "risk assessment", "screen"
        ],
        "integrated_care": [
            "integrated care", "collaborative care", "co-located",
            "behavioral health integration", "primary care behavioral health",
            "integrated behavioral health"
        ],
        
        # Populations
        "latino": [
            "latino", "hispanic", "latinx", "spanish speaking",
            "mexican american", "puerto rican", "promotora"
        ],
        "african_american": [
            "african american", "black", "african-american"
        ],
        "pediatric": [
            "pediatric", "children", "child", "adolescent", "youth",
            "school based", "pediatric", "child health"
        ],
        "geriatric": [
            "older adult", "geriatric", "elderly", "aging", "senior",
            "aged", "geriatric"
        ],
        "rural": [
            "rural", "appalachian", "frontier", "rural health",
            "non-urban", "rural population"
        ],
        "low_income": [
            "low-income", "low income", "poverty", "economically disadvantaged",
            "underserved", "medicaid", "uninsured"
        ],
        
        # Settings
        "fqhc": [
            "federally qualified health center", "fqhc",
            "community health center", "safety-net clinic",
            "safety net", "health center", "community clinic"
        ],
        "primary_care": [
            "primary care", "primary healthcare", "general practice",
            "family medicine", "primary care setting"
        ]
    }
    
    def __init__(self):
        # Build reverse index: keyword -> topics
        self.keyword_to_topics = defaultdict(set)
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            for kw in keywords:
                self.keyword_to_topics[kw].add(topic)
    
    def extract_query_topics(self, query: str) -> Set[str]:
        """Extract which topics a query is asking about"""
        query_lower = query.lower()
        topics = set()
        
        # Direct topic matching
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                topics.add(topic)
        
        return topics
    
    def extract_chunk_topics(self, chunk_text: str) -> Set[str]:
        """Extract which topics appear in a chunk"""
        if not isinstance(chunk_text, str):
            return set()
        
        chunk_lower = chunk_text.lower()
        topics = set()
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(kw in chunk_lower for kw in keywords):
                topics.add(topic)
        
        return topics
    
    def compute_relevance_score(self, query_topics: Set[str], chunk_topics: Set[str]) -> float:
        """
        Compute how relevant a chunk is to a query based on topic overlap
        Returns score 0-1
        """
        if not query_topics:
            return 0.0
        
        intersection = query_topics.intersection(chunk_topics)
        if not intersection:
            return 0.0
        
        # Weight: proportion of query topics covered
        coverage = len(intersection) / len(query_topics)
        
        # Bonus for exact matches of key terms
        return coverage
    
    def is_relevant(self, query: str, chunk_text: str, threshold: float = 0.3) -> bool:
        """Binary relevance判定 for a single chunk"""
        query_topics = self.extract_query_topics(query)
        chunk_topics = self.extract_chunk_topics(chunk_text)
        score = self.compute_relevance_score(query_topics, chunk_topics)
        return score >= threshold
    
    def generate_ground_truth_from_corpus(self, chunks_df: pd.DataFrame, num_queries: int = 20) -> List[Dict]:
        """
        Generate ground truth by finding chunks that naturally contain query topics
        This is more realistic than using pre-defined grant IDs
        """
        print(f"\n🎯 Generating ground truth from {len(chunks_df)} chunks...")
        
        # Sample diverse queries based on actual content
        all_topics = list(self.TOPIC_KEYWORDS.keys())
        queries = []
        
        # Common query patterns
        patterns = [
            "{condition} prevention and management in {setting}",
            "{intervention} for {condition} in {population} patients",
            "{population} health interventions for {condition}",
            "addressing {condition} through {intervention} in {setting}",
            "{intervention} programs in {setting} for {population}"
        ]
        
        for i in range(num_queries):
            # Randomly select topics
            condition = np.random.choice([t for t in all_topics if t in [
                "diabetes", "hypertension", "cancer", "depression", 
                "hiv", "asthma", "substance_use"
            ]])
            intervention = np.random.choice([t for t in all_topics if t in [
                "chw", "telehealth", "navigation", "screening", "integrated_care"
            ]])
            population = np.random.choice([t for t in all_topics if t in [
                "latino", "african_american", "pediatric", "geriatric", 
                "rural", "low_income"
            ]])
            setting = np.random.choice([t for t in all_topics if t in [
                "fqhc", "primary_care"
            ]])
            
            pattern = np.random.choice(patterns)
            query = pattern.format(
                condition=condition.replace("_", " "),
                intervention=intervention.replace("_", " "),
                population=population.replace("_", " "),
                setting=setting.replace("_", " ")
            )
            
            # Find relevant chunks for this query
            query_topics = self.extract_query_topics(query)
            relevant_chunks = []
            
            for idx, chunk in chunks_df.iterrows():
                chunk_topics = self.extract_chunk_topics(chunk.get('text', ''))
                if self.compute_relevance_score(query_topics, chunk_topics) > 0:
                    relevant_chunks.append({
                        'chunk_id': chunk.get('chunk_id', f'chunk_{idx}'),
                        'grant_id': chunk.get('grant_id', ''),
                        'topics': list(chunk_topics),
                        'score': self.compute_relevance_score(query_topics, chunk_topics)
                    })
            
            # Sort by relevance
            relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
            
            queries.append({
                'query_id': f'Q{i+1:03d}_TOPIC',
                'query': query,
                'query_topics': list(query_topics),
                'relevant_chunk_ids': [c['chunk_id'] for c in relevant_chunks[:10]],
                'relevant_grant_ids': list(set([c['grant_id'] for c in relevant_chunks[:10]])),
                'all_relevant_counts': len(relevant_chunks),
                'generation_method': 'topic_based'
            })
        
        print(f"✅ Generated {len(queries)} topic-based queries")
        print(f"   Average relevant chunks per query: {np.mean([len(q['relevant_chunk_ids']) for q in queries]):.1f}")
        
        return queries


class ProperRAGEvaluator:
    """
    Evaluates RAG performance using proper metrics
    """
    
    def __init__(self, rag_system, evaluator: TopicBasedEvaluator = None):
        self.rag = rag_system
        self.evaluator = evaluator or TopicBasedEvaluator()
        self.results_history = []
    
    def precision_at_k(self, relevant: Set, retrieved: List, k: int) -> float:
        """Precision@K = (# relevant in top K) / K"""
        if k == 0 or not retrieved:
            return 0.0
        top_k = retrieved[:k]
        relevant_in_top = len([item for item in top_k if item in relevant])
        return relevant_in_top / k
    
    def recall_at_k(self, relevant: Set, retrieved: List, k: int) -> float:
        """Recall@K = (# relevant in top K) / total relevant"""
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        relevant_in_top = len([item for item in top_k if item in relevant])
        return relevant_in_top / len(relevant)
    
    def average_precision(self, relevant: Set, retrieved: List) -> float:
        """Average Precision (AP)"""
        if not relevant:
            return 0.0
        
        score = 0.0
        num_hits = 0
        
        for i, item in enumerate(retrieved):
            if item in relevant:
                num_hits += 1
                score += num_hits / (i + 1)
        
        return score / len(relevant) if relevant else 0.0
    
    def ndcg_at_k(self, relevant: Set, retrieved: List, k: int) -> float:
        """Normalized Discounted Cumulative Gain @ K"""
        if k == 0 or not retrieved:
            return 0.0
        
        # Simple binary relevance for now
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, item in enumerate(retrieved[:k]):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0
        
        # Calculate IDCG (ideal DCG)
        num_relevant = min(len(relevant), k)
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_query(self, query: str, ground_truth: Dict, 
                      retrieve_func, top_k: int = 10, **kwargs) -> Dict:
        """
        Evaluate a single query using ground truth
        """
        # Get query topics
        query_topics = self.evaluator.extract_query_topics(query)
        
        # Retrieve results
        start_time = time.time()
        results = retrieve_func(query, top_k=top_k, **kwargs)
        retrieval_time = time.time() - start_time
        
        # Extract chunk IDs and grant IDs
        retrieved_chunk_ids = []
        retrieved_grant_ids = []
        chunk_scores = []
        
        for r in results:
            chunk_id = r.get('chunk_id', '')
            grant_id = r.get('grant_id', '')
            score = r.get('similarity', r.get('score', 0))
            
            retrieved_chunk_ids.append(chunk_id)
            retrieved_grant_ids.append(grant_id)
            chunk_scores.append(score)
        
        # Get ground truth sets
        relevant_chunks = set(ground_truth.get('relevant_chunk_ids', []))
        relevant_grants = set(ground_truth.get('relevant_grant_ids', []))
        
        # Calculate metrics (using chunk-level relevance)
        metrics = {
            'query': query,
            'query_topics': list(query_topics),
            'retrieval_time': retrieval_time,
            'num_results': len(results),
            'avg_score': np.mean(chunk_scores) if chunk_scores else 0,
            'max_score': max(chunk_scores) if chunk_scores else 0,
            
            # Precision@K
            'p@1': self.precision_at_k(relevant_chunks, retrieved_chunk_ids, 1),
            'p@3': self.precision_at_k(relevant_chunks, retrieved_chunk_ids, 3),
            'p@5': self.precision_at_k(relevant_chunks, retrieved_chunk_ids, 5),
            'p@10': self.precision_at_k(relevant_chunks, retrieved_chunk_ids, 10),
            
            # Recall@K
            'r@1': self.recall_at_k(relevant_chunks, retrieved_chunk_ids, 1),
            'r@3': self.recall_at_k(relevant_chunks, retrieved_chunk_ids, 3),
            'r@5': self.recall_at_k(relevant_chunks, retrieved_chunk_ids, 5),
            'r@10': self.recall_at_k(relevant_chunks, retrieved_chunk_ids, 10),
            
            # Other metrics
            'map': self.average_precision(relevant_chunks, retrieved_chunk_ids),
            'ndcg@5': self.ndcg_at_k(relevant_chunks, retrieved_chunk_ids, 5),
            'ndcg@10': self.ndcg_at_k(relevant_chunks, retrieved_chunk_ids, 10),
            
            # Grant-level metrics (less strict than chunk-level)
            'p@5_grants': self.precision_at_k(relevant_grants, retrieved_grant_ids, 5),
            
            # Detailed breakdown
            'retrieved_chunk_ids': retrieved_chunk_ids[:10],
            'retrieved_grant_ids': retrieved_grant_ids[:10],
            'relevant_in_results': len([c for c in retrieved_chunk_ids if c in relevant_chunks]),
            'total_relevant': len(relevant_chunks),
        }
        
        # Add topical relevance check
        if query_topics:
            topical_relevant = 0
            for i, r in enumerate(results[:5]):
                chunk_topics = self.evaluator.extract_chunk_topics(r.get('text', ''))
                if chunk_topics.intersection(query_topics):
                    topical_relevant += 1
            metrics['topical_p@5'] = topical_relevant / 5
        else:
            metrics['topical_p@5'] = 0
        
        return metrics
    
    def evaluate_system(self, rag_system, test_queries: List[Dict] = None,
                       num_queries: int = 20, verbose: bool = True) -> Dict:
        """
        Complete system evaluation
        """
        print("\n" + "="*70)
        print("📊 PROPER RAG SYSTEM EVALUATION")
        print("="*70)
        
        # Generate or load test queries
        if test_queries is None:
            # Try to load from chunks
            if hasattr(rag_system, 'chunks_df') and rag_system.chunks_df is not None:
                test_queries = self.evaluator.generate_ground_truth_from_corpus(
                    rag_system.chunks_df, num_queries
                )
            else:
                print("❌ No test queries or chunks available")
                return {}
        
        all_metrics = []
        
        for i, q_data in enumerate(test_queries):
            query = q_data['query'] if isinstance(q_data, dict) else q_data
            
            if verbose:
                print(f"\n🔍 [{i+1}/{len(test_queries)}] {query[:60]}...")
            
            # Use the rag_system's search method
            metrics = self.evaluate_query(
                query=query,
                ground_truth=q_data,
                retrieve_func=rag_system.search,
                top_k=10,
                fqhc_boost=True
            )
            
            all_metrics.append(metrics)
            
            if verbose:
                print(f"   P@5: {metrics['p@5']:.3f} | Topical P@5: {metrics.get('topical_p@5', 0):.3f} | "
                      f"Max sim: {metrics['max_score']:.3f}")
        
        # Aggregate metrics
        aggregated = self.aggregate_metrics(all_metrics)
        
        # Store results
        self.results_history.append({
            'timestamp': datetime.now().isoformat(),
            'system': str(type(rag_system).__name__),
            'num_queries': len(all_metrics),
            'metrics': aggregated,
            'detailed': all_metrics
        })
        
        return aggregated
    
    def aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across all queries"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Metrics to aggregate
        keys = [
            'p@1', 'p@3', 'p@5', 'p@10',
            'r@1', 'r@3', 'r@5', 'r@10',
            'map', 'ndcg@5', 'ndcg@10',
            'p@5_grants', 'topical_p@5',
            'retrieval_time', 'avg_score', 'max_score'
        ]
        
        for key in keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                aggregated[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # Add summary stats
        aggregated['num_queries'] = len(metrics_list)
        aggregated['zero_precision_queries'] = len([
            m for m in metrics_list if m.get('p@5', 0) == 0
        ])
        aggregated['perfect_precision_queries'] = len([
            m for m in metrics_list if m.get('p@5', 0) == 1
        ])
        
        return aggregated
    
    def compare_strategies(self, rag_system, strategies: List[Dict]) -> pd.DataFrame:
        """
        Compare different retrieval strategies (alpha tuning, boost values, etc.)
        """
        print("\n" + "="*70)
        print("🔄 COMPARING RETRIEVAL STRATEGIES")
        print("="*70)
        
        # Generate base test queries once
        if hasattr(rag_system, 'chunks_df') and rag_system.chunks_df is not None:
            test_queries = self.evaluator.generate_ground_truth_from_corpus(
                rag_system.chunks_df, num_queries=15
            )
        else:
            print("❌ Cannot compare - no chunks available")
            return pd.DataFrame()
        
        results = []
        
        for strategy in strategies:
            print(f"\n📌 Testing: {strategy['name']}")
            
            # Patch the search method with strategy parameters
            original_search = rag_system.search
            
            def patched_search(query, top_k=5, **kwargs):
                return original_search(query, top_k=top_k, **strategy['params'])
            
            rag_system.search = patched_search
            
            # Evaluate
            metrics = self.evaluate_system(
                rag_system, 
                test_queries=test_queries,
                verbose=False
            )
            
            results.append({
                'strategy': strategy['name'],
                'params': strategy['params'],
                'p@5': metrics.get('p@5', {}).get('mean', 0),
                'topical_p@5': metrics.get('topical_p@5', {}).get('mean', 0),
                'map': metrics.get('map', {}).get('mean', 0),
                'ndcg@5': metrics.get('ndcg@5', {}).get('mean', 0),
                'retrieval_time': metrics.get('retrieval_time', {}).get('mean', 0)
            })
            
            # Restore original
            rag_system.search = original_search
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        
        print("\n📊 STRATEGY COMPARISON:")
        print(df.to_string(index=False))
        
        return df
    
    def detailed_report(self, output_path: str = "evaluation/proper_evaluation.json"):
        """Generate detailed evaluation report"""
        if not self.results_history:
            print("No evaluation results to report")
            return
        
        latest = self.results_history[-1]
        
        report = {
            'summary': {
                'timestamp': latest['timestamp'],
                'system': latest['system'],
                'num_queries': latest['num_queries'],
                'p@5_mean': latest['metrics'].get('p@5', {}).get('mean', 0),
                'p@5_std': latest['metrics'].get('p@5', {}).get('std', 0),
                'topical_p@5_mean': latest['metrics'].get('topical_p@5', {}).get('mean', 0),
                'map_mean': latest['metrics'].get('map', {}).get('mean', 0),
                'zero_precision_queries': latest['metrics'].get('zero_precision_queries', 0),
                'perfect_precision_queries': latest['metrics'].get('perfect_precision_queries', 0)
            },
            'detailed_metrics': latest['metrics'],
            'sample_queries': [
                {
                    'query': m['query'][:100],
                    'p@5': m['p@5'],
                    'topical_p@5': m.get('topical_p@5', 0),
                    'max_score': m['max_score']
                }
                for m in latest['detailed'][:5]
            ]
        }
        
        # Save to file
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📁 Report saved to {output_path}")
        
        return report
    
    def print_summary(self):
        """Print human-readable summary of latest evaluation"""
        if not self.results_history:
            print("No evaluation results yet")
            return
        
        latest = self.results_history[-1]
        m = latest['metrics']
        
        print("\n" + "="*70)
        print("📈 RAG SYSTEM PERFORMANCE SUMMARY")
        print("="*70)
        print(f"System:           {latest['system']}")
        print(f"Timestamp:        {latest['timestamp']}")
        print(f"Test queries:     {latest['num_queries']}")
        print("-" * 70)
        print("\n📊 PRECISION@K:")
        print(f"  P@1:  {m.get('p@1', {}).get('mean', 0):.3f} ± {m.get('p@1', {}).get('std', 0):.3f}")
        print(f"  P@3:  {m.get('p@3', {}).get('mean', 0):.3f} ± {m.get('p@3', {}).get('std', 0):.3f}")
        print(f"  P@5:  {m.get('p@5', {}).get('mean', 0):.3f} ± {m.get('p@5', {}).get('std', 0):.3f}")
        print(f"  P@10: {m.get('p@10', {}).get('mean', 0):.3f} ± {m.get('p@10', {}).get('std', 0):.3f}")
        
        print("\n📊 RECALL@K:")
        print(f"  R@1:  {m.get('r@1', {}).get('mean', 0):.3f} ± {m.get('r@1', {}).get('std', 0):.3f}")
        print(f"  R@3:  {m.get('r@3', {}).get('mean', 0):.3f} ± {m.get('r@3', {}).get('std', 0):.3f}")
        print(f"  R@5:  {m.get('r@5', {}).get('mean', 0):.3f} ± {m.get('r@5', {}).get('std', 0):.3f}")
        
        print("\n📊 OTHER METRICS:")
        print(f"  MAP:          {m.get('map', {}).get('mean', 0):.3f}")
        print(f"  nDCG@5:       {m.get('ndcg@5', {}).get('mean', 0):.3f}")
        print(f"  Topical P@5:  {m.get('topical_p@5', {}).get('mean', 0):.3f}")
        print(f"  Avg score:    {m.get('avg_score', {}).get('mean', 0):.3f}")
        print(f"  Retrieval:    {m.get('retrieval_time', {}).get('mean', 0):.4f}s")
        
        print("\n📊 QUERY STATS:")
        print(f"  Zero precision:   {m.get('zero_precision_queries', 0)} queries")
        print(f"  Perfect precision:{m.get('perfect_precision_queries', 0)} queries")
        
        if 'p@5' in m and 'topical_p@5' in m:
            gap = m['p@5']['mean'] - m['topical_p@5']['mean']
            print(f"\n📌 Gap (P@5 - Topical P@5): {gap:.3f}")
            if gap < -0.1:
                print("   ⚠️  Topical relevance > strict matching - evaluation too strict!")
            elif gap > 0.1:
                print("   ✅ System finds exact matches well")
        
        print("="*70)


# ============ USAGE EXAMPLE ============

def evaluate_your_system(rag_system):
    """
    Complete evaluation of your RAG system with proper metrics
    """
    
    # 1. Create evaluator
    evaluator = TopicBasedEvaluator()
    proper_eval = ProperRAGEvaluator(rag_system, evaluator)
    
    # 2. Generate topic-based ground truth from your chunks
    print("\n📊 Step 1: Generating topic-based ground truth...")
    test_queries = evaluator.generate_ground_truth_from_corpus(
        rag_system.chunks_df, 
        num_queries=25  # Adjust based on your corpus size
    )
    
    # 3. Run evaluation
    print("\n📊 Step 2: Running evaluation...")
    results = proper_eval.evaluate_system(
        rag_system,
        test_queries=test_queries,
        verbose=True
    )
    
    # 4. Print summary
    proper_eval.print_summary()
    
    # 5. Save report
    proper_eval.detailed_report("evaluation/proper_eval_results.json")
    
    # 6. Compare different alpha values (if using hybrid search)
    if hasattr(rag_system, 'search') and 'alpha' in rag_system.search.__code__.co_varnames:
        print("\n📊 Step 3: Comparing alpha values...")
        strategies = [
            {'name': 'BM25 only', 'params': {'alpha': 0.0}},
            {'name': 'Hybrid 0.3', 'params': {'alpha': 0.3}},
            {'name': 'Hybrid 0.5', 'params': {'alpha': 0.5}},
            {'name': 'Hybrid 0.7', 'params': {'alpha': 0.7}},
            {'name': 'Vector only', 'params': {'alpha': 1.0}},
        ]
        comparison = proper_eval.compare_strategies(rag_system, strategies)
        
        # Find best strategy
        best = comparison.loc[comparison['p@5'].idxmax()]
        print(f"\n✅ Best strategy: {best['strategy']} with P@5={best['p@5']:.3f}")
    
    return proper_eval, results


# ============ QUICK VISUALIZATION ============

def visualize_evaluation_results(evaluator: ProperRAGEvaluator):
    """Create visualizations of evaluation results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not evaluator.results_history:
            print("No results to visualize")
            return
        
        latest = evaluator.results_history[-1]
        metrics = latest['detailed']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RAG System Proper Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Precision@K distribution
        ax = axes[0, 0]
        p5_values = [m['p@5'] for m in metrics]
        ax.hist(p5_values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(p5_values), color='red', linestyle='--', label=f'Mean: {np.mean(p5_values):.3f}')
        ax.set_xlabel('Precision@5')
        ax.set_ylabel('Frequency')
        ax.set_title('Precision@5 Distribution')
        ax.legend()
        
        # 2. Topical P@5 vs Strict P@5
        ax = axes[0, 1]
        topical = [m.get('topical_p@5', 0) for m in metrics]
        strict = [m['p@5'] for m in metrics]
        
        x = np.arange(len(metrics[:15]))
        width = 0.35
        ax.bar(x - width/2, strict[:15], width, label='Strict P@5', color='lightcoral')
        ax.bar(x + width/2, topical[:15], width, label='Topical P@5', color='lightgreen')
        ax.set_xlabel('Query')
        ax.set_ylabel('Precision')
        ax.set_title('Strict vs Topical Relevance')
        ax.legend()
        
        # 3. Similarity scores
        ax = axes[0, 2]
        max_scores = [m['max_score'] for m in metrics]
        ax.scatter(range(len(max_scores)), max_scores, alpha=0.6)
        ax.axhline(np.mean(max_scores), color='red', linestyle='--', label=f'Mean: {np.mean(max_scores):.3f}')
        ax.set_xlabel('Query')
        ax.set_ylabel('Max Similarity Score')
        ax.set_title('Retrieval Confidence Scores')
        ax.legend()
        
        # 4. MAP distribution
        ax = axes[1, 0]
        map_values = [m['map'] for m in metrics if m['map'] > 0]
        if map_values:
            ax.hist(map_values, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.axvline(np.mean(map_values), color='red', linestyle='--', label=f'Mean: {np.mean(map_values):.3f}')
            ax.set_xlabel('Mean Average Precision')
            ax.set_ylabel('Frequency')
            ax.set_title('MAP Distribution')
            ax.legend()
        
        # 5. Retrieval time
        ax = axes[1, 1]
        times = [m['retrieval_time'] for m in metrics]
        ax.boxplot(times)
        ax.set_ylabel('Retrieval Time (seconds)')
        ax.set_title('Retrieval Speed')
        
        # 6. Summary table
        ax = axes[1, 2]
        ax.axis('off')
        
        agg = latest['metrics']
        summary = [
            ['Metric', 'Mean ± Std'],
            ['P@5', f"{agg.get('p@5', {}).get('mean', 0):.3f} ± {agg.get('p@5', {}).get('std', 0):.3f}"],
            ['Topical P@5', f"{agg.get('topical_p@5', {}).get('mean', 0):.3f} ± {agg.get('topical_p@5', {}).get('std', 0):.3f}"],
            ['MAP', f"{agg.get('map', {}).get('mean', 0):.3f}"],
            ['nDCG@5', f"{agg.get('ndcg@5', {}).get('mean', 0):.3f}"],
            ['Zero P@5', str(agg.get('zero_precision_queries', 0))],
            ['Perfect P@5', str(agg.get('perfect_precision_queries', 0))]
        ]
        
        table = ax.table(cellText=summary, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title('Evaluation Summary', pad=20)
        
        plt.tight_layout()
        plt.savefig('evaluation/proper_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 Visualization saved to evaluation/proper_evaluation_results.png")
        
    except ImportError:
        print("⚠️  Visualization libraries not available")


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 PROPER RAG EVALUATION SYSTEM")
    print("="*70)
    print("\nThis module helps you evaluate your RAG system properly.")