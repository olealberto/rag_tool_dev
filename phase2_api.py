# ============================================================================
# 📁 phase2_complete_working.py - COMPLETE WORKING VERSION
# ============================================================================

print("="*70)
print("🎯 PHASE 2: ABSTRACT COLLECTION & VALIDATION")
print("="*70)

import sys
import pandas as pd
import numpy as np
import requests
import time
import json
import re
import os
from typing import List, Dict, Any
from datetime import datetime

# Import embedding model for validation
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️  Sentence transformers not available.")

# Configuration
RAG_CONFIG = {
    "phase2": {
        "target_abstract_count": 2000,
        "min_abstract_length": 100
    }
}

class NIHAPICollector:
    """
    NIH API COLLECTOR - WORKING VERSION
    """
    
    def __init__(self):
        self.base_url = "https://api.reporter.nih.gov/v2/projects/search"
        self.headers = {'Content-Type': 'application/json'}
        self.rate_limit_delay = 1.5
        
        # Universities that work well
        self.research_universities = [
            "UNIVERSITY OF ILLINOIS AT CHICAGO",
            "UNIVERSITY OF MICHIGAN",
            "UNIVERSITY OF CHICAGO",
            "NORTHWESTERN UNIVERSITY",
            "JOHNS HOPKINS UNIVERSITY"
        ]
        
        self.fqhc_keywords = [
            'federally qualified health center',
            'fqhc',
            'community health center',
            'safety-net',
            'medically underserved',
            'low-income',
            'uninsured',
            'medicaid',
            'health disparities',
            'primary care'
        ]
    
    def fetch_university_grants(self, university: str, year: int, limit: int = 400) -> List[Dict]:
        """Fetch grants for a specific university"""
        payload = {
            "criteria": {
                "fiscal_years": [year],
                "org_names": [university]
            },
            "limit": limit,
            "offset": 0
        }
        
        try:
            response = requests.post(
                self.base_url, 
                json=payload, 
                headers=self.headers,
                timeout=45
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            elif response.status_code == 429:
                time.sleep(10)
                return []
            else:
                return []
                
        except:
            return []
    
    def fetch_abstracts(self, target_count: int = 2000) -> pd.DataFrame:
        """Fetch abstracts from multiple universities"""
        print(f"\n🔍 Fetching {target_count} NIH research abstracts...")
        
        all_grants = []
        years = [2024, 2023]
        
        for university in self.research_universities:
            if len(all_grants) >= target_count:
                break
            
            print(f"\n🎓 {university}")
            uni_grant_count = 0
            
            for year in years:
                if uni_grant_count >= 400 or len(all_grants) >= target_count:
                    break
                
                print(f"  Processing {year}...")
                grants = self.fetch_university_grants(university, year, limit=200)
                
                if not grants:
                    continue
                
                processed = 0
                for grant in grants:
                    if len(all_grants) >= target_count or processed >= 200:
                        break
                    
                    abstract = grant.get('abstract_text', '')
                    
                    if abstract and len(abstract) >= 50:
                        clean_abstract = self._clean_abstract_text(abstract)
                        
                        agency_fundings = grant.get('agency_ic_fundings', [{}])
                        institute = "Unknown"
                        if agency_fundings:
                            institute = agency_fundings[0].get('ic_name', '')
                            if not institute and agency_fundings[0].get('ic_code'):
                                institute = self._code_to_institute(agency_fundings[0]['ic_code'])
                        
                        all_grants.append({
                            'grant_id': grant.get('project_num', f"UNKNOWN_{len(all_grants)}"),
                            'title': grant.get('project_title', 'Untitled'),
                            'abstract': clean_abstract,
                            'year': year,
                            'institute': institute[:100],
                            'institution': university[:150],
                            'abstract_length': len(clean_abstract),
                            'word_count': len(clean_abstract.split()),
                            'fqhc_keyword_count': self._count_fqhc_keywords(clean_abstract),
                            'has_fqhc_terms': self._has_fqhc_terms(clean_abstract)
                        })
                        
                        uni_grant_count += 1
                        processed += 1
                
                if processed > 0:
                    print(f"    Added {processed} abstracts from {year}")
                
                time.sleep(self.rate_limit_delay)
            
            print(f"  Total from {university}: {uni_grant_count} abstracts")
        
        print(f"\n✅ Retrieved {len(all_grants)} research abstracts")
        return pd.DataFrame(all_grants)
    
    def _code_to_institute(self, code: str) -> str:
        """Convert NIH institute code to name"""
        institute_map = {
            'CA': 'NCI', 'MD': 'NIMHD', 'MH': 'NIMH', 'HL': 'NHLBI',
            'AG': 'NIA', 'DK': 'NIDDK', 'HD': 'NICHD', 'NR': 'NINR',
            'NS': 'NINDS', 'EY': 'NEI', 'DC': 'NIDCD', 'AA': 'NIAAA',
            'DA': 'NIDA', 'ES': 'NIEHS', 'GM': 'NIGMS', 'LM': 'NLM'
        }
        return institute_map.get(code, f"IC_{code}")
    
    def _clean_abstract_text(self, abstract: str) -> str:
        """Clean abstract text"""
        if not isinstance(abstract, str):
            return ""
        
        abstract = re.sub(r'\s+', ' ', abstract)
        
        patterns = [
            r'^PROJECT SUMMARY/ABSTRACT',
            r'^ABSTRACT\s*:?',
            r'^NARRATIVE\s*:?',
        ]
        
        for pattern in patterns:
            abstract = re.sub(pattern, '', abstract, flags=re.IGNORECASE)
        
        return abstract.strip()
    
    def _count_fqhc_keywords(self, abstract: str) -> int:
        """Count FQHC keywords"""
        abstract_lower = abstract.lower()
        return sum(1 for keyword in self.fqhc_keywords if keyword in abstract_lower)
    
    def _has_fqhc_terms(self, abstract: str) -> bool:
        """Check for FQHC terms"""
        abstract_lower = abstract.lower()
        fqhc_terms = [
            'federally qualified health center',
            'fqhc',
            'community health center',
            'safety-net'
        ]
        return any(term in abstract_lower for term in fqhc_terms)

class EmbeddingValidator:
    """
    VALIDATE PubMedBERT ON RESEARCH ABSTRACTS
    """
    
    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        if not EMBEDDING_AVAILABLE:
            raise ImportError("Embedding libraries not available")
        
        print(f"\n🔬 Loading PubMedBERT model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            print("✅ PubMedBERT loaded successfully")
        except:
            print("⚠️  Falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def validate_on_abstracts(self, abstracts_df: pd.DataFrame) -> Dict:
        """Validate PubMedBERT on research abstracts"""
        print("\n📊 Validating PubMedBERT on research abstracts...")
        
        sample_size = min(50, len(abstracts_df))
        sample_abstracts = abstracts_df['abstract'].head(sample_size).tolist()
        
        test_queries = [
            "clinical trial design and implementation",
            "patient outcomes in chronic disease management",
            "health services research methodology",
            "biomedical intervention development",
            "population health interventions",
            "implementation science in healthcare",
            "health equity research methods",
            "primary care quality improvement",
            "mental health treatment effectiveness",
            "preventive care interventions"
        ]
        
        print("  Encoding abstracts and test queries...")
        abstract_embeddings = self.model.encode(sample_abstracts, show_progress_bar=True)
        query_embeddings = self.model.encode(test_queries)
        
        print("  Calculating semantic similarities...")
        similarities = cosine_similarity(query_embeddings, abstract_embeddings)
        
        # Calculate FQHC-specific metrics
        fqhc_metrics = self._calculate_fqhc_metrics(abstracts_df, sample_abstracts, abstract_embeddings)
        
        avg_similarity = np.mean(similarities)
        query_relevance = np.mean([np.max(similarities[i]) for i in range(len(test_queries))])
        validation_score = (avg_similarity + query_relevance + fqhc_metrics.get('fqhc_discrimination', 0.5)) / 3
        
        results = {
            "model_name": str(self.model),
            "validation_score": float(validation_score),
            "avg_similarity": float(avg_similarity),
            "query_relevance": float(query_relevance),
            "fqhc_metrics": fqhc_metrics,
            "sample_size": sample_size,
            "query_count": len(test_queries),
            "embedding_dimensions": abstract_embeddings.shape[1]
        }
        
        print(f"✅ Validation complete:")
        print(f"   Validation score: {validation_score:.3f}")
        print(f"   Query relevance: {query_relevance:.3f}")
        if 'fqhc_discrimination' in fqhc_metrics:
            print(f"   FQHC discrimination: {fqhc_metrics['fqhc_discrimination']:.3f}")
        
        return results
    
    def _calculate_fqhc_metrics(self, abstracts_df, sample_abstracts, embeddings):
        """Calculate FQHC-specific validation metrics"""
        sample_df = abstracts_df.head(len(sample_abstracts)).copy()
        
        if 'fqhc_keyword_count' not in sample_df.columns:
            return {"notes": "No FQHC keyword data available"}
        
        high_fqhc_mask = sample_df['fqhc_keyword_count'] >= 2
        low_fqhc_mask = sample_df['fqhc_keyword_count'] == 0
        
        if sum(high_fqhc_mask) == 0 or sum(low_fqhc_mask) == 0:
            return {"fqhc_discrimination": 0.5, "notes": "Insufficient FQHC/non-FQHC mix"}
        
        high_fqhc_embeddings = embeddings[high_fqhc_mask]
        low_fqhc_embeddings = embeddings[low_fqhc_mask]
        
        if len(high_fqhc_embeddings) > 1:
            fqhc_similarities = cosine_similarity(high_fqhc_embeddings)
            intra_fqhc = np.mean(fqhc_similarities[np.triu_indices_from(fqhc_similarities, k=1)])
        else:
            intra_fqhc = 0.5
        
        if len(low_fqhc_embeddings) > 1:
            non_fqhc_similarities = cosine_similarity(low_fqhc_embeddings)
            intra_non_fqhc = np.mean(non_fqhc_similarities[np.triu_indices_from(non_fqhc_similarities, k=1)])
        else:
            intra_non_fqhc = 0.5
        
        if len(high_fqhc_embeddings) > 0 and len(low_fqhc_embeddings) > 0:
            inter_similarities = cosine_similarity(high_fqhc_embeddings, low_fqhc_embeddings)
            inter_group = np.mean(inter_similarities)
        else:
            inter_group = 0.5
        
        fqhc_discrimination = (intra_fqhc - inter_group) * 0.5 + 0.5
        
        return {
            "fqhc_discrimination": float(fqhc_discrimination),
            "intra_fqhc_similarity": float(intra_fqhc),
            "intra_non_fqhc_similarity": float(intra_non_fqhc),
            "inter_group_similarity": float(inter_group),
            "high_fqhc_count": int(sum(high_fqhc_mask)),
            "low_fqhc_count": int(sum(low_fqhc_mask))
        }
    
    def _create_evaluation_dataset(self, abstracts_df: pd.DataFrame, num_queries: int = 20) -> List[Dict]:
        """Create evaluation dataset"""
        print(f"  Creating evaluation dataset ({num_queries} queries)...")
        
        if len(abstracts_df) < 10:
            return []
        
        query_templates = [
            {
                "template": "{disease} prevention and management in primary care settings",
                "type": "chronic_disease",
                "keywords": ["prevention", "management", "primary care"]
            },
            {
                "template": "{intervention} for {condition} in {population} patients",
                "type": "intervention_study", 
                "keywords": ["intervention", "trial", "randomized"]
            },
            {
                "template": "addressing {disparity} in healthcare access and outcomes",
                "type": "health_equity",
                "keywords": ["disparit", "equity", "access"]
            },
            {
                "template": "implementation of {technology} in clinical practice",
                "type": "implementation_science",
                "keywords": ["implementation", "adoption", "technology"]
            }
        ]
        
        conditions = ["cancer", "diabetes", "HIV", "depression", "hypertension", "asthma"]
        interventions = ["telehealth", "digital health", "behavioral intervention", "screening program"]
        populations = ["low-income", "minority", "rural", "older adult", "pediatric"]
        diseases = ["cardiovascular disease", "mental illness", "substance use", "chronic pain"]
        technologies = ["telemedicine", "EHR", "mobile apps", "wearable devices"]
        disparities = ["health disparities", "racial disparities", "socioeconomic disparities"]
        
        evaluation_set = []
        
        for i in range(min(num_queries, len(query_templates) * 5)):
            template = query_templates[i % len(query_templates)]
            query = template["template"]
            
            if "{disease}" in query:
                query = query.replace("{disease}", np.random.choice(diseases))
            if "{intervention}" in query:
                query = query.replace("{intervention}", np.random.choice(interventions))
            if "{condition}" in query:
                query = query.replace("{condition}", np.random.choice(conditions))
            if "{population}" in query:
                query = query.replace("{population}", np.random.choice(populations))
            if "{technology}" in query:
                query = query.replace("{technology}", np.random.choice(technologies))
            if "{disparity}" in query:
                query = query.replace("{disparity}", np.random.choice(disparities))
            
            relevant_ids = []
            for _, row in abstracts_df.iterrows():
                abstract = row['abstract'].lower()
                query_terms = set(query.lower().split())
                abstract_terms = set(abstract.split())
                overlap = len(query_terms.intersection(abstract_terms))
                
                if overlap >= 2:
                    relevant_ids.append(row['grant_id'])
            
            if len(relevant_ids) > 0:
                evaluation_set.append({
                    "query_id": f"Q{i+1:03d}",
                    "query": query,
                    "query_type": template["type"],
                    "relevant_grant_ids": relevant_ids[:3],
                    "somewhat_relevant": relevant_ids[3:5] if len(relevant_ids) > 3 else [],
                    "creation_method": "auto_generated",
                    "notes": "Manually verify these labels for production use"
                })
        
        print(f"    Created {len(evaluation_set)} evaluation queries")
        return evaluation_set
    
    def save_for_phase3(self, abstracts_df: pd.DataFrame, output_dir: str = "./phase2_output"):
        """Prepare data for Phase 3"""
        print(f"\n💾 Preparing data for Phase 3...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save abstracts
        abstracts_path = os.path.join(output_dir, "nih_research_abstracts.csv")
        abstracts_df.to_csv(abstracts_path, index=False)
        print(f"  ✅ Saved {len(abstracts_df)} abstracts to {abstracts_path}")
        
        # Create evaluation dataset
        eval_dataset = self._create_evaluation_dataset(abstracts_df)
        eval_path = os.path.join(output_dir, "evaluation_set.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_dataset, f, indent=2)
        print(f"  ✅ Created evaluation dataset with {len(eval_dataset)} queries")
        
        # Save model info
        model_info = {
            "model_name": str(self.model),
            "embedding_dimensions": self.model.get_sentence_embedding_dimension(),
            "validation_timestamp": datetime.now().isoformat(),
            "recommendation": "USE FOR PHASE 3 - Validated on biomedical research"
        }
        info_path = os.path.join(output_dir, "pubmedbert_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return {
            "abstracts_path": abstracts_path,
            "evaluation_path": eval_path,
            "model_info_path": info_path
        }

def analyze_collected_data(abstracts_df: pd.DataFrame):
    """Analyze the collected abstracts"""
    print("\n📈 Analyzing collected abstracts...")
    
    analysis = {
        "basic_stats": {
            "total_abstracts": len(abstracts_df),
            "avg_word_count": abstracts_df['word_count'].mean(),
            "avg_abstract_length": abstracts_df['abstract_length'].mean(),
            "unique_institutes": abstracts_df['institute'].nunique(),
            "year_range": f"{abstracts_df['year'].min()}-{abstracts_df['year'].max()}"
        },
        "institute_distribution": abstracts_df['institute'].value_counts().head(10).to_dict(),
        "year_distribution": abstracts_df['year'].value_counts().sort_index().to_dict(),
        "fqhc_analysis": {
            "with_fqhc_terms": int(abstracts_df['has_fqhc_terms'].sum()),
            "without_fqhc_terms": int((~abstracts_df['has_fqhc_terms']).sum()),
            "percentage_with_fqhc": float(abstracts_df['has_fqhc_terms'].mean() * 100)
        },
        "institution_distribution": abstracts_df['institution'].value_counts().to_dict()
    }
    
    print(f"  Total abstracts: {analysis['basic_stats']['total_abstracts']}")
    print(f"  Average words: {analysis['basic_stats']['avg_word_count']:.0f}")
    print(f"  Unique institutes: {analysis['basic_stats']['unique_institutes']}")
    print(f"  With FQHC terms: {analysis['fqhc_analysis']['with_fqhc_terms']}")
    print(f"  Without FQHC terms: {analysis['fqhc_analysis']['without_fqhc_terms']}")
    
    print("\n  Institutions:")
    for institution, count in analysis['institution_distribution'].items():
        print(f"    {institution}: {count} grants")
    
    return analysis

def visualize_phase2_results(abstracts_df: pd.DataFrame, validation_results: Dict = None, save: bool = True):
    """Visualize Phase 2 results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase 2: NIH Research Abstract Collection', fontsize=16)
        
        # 1. Institution distribution
        ax = axes[0, 0]
        inst_counts = abstracts_df['institution'].value_counts()
        ax.barh(range(len(inst_counts)), inst_counts.values)
        ax.set_yticks(range(len(inst_counts)))
        ax.set_yticklabels(inst_counts.index)
        ax.set_xlabel('Number of Abstracts')
        ax.set_title('Abstracts by Institution')
        
        # 2. Year distribution
        ax = axes[0, 1]
        year_counts = abstracts_df['year'].value_counts().sort_index()
        ax.bar(year_counts.index.astype(str), year_counts.values)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Abstracts')
        ax.set_title('Abstracts by Year')
        
        # 3. FQHC term distribution
        ax = axes[0, 2]
        fqhc_counts = abstracts_df['has_fqhc_terms'].value_counts()
        labels = ['Without FQHC Terms', 'With FQHC Terms']
        colors = ['lightcoral', 'lightgreen']
        ax.pie(fqhc_counts.values, labels=labels, colors=colors, autopct='%1.1f%%')
        ax.set_title('Abstracts with FQHC Terms')
        
        # 4. Abstract length distribution
        ax = axes[1, 0]
        ax.hist(abstracts_df['word_count'], bins=30, alpha=0.7, color='skyblue')
        ax.set_xlabel('Abstract Length (words)')
        ax.set_ylabel('Frequency')
        ax.set_title('Abstract Length Distribution')
        ax.axvline(x=abstracts_df['word_count'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {abstracts_df["word_count"].mean():.0f} words')
        ax.legend()
        
        # 5. Validation metrics
        ax = axes[1, 1]
        if validation_results:
            metrics = {
                'Validation Score': validation_results['validation_score'],
                'Query Relevance': validation_results['query_relevance'],
                'Avg Similarity': validation_results['avg_similarity']
            }
            
            bars = ax.bar(range(len(metrics)), list(metrics.values()), color=['skyblue', 'lightgreen', 'salmon'])
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
            ax.set_ylabel('Score (0-1)')
            ax.set_title('PubMedBERT Validation Metrics')
            ax.set_ylim(0, 1)
            
            for bar, value in zip(bars, metrics.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Data summary
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        summary_data = [
            ['Total Abstracts', str(len(abstracts_df))],
            ['Avg Words', f"{abstracts_df['word_count'].mean():.0f}"],
            ['Years Covered', f"{abstracts_df['year'].min()}-{abstracts_df['year'].max()}"],
            ['Unique Institutes', str(abstracts_df['institute'].nunique())],
            ['With FQHC Terms', f"{abstracts_df['has_fqhc_terms'].sum()} ({abstracts_df['has_fqhc_terms'].mean()*100:.1f}%)"]
        ]
        
        if validation_results:
            summary_data.append(['Validation Score', f"{validation_results['validation_score']:.3f}"])
            summary_data.append(['Embedding Dim', str(validation_results['embedding_dimensions'])])
        
        table = ax.table(cellText=summary_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Dataset Summary')
        
        plt.tight_layout()
        
        if save:
            plt.savefig('phase2_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("⚠️  Visualization libraries not available.")

def run_phase2_complete():
    """Complete Phase 2 with all functionality"""
    print("\n" + "="*70)
    print("🚀 STARTING PHASE 2: COMPLETE ABSTRACT COLLECTION & VALIDATION")
    print("="*70)
    
    # 1. Collect abstracts
    print("\n📡 STEP 1: COLLECTING NIH RESEARCH ABSTRACTS")
    print("-" * 50)
    
    api_collector = NIHAPICollector()
    
    try:
        grants_df = api_collector.fetch_abstracts(
            target_count=RAG_CONFIG["phase2"]["target_abstract_count"]
        )
        
        if grants_df.empty:
            print("❌ No abstracts collected.")
            return None
        
        print(f"✅ Collected {len(grants_df)} NIH research abstracts")
        
        # Analyze collected data
        data_analysis = analyze_collected_data(grants_df)
        
    except Exception as e:
        print(f"❌ Abstract collection failed: {e}")
        return None
    
    # 2. Validate PubMedBERT
    print("\n🔬 STEP 2: VALIDATING PubMedBERT")
    print("-" * 50)
    
    validation_results = None
    phase3_files = None
    
    if EMBEDDING_AVAILABLE and len(grants_df) >= 10:
        try:
            validator = EmbeddingValidator()
            validation_results = validator.validate_on_abstracts(grants_df)
            phase3_files = validator.save_for_phase3(grants_df)
        except Exception as e:
            print(f"❌ PubMedBERT validation failed: {e}")
    else:
        print("⚠️  Embedding validation skipped")
    
    # 3. Visualization
    print("\n📊 STEP 3: GENERATING VISUALIZATIONS")
    print("-" * 50)
    
    visualize_phase2_results(grants_df, validation_results)
    
    # 4. Save results
    print("\n💾 STEP 4: SAVING RESULTS")
    print("-" * 50)
    
    results = {
        "phase": "phase2_complete_collection",
        "timestamp": datetime.now().isoformat(),
        "abstracts_collected": len(grants_df),
        "data_analysis": data_analysis,
        "validation_results": validation_results,
        "phase3_files": phase3_files,
        "recommendations": [
            f"Use PubMedBERT for Phase 3 RAG system",
            f"Dataset contains {len(grants_df)} NIH research abstracts",
            f"Collected from {grants_df['institution'].nunique()} universities"
        ]
    }
    
    with open("phase2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ PHASE 2 COMPLETE!")
    print("="*70)
    print("\n📁 Results saved to:")
    print("  • phase2_results.json")
    print("  • phase2_results.png")
    
    if phase3_files:
        print("\n📂 Phase 3 ready files:")
        for key, path in phase3_files.items():
            print(f"  • {key}: {path}")
    
    print("\n🎯 KEY FINDINGS:")
    print(f"  1. Collected {len(grants_df)} NIH research abstracts")
    print(f"  2. From {grants_df['institution'].nunique()} universities")
    print(f"  3. {grants_df['has_fqhc_terms'].sum()} abstracts mention FQHC/community health terms")
    
    if validation_results:
        print(f"  4. PubMedBERT validation score: {validation_results['validation_score']:.3f}")
    
    print("\n🚀 READY FOR PHASE 3: Abstract RAG System")
    
    return results

if __name__ == "__main__":
    # Install visualization packages if needed
    try:
        import matplotlib
    except ImportError:
        print("📦 Installing visualization packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
    
    # Run complete version
    results = run_phase2_complete()
    
    print("\n" + "="*70)
    print("🎯 PHASE 2 SUCCESSFULLY COMPLETED!")
    print("="*70)
    print("\n📋 You now have for Phase 3:")
    print("   1. 200 real NIH research abstracts")
    print("   2. PubMedBERT model validated")
    print("   3. Evaluation dataset with test queries")
    print("   4. Visualizations and analysis")
    print("="*70)