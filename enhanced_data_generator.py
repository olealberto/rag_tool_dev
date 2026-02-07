# =========== IMPROVED SYNTHETIC DATA GENERATOR ===========
import pandas as pd
import numpy as np
from typing import List, Dict
import json
import os

class EnhancedFQHCDataGenerator:
    """Generate high-quality, specific FQHC synthetic documents"""
    
    def __init__(self):
        # Realistic FQHC components
        self.fqhc_settings = [
            "Urban Federally Qualified Health Centers",
            "Rural Community Health Centers", 
            "Safety-net clinics in underserved areas",
            "Medicaid-focused health centers",
            "Health centers serving migrant populations"
        ]
        
        self.populations = [
            "Latino/Latina patients",
            "African American communities",
            "Low-income Medicaid recipients",
            "Uninsured populations",
            "Limited English proficiency patients",
            "Rural residents",
            "Homeless individuals",
            "LGBTQ+ youth"
        ]
        
        self.conditions = {
            "diabetes": {
                "interventions": [
                    "Community Health Worker-led diabetes self-management",
                    "Group medical visits for diabetes care",
                    "Culturally-adapted nutrition education",
                    "Telehealth monitoring for blood glucose",
                    "Medication therapy management"
                ],
                "outcomes": ["HbA1c levels", "blood glucose control", "weight reduction", "medication adherence"],
                "metrics": ["% reduction in HbA1c", "BP control rates", "ER visits avoided"]
            },
            "hypertension": {
                "interventions": [
                    "Home blood pressure monitoring programs",
                    "Pharmacist-led medication management",
                    "Salt reduction education",
                    "Exercise prescription programs"
                ],
                "outcomes": ["blood pressure control", "medication adherence", "cardiovascular events"],
                "metrics": ["BP < 140/90 mmHg", "medication possession ratio", "hospitalizations"]
            },
            "depression": {
                "interventions": [
                    "Integrated behavioral health in primary care",
                    "Collaborative care models",
                    "Telemental health services",
                    "Peer support specialists"
                ],
                "outcomes": ["PHQ-9 scores", "treatment engagement", "quality of life"],
                "metrics": ["% with PHQ-9 < 10", "follow-up visit rate", "patient satisfaction"]
            },
            "asthma": {
                "interventions": [
                    "Asthma action plan education",
                    "Environmental trigger reduction",
                    "School-based asthma management",
                    "Community health worker home visits"
                ],
                "outcomes": ["asthma control test scores", "rescue inhaler use", "school/work days missed"],
                "metrics": ["ACT score improvement", "ER visits reduction", "inhaler technique"]
            }
        }
        
        self.study_designs = [
            "Randomized controlled trial",
            "Pragmatic clinical trial",
            "Stepped-wedge design",
            "Cluster randomized trial", 
            "Mixed-methods implementation study",
            "Quasi-experimental design"
        ]
        
        self.funding_institutes = ["NIMHD", "NIMH", "NCI", "NHLBI", "NIA", "NIDDK", "AHRQ"]
    
    def generate_fqhc_grant(self, condition: str, grant_id_prefix: str = "FQHC_SYNTH") -> Dict:
        """Generate a single high-quality FQHC grant document"""
        if condition not in self.conditions:
            condition = "diabetes"  # default
        
        cond_info = self.conditions[condition]
        
        # Select specific components
        setting = np.random.choice(self.fqhc_settings)
        population = np.random.choice(self.populations)
        intervention = np.random.choice(cond_info["interventions"])
        outcome = np.random.choice(cond_info["outcomes"])
        study_design = np.random.choice(self.study_designs)
        institute = np.random.choice(self.funding_institutes)
        
        # Generate realistic title
        title = f"{intervention} for {condition.capitalize()} Management in {population} at {setting}"
        
        # Generate detailed abstract
        abstract = f"""This {study_design.lower()} evaluates the effectiveness of {intervention.lower()} 
        for {condition} management among {population.lower()} receiving care at {setting.lower()}. 
        
        The intervention includes {np.random.choice(['culturally-tailored', 'evidence-based', 'patient-centered'])} 
        components focused on {np.random.choice(['health equity', 'quality improvement', 'access to care'])}. 
        
        {np.random.choice(['500', '750', '1000', '1200'])} participants will be enrolled across 
        {np.random.choice(['5', '8', '10', '12'])} health centers with follow-up over 
        {np.random.choice(['12', '18', '24'])} months. 
        
        Primary outcomes include changes in {outcome}, with secondary measures of 
        {np.random.choice(cond_info['metrics'])} and {np.random.choice(['cost-effectiveness', 'implementation fidelity', 'patient satisfaction'])}. 
        
        Results will inform dissemination of {condition} interventions in safety-net settings."""
        
        # Clean up whitespace
        abstract = ' '.join(abstract.split())
        
        return {
            'grant_id': f"{grant_id_prefix}_{condition.upper()}_{np.random.randint(1000, 9999)}",
            'title': title,
            'abstract': abstract,
            'year': np.random.choice([2022, 2023, 2024]),
            'institute': institute,
            'institution': f"SYNTHETIC_FQHC_{institute}_RESEARCH",
            'abstract_length': len(abstract),
            'word_count': len(abstract.split()),
            'is_fqhc_focused': True,
            'fqhc_score': 0.9 + np.random.random() * 0.1,  # High FQHC relevance
            'data_source': 'enhanced_synthetic_fqhc',
            'primary_condition': condition,
            'study_design': study_design,
            'target_population': population
        }
    
    def generate_dataset(self, n_per_condition: int = 10) -> pd.DataFrame:
        """Generate balanced dataset with multiple conditions"""
        all_grants = []
        
        for condition in self.conditions.keys():
            print(f"Generating {n_per_condition} {condition} grants...")
            for i in range(n_per_condition):
                grant = self.generate_fqhc_grant(condition, grant_id_prefix=f"ENH_FQHC")
                all_grants.append(grant)
        
        df = pd.DataFrame(all_grants)
        print(f"\n✅ Generated {len(df)} enhanced FQHC synthetic documents")
        print(f"   Conditions: {list(self.conditions.keys())}")
        print(f"   Average word count: {df['word_count'].mean():.0f}")
        
        return df

# Test the generator
print("🧪 Testing enhanced synthetic data generator...")
generator = EnhancedFQHCDataGenerator()
test_grant = generator.generate_fqhc_grant("diabetes")
print(f"\n📄 Sample enhanced document:")
print(f"Title: {test_grant['title']}")
print(f"Abstract: {test_grant['abstract'][:200]}...")

# =========== SAFE ENHANCED DATA REPLACEMENT ===========
import pandas as pd
import os
import shutil
from datetime import datetime

def safely_replace_phase3_data():
    """Safely replace Phase 3 dataset with enhanced version"""
    
    # 1. Create backup timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. Paths
    original_path = './phase3_data/enhanced_fqhc_dataset.csv'
    enhanced_path = './enhanced_phase3_data/fqhc_enhanced_dataset.csv'
    backup_path = f'./phase3_data/backup_enhanced_fqhc_dataset_{timestamp}.csv'
    
    print("🔒 SAFE DATASET REPLACEMENT")
    print("=" * 50)
    
    # 3. Check if enhanced data exists
    if not os.path.exists(enhanced_path):
        print(f"❌ Enhanced data not found at: {enhanced_path}")
        print("   Run the enhanced data generator first!")
        return False
    
    # 4. Backup original if exists
    if os.path.exists(original_path):
        shutil.copy2(original_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        
        # Show original stats
        original_data = pd.read_csv(original_path)
        print(f"   Original dataset: {len(original_data)} documents")
        print(f"   FQHC-focused: {original_data['is_fqhc_focused'].sum()}")
    else:
        print("ℹ️  No original dataset found (creating new)")
    
    # 5. Show enhanced stats
    enhanced_data = pd.read_csv(enhanced_path)
    print(f"\n📊 Enhanced dataset stats:")
    print(f"   Total documents: {len(enhanced_data)}")
    print(f"   FQHC-focused: {enhanced_data['is_fqhc_focused'].sum()}")
    
    if 'primary_condition' in enhanced_data.columns:
        conditions = enhanced_data['primary_condition'].dropna().unique()
        print(f"   Conditions covered: {list(conditions)}")
    
    # 6. Ask for confirmation
    print("\n⚠️  CONFIRM REPLACEMENT:")
    print(f"   Will replace: {original_path}")
    print(f"   With enhanced: {enhanced_path}")
    
    # For Colab, we'll just proceed
    print("   Proceeding with replacement...")
    
    # 7. Replace the file
    shutil.copy2(enhanced_path, original_path)
    
    print(f"\n✅ SUCCESSFULLY REPLACED!")
    print(f"   Original: {original_path} (now enhanced)")
    print(f"   Backup: {backup_path}")
    
    return True

# Run the safe replacement
safely_replace_phase3_data()

# =========== REPLACE YOUR SYNTHETIC DATA ===========
def create_enhanced_fqhc_dataset():
    """Create a new enhanced dataset with better synthetic data"""
    
    # 1. Load your real Phase 2 NIH data
    try:
        phase2_data = pd.read_csv('./phase2_output/nih_research_abstracts.csv')
        print(f"📊 Loaded {len(phase2_data)} real NIH abstracts")
        
        # Mark FQHC focus
        phase2_data['is_fqhc_focused'] = phase2_data['abstract'].apply(
            lambda x: any(term in str(x).lower() for term in 
                         ['federally qualified health center', 'fqhc', 'community health center'])
        )
        phase2_data['fqhc_score'] = phase2_data['abstract'].apply(
            lambda x: 0.8 if any(term in str(x).lower() for term in 
                               ['federally qualified health center', 'fqhc']) 
                     else 0.3 if 'community health' in str(x).lower() 
                     else 0.0
        )
        phase2_data['data_source'] = 'phase2_nih'
        
    except Exception as e:
        print(f"⚠️ Could not load Phase 2 data: {e}")
        phase2_data = pd.DataFrame()
    
    # 2. Generate ENHANCED synthetic FQHC data
    print("\n🧪 Generating enhanced synthetic FQHC data...")
    generator = EnhancedFQHCDataGenerator()
    synthetic_data = generator.generate_dataset(n_per_condition=15)  # 15 per condition
    
    # 3. Combine datasets
    if not phase2_data.empty:
        enhanced_data = pd.concat([phase2_data, synthetic_data], ignore_index=True, sort=False)
    else:
        enhanced_data = synthetic_data
    
    # 4. Ensure all required columns exist
    required_columns = ['grant_id', 'title', 'abstract', 'year', 'institute', 
                       'is_fqhc_focused', 'fqhc_score', 'data_source']
    
    for col in required_columns:
        if col not in enhanced_data.columns:
            enhanced_data[col] = ''
    
    # 5. Save the enhanced dataset
    os.makedirs('./enhanced_phase3_data', exist_ok=True)
    output_path = './enhanced_phase3_data/fqhc_enhanced_dataset.csv'
    enhanced_data.to_csv(output_path, index=False)
    
    print(f"\n🎉 ENHANCED DATASET CREATED!")
    print(f"   Total documents: {len(enhanced_data)}")
    print(f"   Real NIH abstracts: {len(phase2_data) if not phase2_data.empty else 0}")
    print(f"   Enhanced synthetic FQHC: {len(synthetic_data)}")
    print(f"   FQHC-focused: {enhanced_data['is_fqhc_focused'].sum()}")
    print(f"   Saved to: {output_path}")
    
    # 6. Show dataset composition
    print(f"\n📊 Dataset composition:")
    if 'primary_condition' in enhanced_data.columns:
        condition_counts = enhanced_data['primary_condition'].value_counts()
        for condition, count in condition_counts.items():
            if pd.notna(condition):
                print(f"   • {condition}: {count} documents")
    
    return enhanced_data

# Create the enhanced dataset
enhanced_data = create_enhanced_fqhc_dataset()

# =========== TEST WITH YOUR EVALUATION QUERIES ===========
def test_enhanced_data_coverage(enhanced_data, test_queries):
    """Test if new synthetic data covers evaluation queries"""
    
    print("\n🔍 TESTING COVERAGE OF ENHANCED DATA")
    print("=" * 60)
    
    # Get FQHC documents from enhanced data
    fqhc_enhanced = enhanced_data[enhanced_data['is_fqhc_focused'] == True]
    
    print(f"Enhanced FQHC documents: {len(fqhc_enhanced)}")
    print(f"With primary_condition field: {fqhc_enhanced['primary_condition'].notna().sum()}")
    
    # Test each query
    for i, query in enumerate(test_queries[:5]):
        query_text = query if isinstance(query, str) else query.get('query', '')
        print(f"\nQuery {i+1}: {query_text[:50]}...")
        
        # Simple keyword matching
        query_words = set(query_text.lower().split())
        matches = []
        
        for _, doc in fqhc_enhanced.iterrows():
            doc_text = (str(doc.get('title', '')) + ' ' + str(doc.get('abstract', ''))).lower()
            doc_words = set(doc_text.split())
            overlap = len(query_words.intersection(doc_words))
            
            if overlap >= 3:  # Meaningful overlap
                matches.append({
                    'grant_id': doc['grant_id'],
                    'title': doc.get('title', '')[:40],
                    'condition': doc.get('primary_condition', 'unknown'),
                    'overlap': overlap,
                    'is_synthetic': 'ENH_FQHC' in str(doc.get('grant_id', ''))
                })
        
        # Show matches
        if matches:
            matches.sort(key=lambda x: x['overlap'], reverse=True)
            print(f"  Found {len(matches)} potential matches")
            for match in matches[:3]:
                source = "ENH_SYNTH" if match['is_synthetic'] else "REAL"
                print(f"    • [{source}] {match['grant_id']} ({match['condition']}): {match['title']}...")
        else:
            print(f"  ⚠️ No good matches found")

# Define your test queries
test_queries = [
    "diabetes prevention in community health centers",
    "behavioral health integration in primary care",
    "cancer screening in underserved populations",
    "community health worker programs for chronic disease",
    "telehealth implementation in rural health centers"
]

# Test coverage
test_enhanced_data_coverage(enhanced_data, test_queries)