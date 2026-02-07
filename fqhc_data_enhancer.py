# fqhc_data_enhancer.py
"""
BEFORE RUNNING ANY PHASE, CREATE ENHANCED DATASET
"""

def create_enhanced_fqhc_dataset():
    # 1. Load your Phase 2 NIH abstracts
    phase2_data = pd.read_csv('./phase2_output/nih_research_abstracts.csv')
    
    # 2. Add targeted FQHC searches (enhanced NIH API queries)
    fqhc_targeted = search_fqhc_specific_grants()  # New function
    
    # 3. Add synthetic FQHC examples (for testing)
    synthetic_fqhc = generate_fqhc_synthetic_data(50)
    
    # 4. Combine (300-350 total documents)
    enhanced_data = pd.concat([phase2_data, fqhc_targeted, synthetic_fqhc])
    
    # 5. Add FQHC metadata flags
    enhanced_data['is_fqhc_focused'] = enhanced_data.apply(detect_fqhc_focus, axis=1)
    enhanced_data['fqhc_score'] = enhanced_data.apply(calculate_fqhc_score, axis=1)
    
    return enhanced_data