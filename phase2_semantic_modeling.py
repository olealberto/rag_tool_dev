def generate_test_queries_from_abstracts(abstracts, titles, num_queries=10):
    """Generate realistic test queries based on actual abstract content"""
    
    # Extract key terms from titles and abstracts
    medical_terms = []
    for title, abstract in zip(titles, abstracts):
        # Simple extraction of likely key terms
        text = (title + " " + abstract).lower()
        
        # Look for disease/condition terms
        disease_terms = ['cancer', 'diabetes', 'hiv', 'depression', 'hypertension', 
                        'asthma', 'alzheimer', 'cardiovascular', 'mental health']
        
        for term in disease_terms:
            if term in text:
                medical_terms.append(term)
        
        # Look for intervention terms
        intervention_terms = ['telehealth', 'screening', 'intervention', 'therapy',
                            'medication', 'prevention', 'management', 'treatment']
        
        for term in intervention_terms:
            if term in text:
                medical_terms.append(term)
    
    # Create diverse queries
    queries = []
    query_templates = [
        "{disease} prevention and management",
        "{intervention} for {condition}",
        "clinical trials for {disease}",
        "health disparities in {population}",
        "{methodology} in health research",
        "patient outcomes for {condition}",
        "implementation of {technology} in healthcare",
        "cost-effectiveness of {intervention}",
        "quality of life in {disease} patients",
        "barriers to {service} access"
    ]
    
    populations = ['low-income', 'minority', 'older adults', 'pediatric', 'rural']
    methodologies = ['randomized controlled', 'qualitative', 'mixed methods', 'observational']
    technologies = ['telemedicine', 'digital health', 'EHR', 'mobile apps']
    services = ['mental health', 'preventive care', 'specialty care']
    
    for i in range(min(num_queries, len(query_templates))):
        template = query_templates[i]
        query = template
        
        # Fill template with actual terms from your data
        if '{disease}' in query and medical_terms:
            query = query.replace('{disease}', np.random.choice([t for t in medical_terms if t in ['cancer', 'diabetes', 'hiv', 'depression', 'hypertension']]))
        if '{condition}' in query and medical_terms:
            query = query.replace('{condition}', np.random.choice(medical_terms))
        if '{intervention}' in query:
            query = query.replace('{intervention}', np.random.choice([t for t in medical_terms if t in ['telehealth', 'screening', 'intervention', 'therapy']] + ['telehealth', 'screening']))
        if '{population}' in query:
            query = query.replace('{population}', np.random.choice(populations))
        if '{methodology}' in query:
            query = query.replace('{methodology}', np.random.choice(methodologies))
        if '{technology}' in query:
            query = query.replace('{technology}', np.random.choice(technologies))
        if '{service}' in query:
            query = query.replace('{service}', np.random.choice(services))
        
        queries.append(query)
    
    return queries[:num_queries]

# Generate realistic queries
test_queries = generate_test_queries_from_abstracts(test_abstracts[:20], titles[:20], num_queries=10)
print(f"\n🔍 Generated {len(test_queries)} realistic test queries:")
for i, q in enumerate(test_queries, 1):
    print(f"  {i}. {q}")
    
def test_models_on_real_data(models_to_test, abstracts, queries):
    """Test multiple models on real NIH abstracts"""
    
    results = {}
    
    for model_name, display_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING: {display_name}")
        print(f"   Model: {model_name}")
        print('='*60)
        
        try:
            # Load model with timing
            start_time = time.time()
            model = SentenceTransformer(model_name)
            load_time = time.time() - start_time
            print(f"   Loaded in {load_time:.1f}s")
            
            # Create embeddings
            start_time = time.time()
            abstract_embeddings = model.encode(abstracts, show_progress_bar=False)
            query_embeddings = model.encode(queries, show_progress_bar=False)
            encode_time = time.time() - start_time
            print(f"   Encoded {len(abstracts)} abstracts + {len(queries)} queries in {encode_time:.1f}s")
            
            # Calculate similarities
            similarities = cosine_similarity(query_embeddings, abstract_embeddings)
            
            # Calculate metrics
            # 1. Query relevance (how well queries match relevant abstracts)
            relevance_scores = []
            for i, query in enumerate(queries):
                # Find abstract that should be most relevant (simplified)
                query_terms = set(query.lower().split())
                best_match_score = 0
                for j, abstract in enumerate(abstracts):
                    abstract_terms = set(abstract.lower().split())
                    term_overlap = len(query_terms.intersection(abstract_terms))
                    if term_overlap > 2:  # Has meaningful overlap
                        relevance_scores.append(similarities[i][j])
            
            avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
            
            # 2. Self-similarity (abstracts should be somewhat similar to each other)
            abstract_similarities = cosine_similarity(abstract_embeddings)
            np.fill_diagonal(abstract_similarities, 0)  # Remove self-similarity
            avg_abstract_similarity = np.mean(abstract_similarities)
            
            # 3. Query discrimination (queries should be distinct)
            query_similarities = cosine_similarity(query_embeddings)
            np.fill_diagonal(query_similarities, 0)
            avg_query_similarity = np.mean(query_similarities)
            
            # 4. Top-k retrieval simulation
            top_k_scores = []
            for i in range(len(queries)):
                top_indices = np.argsort(similarities[i])[-5:][::-1]  # Top 5
                # Simple relevance check (does top result have term overlap?)
                top_abstract = abstracts[top_indices[0]]
                query_terms = set(queries[i].lower().split())
                abstract_terms = set(top_abstract.lower().split())
                overlap = len(query_terms.intersection(abstract_terms))
                top_k_scores.append(min(1.0, overlap / 5))  # Normalize
            
            avg_top_k = np.mean(top_k_scores)
            
            # Overall score (weighted)
            overall_score = (avg_relevance * 0.4 + avg_top_k * 0.3 + 
                           (1 - avg_query_similarity) * 0.2 + avg_abstract_similarity * 0.1)
            
            results[display_name] = {
                'model': model_name,
                'overall_score': overall_score,
                'relevance': avg_relevance,
                'top_k_score': avg_top_k,
                'abstract_similarity': avg_abstract_similarity,
                'query_discrimination': 1 - avg_query_similarity,
                'embedding_dim': abstract_embeddings.shape[1],
                'encode_time': encode_time,
                'load_time': load_time
            }
            
            print(f"   📊 Results:")
            print(f"      • Overall score: {overall_score:.3f}")
            print(f"      • Query relevance: {avg_relevance:.3f}")
            print(f"      • Top-5 accuracy: {avg_top_k:.3f}")
            print(f"      • Embedding dim: {abstract_embeddings.shape[1]}")
            print(f"      • Speed: {encode_time:.1f}s total")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}")
            results[display_name] = {'error': str(e), 'overall_score': 0}
    
    return results

# Define models to test (biology-focused + general)
models_to_test = [
    ("allenai/specter2", "SPECTER2 (Research Papers)"),
    ("allenai/specter", "SPECTER (Research Papers)"),
    ("nlpie/bio-sbert-nli", "Bio-SBERT-NLI"),
    ("pritamdeka/S-PubMedBert-MS-MARCO", "PubMedBERT-MS-MARCO"),
    ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "PubMedBERT-Abstract"),
    ("dmis-lab/biobert-v1.1", "BioBERT"),
    ("all-mpnet-base-v2", "General MPNet"),
    ("all-MiniLM-L6-v2", "MiniLM (Fast)")
]

# Run tests
print("\n" + "="*70)
print("🧬 COMPREHENSIVE MODEL TESTING WITH REAL NIH ABSTRACTS")
print("="*70)

results = test_models_on_real_data(models_to_test, test_abstracts, test_queries)

def visualize_model_comparison(results):
    """Create visualization of model comparison"""
    
    # Filter successful tests
    successful = {k: v for k, v in results.items() if 'overall_score' in v and v['overall_score'] > 0}
    
    if not successful:
        print("No successful model tests to visualize")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Biology Model Comparison for NIH Abstract RAG', fontsize=16, fontweight='bold')
    
    models = list(successful.keys())
    scores = [successful[m]['overall_score'] for m in models]
    relevance = [successful[m].get('relevance', 0) for m in models]
    top_k = [successful[m].get('top_k_score', 0) for m in models]
    dims = [successful[m].get('embedding_dim', 0) for m in models]
    encode_times = [successful[m].get('encode_time', 0) for m in models]
    
    # 1. Overall scores
    ax = axes[0, 0]
    bars = ax.barh(range(len(models)), scores, color='skyblue')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Overall Score')
    ax.set_title('Model Performance Ranking')
    ax.set_xlim(0, 1)
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center')
    
    # 2. Relevance scores
    ax = axes[0, 1]
    bars = ax.barh(range(len(models)), relevance, color='lightgreen')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Query Relevance')
    ax.set_title('Semantic Relevance to Queries')
    ax.set_xlim(0, 1)
    
    # 3. Top-K accuracy
    ax = axes[0, 2]
    bars = ax.barh(range(len(models)), top_k, color='salmon')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Top-5 Accuracy')
    ax.set_title('Retrieval Accuracy')
    ax.set_xlim(0, 1)
    
    # 4. Embedding dimensions
    ax = axes[1, 0]
    bars = ax.barh(range(len(models)), dims, color='gold')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Dimensions')
    ax.set_title('Embedding Size')
    
    # 5. Encoding speed
    ax = axes[1, 1]
    bars = ax.barh(range(len(models)), encode_times, color='lightcoral')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Encoding Speed (50 abstracts)')
    
    # 6. Recommendations
    ax = axes[1, 2]
    ax.axis('off')
    
    # Find best models
    if successful:
        best_overall = max(successful.items(), key=lambda x: x[1]['overall_score'])
        best_fast = min([(k, v) for k, v in successful.items()], 
                       key=lambda x: x[1].get('encode_time', 100))
        best_balance = max([(k, v) for k, v in successful.items() 
                          if v.get('encode_time', 100) < 5], 
                          key=lambda x: x[1]['overall_score'])
        
        recommendations = [
            ['Metric', 'Best Model', 'Score'],
            ['Overall', best_overall[0], f"{best_overall[1]['overall_score']:.3f}"],
            ['Fastest', best_fast[0], f"{best_fast[1].get('encode_time', 0):.1f}s"],
            ['Balanced', best_balance[0], f"{best_balance[1]['overall_score']:.3f}"]
        ]
        
        table = ax.table(cellText=recommendations, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Recommendations for Phase 3')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return successful

# Visualize results
print("\n" + "="*70)
print("📊 GENERATING VISUAL COMPARISON")
print("="*70)

successful_models = visualize_model_comparison(results)

def get_phase3_recommendation(results):
    """Get specific recommendation for Phase 3 RAG system"""
    
    successful = {k: v for k, v in results.items() if 'overall_score' in v and v['overall_score'] > 0}
    
    if not successful:
        return "Use all-mpnet-base-v2 (fallback - no biology models loaded)"
    
    print("\n" + "="*70)
    print("🎯 PHASE 3 MODEL RECOMMENDATION")
    print("="*70)
    
    # Rank by overall score
    ranked = sorted(successful.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    
    print("\n🏆 TOP 3 MODELS FOR YOUR NIH ABSTRACTS:")
    for i, (name, metrics) in enumerate(ranked[:3], 1):
        print(f"\n{i}. {name}:")
        print(f"   • Overall score: {metrics['overall_score']:.3f}")
        print(f"   • Query relevance: {metrics.get('relevance', 0):.3f}")
        print(f"   • Encoding time: {metrics.get('encode_time', 0):.1f}s")
        print(f"   • Dimensions: {metrics.get('embedding_dim', 0)}")
        print(f"   • Model: {metrics['model']}")
    
    # Recommendation logic
    best = ranked[0]
    
    print(f"\n💡 RECOMMENDATION FOR PHASE 3:")
    print(f"   Use: {best[0]}")
    print(f"   Model: {best[1]['model']}")
    
    if "SPECTER" in best[0]:
        print(f"   Why: SPECTER is specifically designed for research papers like NIH abstracts")
    elif "Bio" in best[0] or "PubMed" in best[0]:
        print(f"   Why: Biomedical-specific model performs best with your NIH data")
    else:
        print(f"   Why: General model performs surprisingly well with your data")
    
    print(f"\n📝 Phase 3 implementation:")
    print(f'   model = SentenceTransformer("{best[1]["model"]}")')
    
    return best[1]['model']

# Get recommendation
recommended_model = get_phase3_recommendation(results)

# Save recommendation
with open('phase3_model_recommendation.txt', 'w') as f:
    f.write(f"Recommended model for Phase 3 RAG:\n")
    f.write(f"Model: {recommended_model}\n")
    f.write(f"Based on testing with {len(test_abstracts)} real NIH abstracts\n")
    f.write(f"Test date: {pd.Timestamp.now()}\n")