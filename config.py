# ============================================================================
# 📁 config.py - ALL CONFIGURATION PARAMETERS
# ============================================================================

"""
EDIT THIS FILE TO CHANGE TEST PARAMETERS
"""

RAG_CONFIG = {
    "project": {
        "name": "FQHC Research RAG System",
        "version": "1.0",
        "description": "Retrieval-Augmented Generation for FQHC Research Abstracts",
        "phases": ["Phase 1: Foundation", "Phase 2: Data Collection", "Phase 3: RAG System"]
    },
    
    # ============ PHASE 1: FOUNDATION TESTS ============
    "phase1": {
        "test_sample_size": 5,  # Number of sample grants for Phase 1
        "validation_sample_size": 20,
        "test_chunk_sizes": [100, 250, 500],  # Words per chunk
        "test_overlaps": [20, 50, 100],       # Overlap between chunks
        
        # Updated with PubMed model
        "test_embedding_models": [
            {"name": "all-MiniLM-L6-v2", "dims": 384, "type": "general"},
            {"name": "all-mpnet-base-v2", "dims": 768, "type": "general"},
            {"name": "multi-qa-mpnet-base-dot-v1", "dims": 768, "type": "retrieval"},
            {"name": "pritamdeka/S-PubMedBert-MS-MARCO", "dims": 768, "type": "biomedical"},
            {"name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "dims": 768, "type": "biomedical"},
        ],
        
        "test_retrieval_strategies": [
            {"name": "simple_similarity", "params": {}},
            {"name": "max_marginal_relevance", "params": {"lambda_param": 0.7}},
            {"name": "hybrid_search", "params": {"alpha": 0.7}},
        ],
        
        "top_k_results": 5,     # Number of results to retrieve
        "synthetic_data": {
            "num_fqhc_grants": 2,
            "num_general_grants": 3,
            "min_abstract_length": 200,
            "max_abstract_length": 1000
        },
        
        "test_queries": [
            "diabetes prevention in Federally Qualified Health Centers",
            "community health worker interventions",
            "behavioral health integration in primary care",
            "telehealth implementation in rural health centers",
            "health disparities reduction strategies"
        ]
    },
    
    # ============ PHASE 2: ABSTRACT COLLECTION & VALIDATION ============
    "phase2": {
        "nih_api_url": "https://api.reporter.nih.gov/v2/projects/search",
        "target_abstract_count": 300,
        "min_abstract_length": 200,  # Characters
        "fiscal_years": [2022, 2023, 2024],
        
        # NIH Institutes
        "institutes_of_interest": ["NIMHD", "NIMH", "NCI", "NHLBI", "NIA", "NIDDK"],
        
        # Research universities for NIH grants
        "research_universities": [
            "UNIVERSITY OF ILLINOIS AT CHICAGO",
            "UNIVERSITY OF MICHIGAN",
            "UNIVERSITY OF CHICAGO",
            "NORTHWESTERN UNIVERSITY",
            "JOHNS HOPKINS UNIVERSITY",
            "UNIVERSITY OF CALIFORNIA, SAN FRANCISCO",
            "UNIVERSITY OF WASHINGTON",
            "COLUMBIA UNIVERSITY"
        ],
        
        # FQHC keywords for filtering
        "fqhc_keywords": [
            "federally qualified health center",
            "fqhc",
            "community health center",
            "safety-net",
            "medically underserved",
            "low-income",
            "uninsured",
            "medicaid",
            "health disparities",
            "primary care",
            "underserved population"
        ],
        
        # PubMed model for validation
        "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
        "alternative_models": [
            "allenai/specter",
            "allenai/specter2",
            "nlpie/bio-sbert-nli",
            "all-mpnet-base-v2"
        ],
        
        "validation": {
            "sample_size": 50,
            "test_queries": [
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
        },
        
        "embedding_training_epochs": 3,
        "batch_size": 32,
        "validation_split": 0.2,
    },
    
    # ============ PHASE 3: DOCUMENT RAG ============
    "phase3": {
        "document_chunk_size": 250,
        "document_overlap": 50,
        "max_documents": 50,  # For testing
        "chunking_method": "paragraph_aware",  # Options: fixed, paragraph_aware, semantic
        
        # Primary embedding model - Using PubMed model
        "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
        "embedding_dimension": 768,
        
        # Alternative models for comparison
        "alternative_models": [
            {"name": "allenai/specter", "type": "research_paper", "dims": 768},
            {"name": "all-mpnet-base-v2", "type": "general", "dims": 768},
            {"name": "nlpie/bio-sbert-nli", "type": "biomedical", "dims": 768}
        ],
        
        "retrieval": {
            "top_k": 5,
            "similarity_threshold": 0.5,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "rerank_enabled": False,
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        },
        
        "vector_store": {
            "type": "faiss",  # Options: faiss, chromadb, pinecone, weaviate
            "index_path": "./vector_store/faiss_index",
            "dimension": 768
        },
        
        "llm": {
            "provider": "openai",  # Options: openai, anthropic, cohere, local
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 500,
            "system_prompt": "You are a helpful research assistant specializing in Federally Qualified Health Centers (FQHCs) and community health research. Provide concise, evidence-based answers based on the provided NIH grant abstracts."
        }
    },
    
    # ============ VISUALIZATION SETTINGS ============
    "visualization": {
        "plot_style": "seaborn",  # Options: matplotlib, seaborn, plotly
        "save_figures": True,
        "figure_format": "png",   # Options: png, pdf, svg
        "figure_dpi": 300,
        "interactive_plots": False,  # Set True for Plotly
        "color_palette": "Set2",
    },
    
    # ============ EVALUATION METRICS ============
    "evaluation": {
        "relevance_threshold": 0.6,  # Cosine similarity threshold for relevance
        "precision_at_k": [1, 3, 5, 10],
        "manual_evaluation_samples": 10,  # Number of samples for manual review
        "retrieval_metrics": ["precision", "recall", "f1_score", "ndcg"],
        "generation_metrics": ["rouge", "bleu", "bertscore", "faithfulness"]
    },
    
    # ============ FILE PATHS ============
    "paths": {
        "phase1_output": "./phase1_output",
        "phase2_output": "./phase2_output",
        "phase3_output": "./phase3_output",
        "data": "./data",
        "models": "./models",
        "logs": "./logs",
        "vector_store": "./vector_store",
        "cache": "./cache"
    },
    
    # ============ LOGGING CONFIGURATION ============
    "logging": {
        "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "./logs/rag_system.log",
        "max_file_size": 10485760,  # 10MB
        "backup_count": 5
    }
}

# ============ NIH INSTITUTE MAPPING ============
NIH_INSTITUTE_MAP = {
    'CA': 'NCI', 'MD': 'NIMHD', 'MH': 'NIMH', 'HL': 'NHLBI',
    'AG': 'NIA', 'DK': 'NIDDK', 'HD': 'NICHD', 'NR': 'NINR',
    'NS': 'NINDS', 'EY': 'NEI', 'DC': 'NIDCD', 'AA': 'NIAAA',
    'DA': 'NIDA', 'ES': 'NIEHS', 'GM': 'NIGMS', 'LM': 'NLM',
    'AI': 'NIAID', 'AR': 'NIAMS', 'DE': 'NIDCR', 'EB': 'NIBIB',
    'HG': 'NHGRI', 'OD': 'OD', 'AT': 'NCCIH', 'TW': 'FIC'
}

# ============ FQHC-SPECIFIC CONFIGURATION ============
FQHC_CONFIG = {
    "target_populations": ["low-income", "uninsured", "medicaid", "underserved", "minority", "rural"],
    "service_areas": [
        "primary care", "dental", "mental health", 
        "substance abuse", "preventive care", "chronic disease management",
        "maternal health", "pediatric care", "geriatric care"
    ],
    "research_focus_areas": [
        "health disparities",
        "access to care",
        "quality improvement",
        "chronic disease management",
        "community health workers",
        "telehealth",
        "social determinants of health",
        "implementation science",
        "patient-centered outcomes",
        "cost-effectiveness"
    ],
    "common_interventions": [
        "community health worker programs",
        "telemedicine implementation",
        "group medical visits",
        "medication therapy management",
        "behavioral health integration",
        "social needs screening",
        "care coordination",
        "health literacy programs",
        "culturally tailored interventions"
    ]
}

# ============ CUSTOM CONFIGURATIONS ============
CUSTOM_CONFIG = {
    "project_name": "FQHC_Grant_RAG",
    "test_mode": "development",  # development, staging, production
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    
    # PubMed model specific settings
    "pubmed_model": {
        "name": "pritamdeka/S-PubMedBert-MS-MARCO",
        "description": "PubMedBERT fine-tuned on MS-MARCO for biomedical retrieval",
        "max_seq_length": 512,
        "pooling_method": "mean",
        "normalize_embeddings": True
    },
    
    # API rate limiting
    "rate_limits": {
        "nih_api": {
            "requests_per_minute": 60,
            "batch_size": 50,
            "delay_between_requests": 1.0
        },
        "embedding_generation": {
            "batch_size": 32,
            "max_concurrent": 4
        }
    },
    
    # Memory management
    "memory": {
        "max_abstracts_in_memory": 1000,
        "chunk_cache_size": 100,
        "embedding_cache_size": 500
    },
    
    # Performance settings
    "performance": {
        "use_gpu_if_available": True,
        "parallel_processing": True,
        "num_workers": 4,
        "prefetch_factor": 2
    }
}

# ============ HELPER FUNCTIONS ============
def get_config_value(*keys, default=None):
    """
    Safely get nested configuration values from RAG_CONFIG
    
    Example:
        get_config_value("phase1", "test_sample_size", default=50)
    """
    current = RAG_CONFIG
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
    return current if current is not None else default

def list_available_models():
    """List all available embedding models in the configuration"""
    models = []
    
    # Phase 1 models
    phase1_models = RAG_CONFIG.get("phase1", {}).get("test_embedding_models", [])
    models.extend([m["name"] for m in phase1_models])
    
    # Phase 2 models
    phase2_model = RAG_CONFIG.get("phase2", {}).get("embedding_model")
    if phase2_model:
        models.append(phase2_model)
    
    phase2_alternatives = RAG_CONFIG.get("phase2", {}).get("alternative_models", [])
    models.extend(phase2_alternatives)
    
    # Phase 3 models
    phase3_model = RAG_CONFIG.get("phase3", {}).get("embedding_model")
    if phase3_model:
        models.append(phase3_model)
    
    phase3_alternatives = RAG_CONFIG.get("phase3", {}).get("alternative_models", [])
    models.extend([m["name"] for m in phase3_alternatives])
    
    return list(set(models))  # Remove duplicates

# ============ VALIDATION ============
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required fields
    required_phase1 = ["test_sample_size", "test_chunk_sizes", "test_embedding_models"]
    for field in required_phase1:
        if field not in RAG_CONFIG.get("phase1", {}):
            errors.append(f"Missing phase1.{field}")
    
    # Check model names
    models = list_available_models()
    if not models:
        errors.append("No embedding models configured")
    
    # Check paths
    paths = RAG_CONFIG.get("paths", {})
    for path_name, path_value in paths.items():
        if not isinstance(path_value, str):
            errors.append(f"paths.{path_name} should be a string")
    
    return errors

# Run validation on import
config_errors = validate_config()
if config_errors:
    print(f"⚠️  Configuration warnings: {config_errors}")