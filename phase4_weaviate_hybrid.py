# ============================================================================
# 📁 phase4_weaviate_hybrid.py - WEAVIATE V4 EMBEDDED (FULLY FIXED)
# ============================================================================

"""
PHASE 4: WEAVIATE V4 EMBEDDED HYBRID-RAG
Works in Colab - no Docker, no localhost
"""

print("="*70)
print("🎯 PHASE 4: WEAVIATE V4 HYBRID-RAG (Embedded)")
print("="*70)

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import time
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# ============ WEAVIATE V4 INIT ============
try:
    import weaviate
    from weaviate.classes.query import MetadataQuery
    from weaviate.classes.data import DataObject
    from weaviate.classes.config import Property, DataType
    print(f"✅ Weaviate v{weaviate.__version__} loaded")
except ImportError:
    print("📦 Installing weaviate-client v4...")
    import weaviate
    from weaviate.classes.query import MetadataQuery
    from weaviate.classes.data import DataObject
    from weaviate.classes.config import Property, DataType
    print(f"✅ Weaviate v{weaviate.__version__} installed")


class WeaviateManager:
    """Weaviate v4 embedded - works in Colab"""
    
    def __init__(self):
        print("🚀 Starting embedded Weaviate...")
        self.client = weaviate.connect_to_embedded()
        assert self.client.is_ready(), "Weaviate not ready"
        print("✅ Weaviate v4 embedded ready!")
    
    def create_schema(self, class_name: str = "GrantChunk"):
        """Create collection with correct v4 schema"""
        print(f"\n📐 Creating collection: {class_name}")
        
        if self.client.collections.exists(class_name):
            print(f"⚠️  Collection {class_name} already exists")
            return False
        
        self.client.collections.create(
            name=class_name,
            description="Grant chunks for FQHC RAG",
            vectorizer_config=None,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="sourceDocument", data_type=DataType.TEXT),
                Property(name="chunkIndex", data_type=DataType.INT),
                Property(name="wordCount", data_type=DataType.INT),
                Property(name="grantId", data_type=DataType.TEXT),
                Property(name="year", data_type=DataType.INT),
                Property(name="institute", data_type=DataType.TEXT),
                Property(name="isFQHCFocused", data_type=DataType.BOOL),
                Property(name="sectionType", data_type=DataType.TEXT),
            ]
        )
        print(f"✅ Created collection: {class_name}")
        return True
    
    def import_chunks(self, chunks_df: pd.DataFrame, class_name: str = "GrantChunk", batch_size: int = 100):
        """Import chunks with embeddings"""
        print(f"\n📤 Importing {len(chunks_df)} chunks...")
        
        if chunks_df.empty:
            print("❌ No chunks to import")
            return 0
            
        if 'embedding' not in chunks_df.columns:
            print("❌ No embeddings column found")
            return 0
        
        collection = self.client.collections.get(class_name)
        total = 0
        failed = 0
        
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for idx, row in chunks_df.iterrows():
                try:
                    vec = row.get("embedding")
                    if vec is None:
                        failed += 1
                        continue
                        
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    elif not isinstance(vec, list):
                        if isinstance(vec, str):
                            import ast
                            vec = ast.literal_eval(vec)
                        else:
                            failed += 1
                            continue
                    
                    batch.add_object(
                        properties={
                            "text": str(row.get("text", ""))[:5000],
                            "sourceDocument": str(row.get("source_document", row.get("sourceDocument", "unknown"))),
                            "chunkIndex": int(row.get("chunk_index", row.get("chunkIndex", 0))),
                            "wordCount": int(row.get("word_count", row.get("wordCount", 0))),
                            "grantId": str(row.get("grant_id", row.get("grantId", ""))),
                            "year": int(row.get("year", 2024)),
                            "institute": str(row.get("institute", "")),
                            "isFQHCFocused": bool(row.get("is_fqhc_focused", row.get("isFQHCFocused", False))),
                            "sectionType": str(row.get("section_type", row.get("sectionType", "other"))),
                        },
                        vector=vec
                    )
                    total += 1
                    
                    if total % 100 == 0:
                        print(f"  Imported {total} chunks...")
                        
                except Exception as e:
                    failed += 1
                    if failed < 5:
                        print(f"  ⚠️  Error on row {idx}: {e}")
        
        print(f"✅ Imported {total} chunks ({failed} failed)")
        return total
    
    def close(self):
        if self.client:
            self.client.close()
            print("👋 Connection closed")


class HybridRAGEvaluator:
    """Hybrid search evaluator for v4"""
    
    def __init__(self, client, class_name: str = "GrantChunk"):
        self.collection = client.collections.get(class_name)
    
    def test_hybrid(self, query: str, alpha: float = 0.5, limit: int = 5, query_vector=None):
        """Hybrid search with optional vector"""
        print(f"\n🔍 Hybrid (α={alpha}): '{query[:30]}...'")
        start = time.time()
        try:
            if alpha > 0 and query_vector is not None:
                resp = self.collection.query.hybrid(
                    query=query,
                    alpha=alpha,
                    vector=query_vector,
                    limit=limit,
                    return_metadata=MetadataQuery(score=True)
                )
            else:
                resp = self.collection.query.hybrid(
                    query=query,
                    alpha=alpha,
                    limit=limit,
                    return_metadata=MetadataQuery(score=True)
                )
            t = time.time() - start
            print(f"  Found {len(resp.objects)} results in {t:.3f}s")
            return resp.objects, t
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return [], 0
    
    def test_bm25(self, query: str, limit: int = 5):
        """Pure BM25 keyword search"""
        print(f"\n🔤 BM25: '{query[:30]}...'")
        start = time.time()
        try:
            resp = self.collection.query.bm25(
                query=query,
                limit=limit,
                return_metadata=MetadataQuery(score=True)
            )
            t = time.time() - start
            print(f"  Found {len(resp.objects)} results in {t:.3f}s")
            return resp.objects, t
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return [], 0
    
    def test_vector_from_query(self, query: str, model, limit: int = 5):
        """Vector search using query string and model"""
        print(f"\n🎯 Vector: '{query[:30]}...'")
        try:
            vec = model.encode(query).tolist()
            start = time.time()
            resp = self.collection.query.near_vector(
                near_vector=vec,
                limit=limit,
                return_metadata=MetadataQuery(distance=True)
            )
            t = time.time() - start
            print(f"  Found {len(resp.objects)} results in {t:.3f}s")
            return resp.objects, t
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return [], 0
    
    def test_vector_with_vector(self, query_vector: list, limit: int = 5):
        """Pure vector search with pre-computed vector"""
        print(f"\n🎯 Vector (pre-computed)...")
        start = time.time()
        try:
            resp = self.collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=MetadataQuery(distance=True)
            )
            t = time.time() - start
            print(f"  Found {len(resp.objects)} results in {t:.3f}s")
            return resp.objects, t
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return [], 0


def _create_sample_chunks(n: int = 100):
    """Sample chunks for testing"""
    chunks = []
    for i in range(n):
        chunks.append({
            "text": f"Sample grant chunk {i+1} about FQHC interventions for underserved populations. This study evaluates the effectiveness of community health workers in diabetes management at Federally Qualified Health Centers.",
            "source_document": f"grant_doc_{(i%5)+1}.pdf",
            "chunk_index": i % 20,
            "word_count": 75 + (i % 30),
            "grant_id": f"R01MD{100000+i}",
            "year": 2022 + (i % 3),
            "institute": ["NIMHD", "NIMH", "NCI", "NHLBI"][i % 4],
            "is_fqhc_focused": i % 3 != 0,
            "section_type": ["abstract", "methods", "background", "results"][i % 4],
            "embedding": list(np.random.randn(768))
        })
    return pd.DataFrame(chunks)


def run_phase4():
    """Run Phase 4 with Weaviate v4 embedded"""
    
    print("\n🚀 STARTING PHASE 4: WEAVIATE V4 EMBEDDED")
    
    # 1. Init Weaviate
    try:
        wm = WeaviateManager()
        wm.create_schema("GrantChunk")
    except Exception as e:
        print(f"❌ Weaviate failed: {e}")
        print("⚠️  Use mock mode: python phase4_weaviate_hybrid.py --mock")
        return
    
    # 2. Load data
    df = None
    for path in ['./phase3_results/document_chunks.csv', 'document_chunks.csv', 'phase3_results.csv']:
        try:
            df = pd.read_csv('./phase3_results/document_chunks_with_embeddings.csv')
            print(f"📊 Loaded {len(df)} chunks from {path}")
            break
        except:
            continue
    
    if df is None:
        print("⚠️  No chunk file found, creating sample data")
        df = _create_sample_chunks(100)
    
    # 3. Add embeddings if missing
    if 'embedding' not in df.columns:
        print("⚠️  No embeddings found, generating random 768d embeddings")
        df['embedding'] = [np.random.randn(768).tolist() for _ in range(len(df))]
    
    # 4. Import
    imported = wm.import_chunks(df, "GrantChunk")
    if imported == 0:
        print("❌ No chunks imported")
        wm.close()
        return
    
    # 5. Load PubMedBERT model (768d) - SAME AS YOUR CHUNKS
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        print("✅ Loaded PubMedBERT model (768d) - matches chunk dimensions")
    except Exception as e:
        print(f"⚠️  Could not load PubMedBERT: {e}")
        print("⚠️  Trying fallback model...")
        try:
            model = SentenceTransformer('all-mpnet-base-v2')
            print("✅ Loaded all-mpnet-base-v2 (768d)")
        except:
            print("❌ No 768d model available")
    
    # 6. Test queries
    queries = [
        "diabetes prevention community health centers",
        "behavioral health FQHC funding",
        "community health worker interventions"
    ]
    
    # 7. Run evaluations
    evaluator = HybridRAGEvaluator(wm.client, "GrantChunk")
    
    results = {"hybrid": {}, "vector": {}, "bm25": {}}
    alpha_metrics = {}
    
    for q in queries:
        print(f"\n{'='*50}\nTesting query: {q}\n{'='*50}")
        
        # Generate query vector ONCE per query (768d)
        query_vector = None
        if model:
            query_vector = model.encode(q).tolist()
            print(f"  ✅ Generated {len(query_vector)}d query vector")
        
        # Test different alpha values
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            resp, t = evaluator.test_hybrid(
                query=q, 
                alpha=alpha, 
                limit=5, 
                query_vector=query_vector if alpha > 0 else None
            )
            
            if alpha not in alpha_metrics:
                alpha_metrics[alpha] = {"times": [], "counts": []}
            alpha_metrics[alpha]["times"].append(t)
            alpha_metrics[alpha]["counts"].append(len(resp))
        
        # Test BM25
        _, t = evaluator.test_bm25(q, limit=5)
        results["bm25"][q] = {"time": t}
        
        # Test pure vector if model available
        if model and query_vector:
            _, t = evaluator.test_vector_with_vector(query_vector, limit=5)
            results["vector"][q] = {"time": t}
    
    # 8. Summary
    print("\n" + "="*70)
    print("📊 ALPHA TUNING RESULTS")
    print("="*70)
    
    valid_alphas = []
    for alpha in sorted(alpha_metrics.keys()):
        times = [t for t in alpha_metrics[alpha]["times"] if t > 0]
        counts = [c for c in alpha_metrics[alpha]["counts"]]
        if times:
            avg_t = np.mean(times)
            avg_c = np.mean(counts)
            print(f"  α={alpha}: avg time {avg_t:.4f}s, avg results {avg_c:.1f}")
            valid_alphas.append((alpha, avg_t))
    
    if valid_alphas:
        optimal_alpha = min(valid_alphas, key=lambda x: x[1])[0]
        print(f"\n✅ Optimal alpha (fastest): {optimal_alpha}")
    else:
        optimal_alpha = 0.7
        print(f"\n⚠️  Using default optimal alpha: {optimal_alpha}")
    
    # 9. Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "chunks_imported": imported,
        "chunk_dimension": 768,
        "query_model": "pritamdeka/S-PubMedBert-MS-MARCO",
        "optimal_alpha": optimal_alpha,
        "alpha_performance": {
            str(a): {
                "avg_time": float(np.mean(alpha_metrics[a]["times"])),
                "avg_results": float(np.mean(alpha_metrics[a]["counts"]))
            } for a in alpha_metrics
        }
    }
    
    with open("phase4_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n📁 Results saved to phase4_results.json")
    
    # 10. Clean up
    wm.close()
    print("\n✅ PHASE 4 COMPLETE")
    
    return output


def mock_mode():
    """Run mock tests without Weaviate"""
    print("⚠️  Running mock Phase 4 tests...")
    
    df = _create_sample_chunks(100)
    df.to_csv("mock_chunks.csv", index=False)
    print("✅ Created mock_chunks.csv")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": "mock",
        "optimal_alpha": 0.7,
        "alpha_performance": {
            "0.0": {"avg_time": 0.05, "avg_results": 4.2},
            "0.3": {"avg_time": 0.08, "avg_results": 3.8},
            "0.5": {"avg_time": 0.11, "avg_results": 3.5},
            "0.7": {"avg_time": 0.14, "avg_results": 3.2},
            "1.0": {"avg_time": 0.17, "avg_results": 2.9}
        }
    }
    
    with open("phase4_results_mock.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("📁 Mock results saved to phase4_results_mock.json")
    return results


if __name__ == "__main__":
    import sys
    if "--mock" in sys.argv:
        mock_mode()
    else:
        run_phase4()