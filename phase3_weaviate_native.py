# ============================================================================
# 📁 phase3_weaviate_native.py - PHASE 3 WITH WEAVIATE BACKEND
# ============================================================================

"""
PHASE 3: WEAVIATE-NATIVE RAG SYSTEM
Replaces FAISS with Weaviate as primary vector store
Supports BOTH public corpus AND user uploads
"""

print("="*70)
print("🎯 PHASE 3: WEAVIATE-NATIVE RAG SYSTEM")
print("="*70)

import sys
sys.path.append('.')

from config import RAG_CONFIG
import pandas as pd
import numpy as np
import time
import json
import os
import re
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

# ============ WEAVIATE IMPORTS ============
try:
    import weaviate
    from weaviate.classes.query import MetadataQuery
    from weaviate.classes.config import Property, DataType
    from weaviate.classes.data import DataObject
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    print("⚠️  Weaviate not available. Run: !pip install weaviate-client")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️  Sentence transformers not available.")


class WeaviateRAG:
    """
    WEAVIATE-NATIVE RAG SYSTEM
    - Uses Weaviate as primary vector store (no FAISS)
    - Supports public corpus + user uploads
    - Loads pre-computed embeddings for public corpus
    """
    
    def __init__(self, 
                 model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
                 use_existing_embeddings: bool = True):
        
        print(f"🚀 Initializing Weaviate-Native RAG...")
        print(f"   Model: {model_name}")
        print(f"   Embeddings: {'LOAD pre-computed' if use_existing_embeddings else 'COMPUTE new'}")
        
        # Load embedding model (for queries and new documents)
        if EMBEDDING_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            print(f"✅ Loaded embedding model: {model_name}")
        else:
            self.model = None
            raise ImportError("Sentence transformers not available")
        
        # Connect to Weaviate
        self._connect_weaviate()
        
        # Initialize collections
        self.public_collection = "GrantChunk"
        self.user_collection_prefix = "UserDocs_"
        
        # Load public corpus if embeddings exist
        if use_existing_embeddings:
            self._load_public_corpus()
    
    def _connect_weaviate(self):
        """Connect to Weaviate embedded"""
        print("\n🔗 Connecting to Weaviate...")
        
        try:
            self.client = weaviate.connect_to_embedded()
            assert self.client.is_ready(), "Weaviate not ready"
            print("✅ Connected to Weaviate embedded")
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            raise
    
    def _load_public_corpus(self):
        """Load pre-computed embeddings into Weaviate"""
        print("\n📚 Loading public corpus (4,316 chunks)...")
        
        # Check if collection already exists and has data
        if self.client.collections.exists(self.public_collection):
            collection = self.client.collections.get(self.public_collection)
            count = collection.aggregate.overall(total_count=True).total_count
            if count > 0:
                print(f"✅ Public corpus already loaded: {count} chunks")
                return
        
        # Load chunks with embeddings
        try:
            chunks_df = pd.read_csv('./phase3_results/document_chunks_with_embeddings.csv')
            print(f"📊 Loaded {len(chunks_df)} chunks with embeddings")
            
            # Convert string embeddings to lists
            import ast
            chunks_df['embedding'] = chunks_df['embedding'].apply(ast.literal_eval)
            
            # Create collection if not exists
            if not self.client.collections.exists(self.public_collection):
                self._create_grant_collection(self.public_collection)
            
            collection = self.client.collections.get(self.public_collection)
            
            # Batch import
            print(f"📤 Importing {len(chunks_df)} chunks to Weaviate...")
            total = 0
            
            with collection.batch.fixed_size(batch_size=100) as batch:
                for idx, row in chunks_df.iterrows():
                    batch.add_object(
                        properties={
                            "text": str(row.get("text", ""))[:5000],
                            "sourceDocument": str(row.get("source_document", row.get("sourceDocument", "NIH"))),
                            "chunkIndex": int(row.get("chunk_index", row.get("chunkIndex", 0))),
                            "wordCount": int(row.get("word_count", row.get("wordCount", 0))),
                            "grantId": str(row.get("grant_id", row.get("grantId", ""))),
                            "year": int(row.get("year", 2024)),
                            "institute": str(row.get("institute", "")),
                            "isFQHCFocused": bool(row.get("is_fqhc_focused", row.get("isFQHCFocused", False))),
                            "sectionType": str(row.get("section_type", row.get("sectionType", "other"))),
                            "document_title": str(row.get("document_title", "")),
                            "data_source": str(row.get("data_source", "nih")),
                            "corpus_type": "public"
                        },
                        vector=row['embedding']
                    )
                    total += 1
                    
                    if total % 500 == 0:
                        print(f"  Imported {total} chunks...")
            
            print(f"✅ Loaded {total} chunks to public corpus")
            
        except Exception as e:
            print(f"❌ Failed to load public corpus: {e}")
            print("   Run Phase 3 with FAISS first to generate embeddings")
            raise
    
    def _create_grant_collection(self, collection_name: str):
        """Create Weaviate collection for grant chunks"""
        print(f"\n📐 Creating collection: {collection_name}")
        
        self.client.collections.create(
            name=collection_name,
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
                Property(name="document_title", data_type=DataType.TEXT),
                Property(name="data_source", data_type=DataType.TEXT),
                Property(name="corpus_type", data_type=DataType.TEXT),
                Property(name="user_id", data_type=DataType.TEXT),  # For private docs
            ]
        )
        print(f"✅ Created collection: {collection_name}")
    
    # ============ PUBLIC CORPUS SEARCH ============
    
    def search_public(self, query: str, top_k: int = 10, fqhc_boost: bool = True) -> List[Dict]:
        """
        Search the public NIH grant corpus (2,046 grants, 4,316 chunks)
        """
        print(f"\n🔍 Searching public corpus: '{query[:50]}...'")
        
        # Encode query
        query_vector = self.model.encode(query).tolist()
        
        # Search Weaviate
        collection = self.client.collections.get(self.public_collection)
        
        start_time = time.time()
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k * 2,  # Get more for FQHC boosting
            return_metadata=MetadataQuery(distance=True),
            return_properties=["text", "document_title", "grantId", "year", 
                             "institute", "isFQHCFocused", "fqhc_score", 
                             "sectionType", "sourceDocument"]
        )
        
        search_time = time.time() - start_time
        
        # Process results
        results = []
        for obj in response.objects:
            similarity = 1 - obj.metadata.distance if obj.metadata else 0
            
            # FQHC boost
            boosted_similarity = similarity
            if fqhc_boost and obj.properties.get("isFQHCFocused", False):
                fqhc_score = obj.properties.get("fqhc_score", 0.5)
                boost_factor = 1.0 + (fqhc_score * 0.5)
                boosted_similarity = similarity * boost_factor
            
            results.append({
                'rank': len(results) + 1,
                'text': obj.properties.get("text", "")[:300],
                'document_title': obj.properties.get("document_title", "Untitled"),
                'grant_id': obj.properties.get("grantId", ""),
                'year': obj.properties.get("year", "Unknown"),
                'institute': obj.properties.get("institute", "Unknown"),
                'is_fqhc_focused': obj.properties.get("isFQHCFocused", False),
                'fqhc_score': obj.properties.get("fqhc_score", 0.0),
                'section_type': obj.properties.get("sectionType", "other"),
                'similarity': round(similarity, 3),
                'boosted_similarity': round(boosted_similarity, 3),
                'search_time': round(search_time, 3),
                'source': 'public_corpus'
            })
        
        # Sort by boosted similarity
        results.sort(key=lambda x: x['boosted_similarity'], reverse=True)
        
        print(f"📊 Found {len(results[:top_k])} results in {search_time:.3f}s")
        return results[:top_k]
    
    # ============ USER DOCUMENT UPLOAD ============
    
    def create_user_collection(self, user_id: str) -> str:
        """Create a private collection for user documents"""
        collection_name = f"{self.user_collection_prefix}{user_id}"
        
        if not self.client.collections.exists(collection_name):
            self._create_grant_collection(collection_name)
            print(f"✅ Created private collection for user: {user_id}")
        
        return collection_name
    
    def upload_user_documents(self, 
                             user_id: str,
                             documents: List[Dict],
                             chunk_size: int = 250,
                             overlap: int = 50) -> int:
        """
        Upload and index user's private documents
        
        Args:
            user_id: Unique user identifier
            documents: List of dicts with 'text' and metadata
            chunk_size: Size of chunks in words
            overlap: Overlap between chunks
        
        Returns:
            Number of chunks created
        """
        print(f"\n📤 Uploading documents for user: {user_id}")
        
        # Create or get user collection
        collection_name = self.create_user_collection(user_id)
        collection = self.client.collections.get(collection_name)
        
        # Process each document
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get('text', '')
            doc_id = doc.get('doc_id', f"user_doc_{doc_idx}")
            title = doc.get('title', 'Untitled')
            
            # Simple chunking
            chunks = self._chunk_text(text, chunk_size, overlap)
            
            # Generate embeddings for chunks
            embeddings = self.model.encode(chunks)
            
            # Import to Weaviate
            with collection.batch.fixed_size(batch_size=50) as batch:
                for chunk_idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                    batch.add_object(
                        properties={
                            "text": chunk_text[:5000],
                            "sourceDocument": doc_id,
                            "document_title": title,
                            "chunkIndex": chunk_idx,
                            "wordCount": len(chunk_text.split()),
                            "user_id": user_id,
                            "corpus_type": "private",
                            "year": doc.get('year', datetime.now().year),
                            "grantId": doc.get('grant_id', f"user_{doc_id}"),
                        },
                        vector=embedding.tolist()
                    )
                    all_chunks.append(chunk_text)
            
            print(f"  Processed document {doc_idx + 1}/{len(documents)}: {len(chunks)} chunks")
        
        print(f"✅ Uploaded {len(all_chunks)} chunks for user {user_id}")
        return len(all_chunks)
    
    def _chunk_text(self, text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        """Simple text chunking"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    # ============ HYBRID SEARCH (Public + Private) ============
    
    def search_all(self, 
                  query: str,
                  user_id: Optional[str] = None,
                  top_k_public: int = 5,
                  top_k_private: int = 5,
                  fqhc_boost: bool = True) -> Dict:
        """
        Search BOTH public corpus AND user's private documents
        
        Returns:
            Dict with 'public' and 'private' results
        """
        results = {
            'public': [],
            'private': [],
            'query': query,
            'timestamp': datetime.now().isoformat()
        }
        
        # Search public corpus
        results['public'] = self.search_public(query, top_k=top_k_public, fqhc_boost=fqhc_boost)
        
        # Search user's private corpus if user_id provided
        if user_id:
            collection_name = f"{self.user_collection_prefix}{user_id}"
            
            if self.client.collections.exists(collection_name):
                collection = self.client.collections.get(collection_name)
                query_vector = self.model.encode(query).tolist()
                
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=top_k_private,
                    return_metadata=MetadataQuery(distance=True),
                    return_properties=["text", "document_title", "sourceDocument"]
                )
                
                for obj in response.objects:
                    similarity = 1 - obj.metadata.distance if obj.metadata else 0
                    results['private'].append({
                        'text': obj.properties.get("text", "")[:300],
                        'document_title': obj.properties.get("document_title", "Untitled"),
                        'source': obj.properties.get("sourceDocument", ""),
                        'similarity': round(similarity, 3),
                        'search_time': 0
                    })
        
        return results
    
    # ============ DEMO INTERFACE ============
    
    def interactive_demo(self):
        """Interactive demo with public corpus only"""
        print("\n" + "="*70)
        print("💬 WEAVIATE-NATIVE RAG DEMO")
        print("="*70)
        print(f"📚 Public corpus: 4,316 chunks from 2,046 NIH grants")
        print("📤 Type 'upload' to test document upload")
        print("Type 'quit' to exit")
        print("-" * 70)
        
        while True:
            query = input("\n🔍 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query.lower() == 'upload':
                self._demo_upload()
                continue
            
            if not query:
                continue
            
            results = self.search_public(query, top_k=5)
            
            print(f"\n📊 Top {len(results)} results:")
            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r['document_title']}")
                print(f"   📊 Similarity: {r['similarity']:.3f}")
                print(f"   🏷️  Section: {r['section_type']}")
                print(f"   🎯 FQHC: {r['is_fqhc_focused']} (score: {r['fqhc_score']:.2f})")
                print(f"   📝 {r['text'][:200]}...")
    
    def _demo_upload(self):
        """Demo user document upload"""
        print("\n" + "="*50)
        print("📤 TEST USER DOCUMENT UPLOAD")
        print("="*50)
        
        # Create a test user
        user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        
        # Create a test document
        test_doc = {
            'doc_id': 'demo_proposal_001',
            'title': 'Community Health Worker Diabetes Program',
            'text': """
            Our proposed program will train community health workers to provide 
            diabetes self-management education to Latino patients at our FQHC. 
            The intervention includes 6 monthly home visits focusing on medication 
            adherence, healthy eating, and glucose monitoring. We will measure 
            HbA1c reduction at 6 and 12 months. This builds on our successful 
            hypertension program and addresses the high diabetes burden in our 
            underserved population.
            """
        }
        
        print(f"👤 User ID: {user_id}")
        print(f"📄 Test document: {test_doc['title']}")
        
        # Upload
        chunk_count = self.upload_user_documents(user_id, [test_doc])
        print(f"✅ Created {chunk_count} chunks")
        
        # Test search
        print("\n🔍 Testing search on uploaded document...")
        results = self.search_all("community health worker diabetes", 
                                 user_id=user_id,
                                 top_k_public=2, 
                                 top_k_private=2)
        
        print(f"\n📊 Results:")
        print(f"  Public corpus: {len(results['public'])} results")
        print(f"  Private docs: {len(results['private'])} results")
        
        for r in results['private']:
            print(f"\n  🔒 PRIVATE: {r['document_title']}")
            print(f"     {r['text'][:150]}...")


# ============ MAIN EXECUTION ============

def run_phase3_weaviate():
    """Run Phase 3 with Weaviate backend"""
    
    print("\n" + "="*70)
    print("🚀 PHASE 3: WEAVIATE-NATIVE RAG")
    print("="*70)
    
    # Check dependencies
    if not WEAVIATE_AVAILABLE:
        print("❌ Weaviate not available. Installing...")
        !pip install weaviate-client
        import weaviate
        global WEAVIATE_AVAILABLE
        WEAVIATE_AVAILABLE = True
    
    # Initialize RAG system
    try:
        rag = WeaviateRAG(use_existing_embeddings=True)
        print("\n✅ Weaviate-Native RAG initialized successfully!")
        
        # Print stats
        collection = rag.client.collections.get(rag.public_collection)
        count = collection.aggregate.overall(total_count=True).total_count
        print(f"\n📊 Public corpus: {count} chunks ready")
        
        # Run demo
        rag.interactive_demo()
        
        return rag
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\n📌 First time setup:")
        print("   1. Run FAISS-based Phase 3 to generate embeddings")
        print("   2. Ensure ./phase3_results/document_chunks_with_embeddings.csv exists")
        return None


if __name__ == "__main__":
    rag = run_phase3_weaviate()