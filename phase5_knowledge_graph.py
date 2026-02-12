# ============================================================================
# 📁 phase5_knowledge_graph.py - WEEK 9: KNOWLEDGE GRAPH INTEGRATION (UPDATED)
# ============================================================================

"""
PHASE 5: KNOWLEDGE GRAPH-AUGMENTED RAG
Integration with existing Weaviate vector database from Phase 4
"""

print("="*70)
print("🎯 PHASE 5: KNOWLEDGE GRAPH INTEGRATION (Week 9)")
print("="*70)

import sys
sys.path.append('.')

from config import RAG_CONFIG
from utils import logger, DataProcessor
import pandas as pd
import numpy as np
import time
import json
import networkx as nx
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict

# EDIT HERE: Import knowledge graph and Weaviate libraries
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    from community import community_louvain
    import weaviate
    from weaviate.classes.query import MetadataQuery
    from weaviate.classes.config import Property, DataType
    KG_AVAILABLE = True
    WEAVIATE_AVAILABLE = True
    print("✅ Knowledge graph and Weaviate libraries loaded")
except ImportError as e:
    KG_AVAILABLE = False
    WEAVIATE_AVAILABLE = False
    print(f"⚠️  Missing library: {e}")
    print("   Run: !pip install networkx python-louvain weaviate-client matplotlib")


class WeaviateKnowledgeGraphBridge:
    """
    BRIDGE BETWEEN WEAVIATE VECTOR DB AND NETWORKX KNOWLEDGE GRAPH
    """
    
    def __init__(self):
        """Connect to existing Weaviate instance from Phase 4"""
        print("\n🔗 Connecting to Weaviate embedded...")
        try:
            self.client = weaviate.connect_to_embedded()
            self.is_connected = self.client.is_ready()
            print(f"✅ Connected to Weaviate: {self.is_connected}")
            
            # Get existing collections
            self.grant_collection = self.client.collections.get("GrantChunk")
            print("✅ Retrieved GrantChunk collection")
            
        except Exception as e:
            print(f"❌ Failed to connect to Weaviate: {e}")
            self.client = None
            self.is_connected = False
    
    def extract_grants_from_chunks(self) -> pd.DataFrame:
        """
        Extract grant-level data from your 458 chunks
        """
        print("\n📊 Extracting grant data from Weaviate chunks...")
        
        grants = []
        grant_ids_seen = set()
        
        # Fetch all chunks
        response = self.grant_collection.query.fetch_objects(limit=500)
        
        for obj in response.objects:
            grant_id = obj.properties.get("grantId", "")
            
            # Skip if no grant ID or already processed
            if not grant_id or grant_id in grant_ids_seen:
                continue
                
            grant_ids_seen.add(grant_id)
            
            # Extract grant-level data
            grant = {
                "grant_id": grant_id,
                "title": f"Grant {grant_id}",  # We don't store titles separately
                "abstract": obj.properties.get("text", "")[:500],
                "year": obj.properties.get("year", 2024),
                "institute": obj.properties.get("institute", "Unknown"),
                "sourceDocument": obj.properties.get("sourceDocument", ""),
                "is_fqhc_focused": obj.properties.get("isFQHCFocused", False),
                "sectionType": obj.properties.get("sectionType", ""),
                "chunk_count": 1
            }
            grants.append(grant)
        
        # Count chunks per grant
        for obj in response.objects:
            grant_id = obj.properties.get("grantId", "")
            for grant in grants:
                if grant["grant_id"] == grant_id:
                    grant["chunk_count"] += 1
                    break
        
        df = pd.DataFrame(grants)
        print(f"✅ Extracted {len(df)} unique grants from {len(response.objects)} chunks")
        return df
    
    def get_chunks_by_grant(self, grant_id: str) -> List[Dict]:
        """Get all chunks belonging to a specific grant"""
        response = self.grant_collection.query.bm25(
            query=grant_id,
            properties=["grantId"],
            limit=20
        )
        
        chunks = []
        for obj in response.objects:
            chunks.append({
                "text": obj.properties.get("text", ""),
                "sectionType": obj.properties.get("sectionType", ""),
                "wordCount": obj.properties.get("wordCount", 0),
                "uuid": str(obj.uuid)
            })
        
        return chunks
    
    def close(self):
        """Close Weaviate connection"""
        if self.client:
            self.client.close()
            print("👋 Weaviate connection closed")


class KnowledgeGraphBuilder:
    """
    BUILD KNOWLEDGE GRAPH FROM GRANT DATA WITH WEAVIATE INTEGRATION
    """
    
    def __init__(self, weaviate_bridge: WeaviateKnowledgeGraphBridge = None):
        self.graph = nx.Graph()
        self.weaviate = weaviate_bridge
        self.node_types = {}
        self.edge_types = {}
        
    def build_from_weaviate(self):
        """
        BUILD GRAPH DIRECTLY FROM WEAVIATE DATA
        """
        print("\n🏗️  Building knowledge graph from Weaviate...")
        
        if not self.weaviate or not self.weaviate.is_connected:
            print("❌ Weaviate not connected")
            return self.graph
        
        # 1. Extract grants from chunks
        grants_df = self.weaviate.extract_grants_from_chunks()
        
        # 2. Add grant nodes
        for _, grant in grants_df.iterrows():
            self.graph.add_node(
                grant['grant_id'],
                type='grant',
                year=grant['year'],
                institute=grant['institute'],
                is_fqhc_focused=grant['is_fqhc_focused'],
                chunk_count=grant['chunk_count']
            )
        
        # 3. Add relationships
        self._add_institute_relationships(grants_df)
        self._add_temporal_relationships(grants_df)
        self._add_fqhc_relationships(grants_df)
        self._add_semantic_relationships(grants_df)
        
        print(f"✅ Graph built with {self.graph.number_of_nodes()} nodes and "
              f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _add_institute_relationships(self, grants_df: pd.DataFrame):
        """Connect grants to NIH institutes"""
        print("  Adding institute relationships...")
        
        institute_to_grants = defaultdict(list)
        for _, grant in grants_df.iterrows():
            institute = grant['institute']
            if institute and institute != "Unknown":
                institute_to_grants[institute].append(grant['grant_id'])
        
        # Add institute nodes and connect grants
        for institute, grant_ids in institute_to_grants.items():
            node_id = f"INST_{institute}"
            self.graph.add_node(node_id, type='institute', name=institute)
            
            for grant_id in grant_ids:
                self.graph.add_edge(
                    grant_id, node_id,
                    type='funded_by',
                    weight=1.0
                )
    
    def _add_temporal_relationships(self, grants_df: pd.DataFrame):
        """Connect grants to years and create year-year similarity"""
        print("  Adding temporal relationships...")
        
        year_to_grants = defaultdict(list)
        for _, grant in grants_df.iterrows():
            year = grant['year']
            year_to_grants[year].append(grant['grant_id'])
        
        # Add year nodes and connect grants
        years = sorted(year_to_grants.keys())
        for year, grant_ids in year_to_grants.items():
            node_id = f"YEAR_{year}"
            self.graph.add_node(node_id, type='year', year=year)
            
            for grant_id in grant_ids:
                self.graph.add_edge(
                    grant_id, node_id,
                    type='published_in',
                    weight=0.8
                )
        
        # Connect consecutive years
        for i in range(len(years) - 1):
            year1 = f"YEAR_{years[i]}"
            year2 = f"YEAR_{years[i+1]}"
            self.graph.add_edge(
                year1, year2,
                type='consecutive_year',
                weight=0.5
            )
    
    def _add_fqhc_relationships(self, grants_df: pd.DataFrame):
        """Connect grants based on FQHC focus"""
        print("  Adding FQHC focus relationships...")
        
        fqhc_grants = grants_df[grants_df['is_fqhc_focused']]['grant_id'].tolist()
        non_fqhc_grants = grants_df[~grants_df['is_fqhc_focused']]['grant_id'].tolist()
        
        # Add FQHC focus node
        self.graph.add_node("FQHC_FOCUSED", type='fqhc', description="FQHC-relevant grants")
        
        # Connect all FQHC grants to this node
        for grant_id in fqhc_grants:
            self.graph.add_edge(
                grant_id, "FQHC_FOCUSED",
                type='is_fqhc_relevant',
                weight=0.9
            )
        
        # Connect FQHC grants to each other
        for i in range(len(fqhc_grants)):
            for j in range(i + 1, len(fqhc_grants)):
                self.graph.add_edge(
                    fqhc_grants[i], fqhc_grants[j],
                    type='both_fqhc_focused',
                    weight=0.7
                )
    
    def _add_semantic_relationships(self, grants_df: pd.DataFrame):
        """Connect grants that share similar characteristics"""
        print("  Adding semantic relationships...")
        
        # Group by institute and year
        for (institute, year), group in grants_df.groupby(['institute', 'year']):
            grant_ids = group['grant_id'].tolist()
            
            if len(grant_ids) > 1:
                for i in range(len(grant_ids)):
                    for j in range(i + 1, len(grant_ids)):
                        self.graph.add_edge(
                            grant_ids[i], grant_ids[j],
                            type='same_institute_year',
                            weight=0.6
                        )
    
    def analyze_graph(self) -> Dict:
        """
        ANALYZE GRAPH STRUCTURE AND PROPERTIES
        """
        print("\n📊 Analyzing knowledge graph...")
        
        if self.graph.number_of_nodes() == 0:
            return {}
        
        analysis = {
            "basic_stats": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "connected_components": nx.number_connected_components(self.graph),
            },
            "node_types": defaultdict(int),
            "edge_types": defaultdict(int),
            "centrality": {},
            "community_structure": {}
        }
        
        # Count node types
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            analysis["node_types"][node_type] += 1
        
        # Count edge types
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            analysis["edge_types"][edge_type] += 1
        
        # Calculate centrality for grant nodes
        grant_nodes = [n for n, d in self.graph.nodes(data=True) 
                      if d.get('type') == 'grant']
        
        if grant_nodes:
            subgraph = self.graph.subgraph(grant_nodes)
            
            degree_centrality = nx.degree_centrality(subgraph)
            analysis["centrality"]["degree_top_5"] = [
                (n, round(v, 4)) for n, v in sorted(
                    degree_centrality.items(), key=lambda x: x[1], reverse=True
                )[:5]
            ]
        
        # Detect communities
        if len(grant_nodes) > 5:
            try:
                # Convert to undirected for community detection
                graph_undirected = self.graph.to_undirected()
                partition = community_louvain.best_partition(graph_undirected)
                analysis["community_structure"]["num_communities"] = len(set(partition.values()))
                analysis["community_structure"]["modularity"] = round(
                    community_louvain.modularity(partition, graph_undirected), 4
                )
            except Exception as e:
                print(f"  ⚠️  Community detection skipped: {e}")
        
        return analysis
    
    def find_related_grants(self, grant_id: str, 
                           max_depth: int = 2,
                           min_strength: float = 0.3,
                           max_results: int = 10) -> List[Dict]:
        """
        FIND RELATED GRANTS USING GRAPH TRAVERSAL
        """
        if grant_id not in self.graph:
            return []
        
        related = []
        visited = set()
        
        # BFS traversal
        queue = [(grant_id, 0, [grant_id], 1.0)]
        
        while queue and len(related) < max_results:
            current, depth, path, strength = queue.pop(0)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            # Add grant nodes (not the starting grant)
            if current != grant_id and self.graph.nodes[current].get('type') == 'grant':
                related.append({
                    "grant_id": current,
                    "depth": depth,
                    "relationship_strength": round(strength, 3),
                    "path": " → ".join(path),
                    "node_data": dict(self.graph.nodes[current])
                })
            
            # Explore neighbors
            if depth < max_depth:
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        edge_data = self.graph.get_edge_data(current, neighbor, {})
                        edge_weight = edge_data.get('weight', 0.5)
                        queue.append(
                            (neighbor, depth + 1, path + [neighbor], strength * edge_weight)
                        )
        
        # Sort by relationship strength
        related.sort(key=lambda x: x["relationship_strength"], reverse=True)
        
        return related[:max_results]
    
    def get_subgraph_by_institute(self, institute: str) -> nx.Graph:
        """Get subgraph of all grants from a specific institute"""
        institute_node = f"INST_{institute}"
        
        if institute_node not in self.graph:
            return nx.Graph()
        
        # Get all nodes within 2 hops of the institute node
        nodes = {institute_node}
        for neighbor in self.graph.neighbors(institute_node):
            nodes.add(neighbor)
            for second_neighbor in self.graph.neighbors(neighbor):
                nodes.add(second_neighbor)
        
        return self.graph.subgraph(nodes)
    
    def save_graph(self, filename: str = "knowledge_graph.gml"):
        """Save graph to file"""
        try:
            nx.write_gml(self.graph, filename)
            print(f"💾 Saved graph to {filename}")
        except Exception as e:
            print(f"❌ Failed to save graph: {e}")


class GraphAugmentedRAG:
    """
    KNOWLEDGE GRAPH-AUGMENTED RAG WITH WEAVIATE BACKEND
    """
    
    def __init__(self, weaviate_bridge: WeaviateKnowledgeGraphBridge, 
                 knowledge_graph: nx.Graph):
        self.weaviate = weaviate_bridge
        self.graph = knowledge_graph
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        PURE VECTOR SEARCH (α=1.0) FROM PHASE 4
        """
        from sentence_transformers import SentenceTransformer
        
        # Load PubMedBERT model
        model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        query_vector = model.encode(query).tolist()
        
        # Vector search
        response = self.weaviate.grant_collection.query.near_vector(
            near_vector=query_vector,
            limit=k,
            return_metadata=MetadataQuery(distance=True),
            return_properties=["text", "sourceDocument", "year", "institute", 
                             "isFQHCFocused", "sectionType", "grantId"]
        )
        
        results = []
        for obj in response.objects:
            results.append({
                "grant_id": obj.properties.get("grantId", ""),
                "text": obj.properties.get("text", "")[:200] + "...",
                "source": obj.properties.get("sourceDocument", "Unknown"),
                "year": obj.properties.get("year", "Unknown"),
                "institute": obj.properties.get("institute", "Unknown"),
                "fqhc_focused": obj.properties.get("isFQHCFocused", False),
                "section_type": obj.properties.get("sectionType", "other"),
                "similarity_score": round(1 - obj.metadata.distance, 3) if obj.metadata else 0,
                "retrieval_type": "vector"
            })
        
        return results
    
    def retrieve_with_graph_expansion(self, query: str, 
                                     vector_k: int = 3,
                                     graph_k: int = 2) -> List[Dict]:
        """
        VECTOR SEARCH + GRAPH EXPANSION
        """
        # Step 1: Get initial vector results
        vector_results = self.retrieve(query, k=vector_k)
        results = vector_results.copy()
        
        # Step 2: Expand each result via graph
        for result in vector_results[:2]:  # Expand top 2
            grant_id = result.get("grant_id")
            
            if grant_id and grant_id in self.graph:
                related = self._kg_retrieve(grant_id, k=graph_k)
                
                for rel in related:
                    if not any(r.get("grant_id") == rel["grant_id"] for r in results):
                        results.append({
                            "grant_id": rel["grant_id"],
                            "text": f"Related grant via {rel['path']}",
                            "source": "Knowledge Graph",
                            "year": rel["node_data"].get("year", "Unknown"),
                            "institute": rel["node_data"].get("institute", "Unknown"),
                            "fqhc_focused": rel["node_data"].get("is_fqhc_focused", False),
                            "similarity_score": rel["relationship_strength"],
                            "retrieval_type": "graph_expansion",
                            "original_grant": grant_id,
                            "relationship_path": rel["path"]
                        })
        
        return results[:vector_k + graph_k]
    
    def _kg_retrieve(self, grant_id: str, k: int = 2) -> List[Dict]:
        """Internal graph retrieval"""
        if grant_id not in self.graph:
            return []
        
        related = []
        for neighbor in self.graph.neighbors(grant_id):
            if self.graph.nodes[neighbor].get('type') == 'grant':
                edge_data = self.graph.get_edge_data(grant_id, neighbor, {})
                strength = edge_data.get('weight', 0.5)
                
                related.append({
                    "grant_id": neighbor,
                    "relationship_strength": strength,
                    "path": f"{grant_id} → {neighbor}",
                    "node_data": dict(self.graph.nodes[neighbor])
                })
        
        related.sort(key=lambda x: x["relationship_strength"], reverse=True)
        return related[:k]
    
    def explain_result(self, result: Dict) -> str:
        """Generate human-readable explanation"""
        if result.get("retrieval_type") == "vector":
            return (f"📄 Found via semantic search (similarity: {result['similarity_score']:.3f}) "
                   f"from {result['institute']} ({result['year']})")
        
        elif result.get("retrieval_type") == "graph_expansion":
            return (f"🔄 Discovered via graph relationship: {result['relationship_path']} "
                   f"(strength: {result['similarity_score']:.3f})")
        
        return "Unknown retrieval method"


def visualize_knowledge_graph(knowledge_graph: nx.Graph, 
                             analysis_results: Dict,
                             save: bool = True):
    """Visualize knowledge graph analysis"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 5: Weaviate + Knowledge Graph Integration', fontsize=16)
    
    # 1. Graph size overview
    ax = axes[0, 0]
    stats = analysis_results.get('basic_stats', {})
    ax.bar(['Nodes', 'Edges'], [stats.get('nodes', 0), stats.get('edges', 0)], 
           color=['steelblue', 'coral'])
    ax.set_title('Graph Size')
    ax.set_ylabel('Count')
    
    # 2. Node types
    ax = axes[0, 1]
    node_types = analysis_results.get('node_types', {})
    if node_types:
        types = list(node_types.keys())
        counts = list(node_types.values())
        ax.pie(counts, labels=types, autopct='%1.1f%%')
        ax.set_title('Node Types')
    
    # 3. Edge types
    ax = axes[0, 2]
    edge_types = analysis_results.get('edge_types', {})
    if edge_types:
        types = list(edge_types.keys())[:5]  # Top 5
        counts = list(edge_types.values())[:5]
        ax.barh(range(len(types)), counts, color='lightgreen')
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels(types)
        ax.set_title('Top 5 Edge Types')
    
    # 4. Centrality
    ax = axes[1, 0]
    centrality = analysis_results.get('centrality', {})
    if centrality.get('degree_top_5'):
        grants = [g[:15] for g, _ in centrality['degree_top_5']]
        scores = [s for _, s in centrality['degree_top_5']]
        ax.barh(range(len(grants)), scores, color='salmon')
        ax.set_yticks(range(len(grants)))
        ax.set_yticklabels(grants)
        ax.set_title('Top Grants by Degree Centrality')
        ax.set_xlabel('Centrality Score')
    
    # 5. Community structure
    ax = axes[1, 1]
    community = analysis_results.get('community_structure', {})
    if community:
        metrics = ['Communities', 'Modularity']
        values = [community.get('num_communities', 0), 
                 community.get('modularity', 0) * 10]  # Scale for visualization
        ax.bar(metrics, values, color=['purple', 'teal'])
        ax.set_title('Community Structure')
        if community.get('modularity', 0) > 0:
            ax.text(1, community.get('modularity', 0) * 10 + 0.1,
                   f"M={community['modularity']:.3f}", ha='center')
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')
    
    summary = [
        ['Total Grants', str(node_types.get('grant', 0))],
        ['Total Institutes', str(node_types.get('institute', 0))],
        ['FQHC Focused', str(node_types.get('fqhc', 0))],
        ['Graph Density', f"{stats.get('density', 0):.4f}"],
    ]
    
    table = ax.table(cellText=summary, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title('Graph Summary', pad=20)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase5_knowledge_graph.png', dpi=300, bbox_inches='tight')
        print("💾 Saved visualization to phase5_knowledge_graph.png")
    
    plt.show()


def run_phase5():
    """Main Phase 5 execution with Weaviate integration"""
    
    print("\n" + "="*70)
    print("🚀 STARTING PHASE 5: WEAVIATE KNOWLEDGE GRAPH INTEGRATION")
    print("="*70)
    
    # 1. Connect to Weaviate (your 458 chunks from Phase 4)
    print("\n🔗 STEP 1: CONNECTING TO WEAVIATE")
    print("-" * 50)
    
    weaviate_bridge = WeaviateKnowledgeGraphBridge()
    if not weaviate_bridge.is_connected:
        print("❌ Cannot proceed without Weaviate connection")
        return
    
    # 2. Build Knowledge Graph from Weaviate data
    print("\n🏗️  STEP 2: BUILDING KNOWLEDGE GRAPH")
    print("-" * 50)
    
    kg_builder = KnowledgeGraphBuilder(weaviate_bridge)
    knowledge_graph = kg_builder.build_from_weaviate()
    
    # 3. Analyze Graph
    print("\n📊 STEP 3: ANALYZING KNOWLEDGE GRAPH")
    print("-" * 50)
    
    analysis = kg_builder.analyze_graph()
    print(f"\n📈 Key Findings:")
    print(f"  • Grants in graph: {analysis.get('node_types', {}).get('grant', 0)}")
    print(f"  • Total relationships: {analysis.get('basic_stats', {}).get('edges', 0)}")
    print(f"  • Graph density: {analysis.get('basic_stats', {}).get('density', 0):.4f}")
    
    if 'community_structure' in analysis:
        print(f"  • Communities detected: {analysis['community_structure'].get('num_communities', 0)}")
        print(f"  • Modularity: {analysis['community_structure'].get('modularity', 0):.3f}")
    
    # 4. Test Graph Traversal
    print("\n🔍 STEP 4: TESTING GRAPH TRAVERSAL")
    print("-" * 50)
    
    # Get a sample grant
    grant_nodes = [n for n, d in knowledge_graph.nodes(data=True) 
                  if d.get('type') == 'grant']
    
    if grant_nodes:
        sample_grant = grant_nodes[0]
        print(f"\n🔗 Finding related grants for: {sample_grant}")
        
        related = kg_builder.find_related_grants(sample_grant, max_depth=2, max_results=3)
        
        for i, rel in enumerate(related, 1):
            print(f"  {i}. {rel['grant_id']}")
            print(f"     • Strength: {rel['relationship_strength']:.3f}")
            print(f"     • Path: {rel['path']}")
    
    # 5. Test Graph-Augmented RAG
    print("\n🤖 STEP 5: TESTING GRAPH-AUGMENTED RAG")
    print("-" * 50)
    
    graph_rag = GraphAugmentedRAG(weaviate_bridge, knowledge_graph)
    
    test_queries = [
        "diabetes prevention community health centers",
        "behavioral health FQHC funding",
        "community health worker interventions"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("  " + "-" * 40)
        
        # Pure vector search (Phase 4 baseline)
        vector_results = graph_rag.retrieve(query, k=2)
        print("  📄 Vector Search Results:")
        for r in vector_results:
            print(f"    • {r['grant_id']} - {r['institute']} ({r['similarity_score']:.3f})")
        
        # Graph-expanded search
        expanded_results = graph_rag.retrieve_with_graph_expansion(query, vector_k=2, graph_k=1)
        print("\n  🔄 With Graph Expansion:")
        for r in expanded_results:
            if r.get('retrieval_type') == 'graph_expansion':
                print(f"    • {r['grant_id']} - {graph_rag.explain_result(r)}")
    
    # 6. Visualization
    print("\n📊 STEP 6: GENERATING VISUALIZATIONS")
    print("-" * 50)
    
    visualize_knowledge_graph(knowledge_graph, analysis)
    
    # 7. Save Results
    print("\n💾 STEP 7: SAVING RESULTS")
    print("-" * 50)
    
    kg_builder.save_graph("phase5_knowledge_graph.gml")
    
    results = {
        "phase": "phase5_knowledge_graph",
        "timestamp": datetime.now().isoformat(),
        "graph_stats": analysis.get("basic_stats", {}),
        "node_types": dict(analysis.get("node_types", {})),
        "edge_types": dict(analysis.get("edge_types", {})),
        "community_structure": analysis.get("community_structure", {}),
        "weaviate_connected": weaviate_bridge.is_connected,
        "grants_in_graph": analysis.get("node_types", {}).get("grant", 0)
    }
    
    with open("phase5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 8. Cleanup
    weaviate_bridge.close()
    
    print("\n" + "="*70)
    print("✅ PHASE 5 COMPLETE!")
    print("="*70)
    print("\n📁 Output files:")
    print("  • phase5_results.json - Complete results")
    print("  • phase5_knowledge_graph.png - Visualization")
    print("  • phase5_knowledge_graph.gml - Graph file")
    print("\n🎉 All 5 Phases Complete!")
    print("   Final product: Weaviate Vector DB + Knowledge Graph")
    
    return results


if __name__ == "__main__":
    run_phase5()