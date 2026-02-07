# ============================================================================
# 📁 phase5_knowledge_graph.py - WEEK 9: KNOWLEDGE GRAPH INTEGRATION
# ============================================================================

"""
PHASE 5: KNOWLEDGE GRAPH-AUGMENTED RAG
EDIT THIS FILE FOR CUSTOM KNOWLEDGE GRAPH INTEGRATION
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
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime
from collections import defaultdict

# EDIT HERE: Import knowledge graph libraries
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    from community import community_louvain  # pip install python-louvain
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False
    print("⚠️  Knowledge graph libraries not available. Run: !pip install networkx python-louvain")

class KnowledgeGraphBuilder:
    """
    BUILD KNOWLEDGE GRAPH FROM GRANT DATA
    EDIT THIS CLASS FOR CUSTOM GRAPH CONSTRUCTION
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.node_types = {}
        self.edge_types = {}
        
    def build_from_grants(self, grants_df: pd.DataFrame):
        """
        BUILD GRAPH FROM GRANT DATAFRAME
        EDIT FOR CUSTOM GRAPH STRUCTURE
        """
        print(f"\n🏗️  Building knowledge graph from {len(grants_df)} grants...")
        
        # Clear existing graph
        self.graph.clear()
        
        # Add grant nodes
        for _, grant in grants_df.iterrows():
            grant_id = grant.get('grant_id', f"grant_{_}")
            self.graph.add_node(grant_id, type='grant', **grant.to_dict())
        
        # Add relationships
        self._add_citation_relationships(grants_df)
        self._add_topic_relationships(grants_df)
        self._add_institution_relationships(grants_df)
        self._add_temporal_relationships(grants_df)
        
        print(f"✅ Graph built with {self.graph.number_of_nodes()} nodes and "
              f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _add_citation_relationships(self, grants_df: pd.DataFrame):
        """Add citation relationships - EDIT FOR CUSTOM CITATION LOGIC"""
        print("  Adding citation relationships...")
        
        # Simple citation simulation (in real scenario, parse references)
        for i, grant1 in grants_df.iterrows():
            grant1_id = grant1.get('grant_id', f"grant_{i}")
            
            # Simulate citations to other grants
            for j, grant2 in grants_df.iterrows():
                if i == j:
                    continue
                
                grant2_id = grant2.get('grant_id', f"grant_{j}")
                
                # Simple similarity-based citation simulation
                if self._should_cite(grant1, grant2):
                    self.graph.add_edge(grant1_id, grant2_id, 
                                       type='cites', weight=0.8)
    
    def _add_topic_relationships(self, grants_df: pd.DataFrame):
        """Add topic-based relationships - EDIT FOR CUSTOM TOPIC LOGIC"""
        print("  Adding topic relationships...")
        
        # Extract topics from abstracts
        topics_to_grants = defaultdict(list)
        
        for _, grant in grants_df.iterrows():
            grant_id = grant.get('grant_id', f"grant_{_}")
            abstract = str(grant.get('abstract', ''))
            
            # Extract topics (simplified)
            topics = self._extract_topics(abstract)
            
            for topic in topics:
                topics_to_grants[topic].append(grant_id)
        
        # Connect grants sharing topics
        for topic, grant_ids in topics_to_grants.items():
            if len(grant_ids) > 1:
                for i in range(len(grant_ids)):
                    for j in range(i + 1, len(grant_ids)):
                        self.graph.add_edge(grant_ids[i], grant_ids[j],
                                           type='shares_topic', 
                                           topic=topic, weight=0.6)
    
    def _add_institution_relationships(self, grants_df: pd.DataFrame):
        """Add institution relationships - EDIT FOR CUSTOM INSTITUTION LOGIC"""
        print("  Adding institution relationships...")
        
        institution_to_grants = defaultdict(list)
        
        for _, grant in grants_df.iterrows():
            grant_id = grant.get('grant_id', f"grant_{_}")
            institution = grant.get('institution', 'Unknown')
            
            institution_to_grants[institution].append(grant_id)
        
        # Add institution nodes and connect grants
        for institution, grant_ids in institution_to_grants.items():
            institution_node = f"inst_{institution.replace(' ', '_')}"
            self.graph.add_node(institution_node, type='institution', 
                              name=institution)
            
            for grant_id in grant_ids:
                self.graph.add_edge(grant_id, institution_node,
                                   type='affiliated_with', weight=0.9)
    
    def _add_temporal_relationships(self, grants_df: pd.DataFrame):
        """Add temporal relationships - EDIT FOR CUSTOM TEMPORAL LOGIC"""
        print("  Adding temporal relationships...")
        
        # Group by year
        year_to_grants = defaultdict(list)
        
        for _, grant in grants_df.iterrows():
            grant_id = grant.get('grant_id', f"grant_{_}")
            year = grant.get('year', 2024)
            
            year_to_grants[year].append(grant_id)
        
        # Connect grants from same year
        for year, grant_ids in year_to_grants.items():
            year_node = f"year_{year}"
            self.graph.add_node(year_node, type='year', year=year)
            
            for grant_id in grant_ids:
                self.graph.add_edge(grant_id, year_node,
                                   type='published_in', weight=0.7)
    
    def _should_cite(self, grant1: pd.Series, grant2: pd.Series) -> bool:
        """Determine if grant1 should cite grant2 - EDIT FOR CUSTOM LOGIC"""
        # Simple similarity check
        text1 = str(grant1.get('abstract', ''))
        text2 = str(grant2.get('abstract', ''))
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        similarity = len(words1.intersection(words2)) / max(len(words1), 1)
        
        return similarity > 0.3
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text - EDIT FOR CUSTOM TOPIC EXTRACTION"""
        text_lower = text.lower()
        topics = []
        
        # FQHC-related topics
        fqhc_topics = [
            'diabetes', 'prevention', 'community health', 'behavioral health',
            'primary care', 'underserved', 'health disparities', 'screening',
            'chronic disease', 'mental health', 'substance use', 'pediatric',
            'maternal health', 'geriatric', 'homeless', 'lgbtq', 'rural'
        ]
        
        for topic in fqhc_topics:
            if topic in text_lower:
                topics.append(topic)
        
        return topics[:5]  # Limit to top 5 topics
    
    def analyze_graph(self) -> Dict:
        """
        ANALYZE GRAPH STRUCTURE AND PROPERTIES
        EDIT FOR CUSTOM ANALYSIS METRICS
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
        
        # Calculate centrality for grant nodes only
        grant_nodes = [n for n, d in self.graph.nodes(data=True) 
                      if d.get('type') == 'grant']
        
        if grant_nodes:
            subgraph = self.graph.subgraph(grant_nodes)
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(subgraph)
            analysis["centrality"]["degree_top_5"] = sorted(
                degree_centrality.items(), key=lambda x: x[1], reverse=True
            )[:5]
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(subgraph)
            analysis["centrality"]["betweenness_top_5"] = sorted(
                betweenness_centrality.items(), key=lambda x: x[1], reverse=True
            )[:5]
        
        # Detect communities
        if KG_AVAILABLE and len(grant_nodes) > 10:
            try:
                partition = community_louvain.best_partition(self.graph)
                analysis["community_structure"]["num_communities"] = len(set(partition.values()))
                analysis["community_structure"]["modularity"] = community_louvain.modularity(
                    partition, self.graph
                )
            except:
                pass
        
        return analysis
    
    def find_related_grants(self, grant_id: str, 
                           max_depth: int = 2, 
                           max_results: int = 10) -> List[Dict]:
        """
        FIND RELATED GRANTS USING GRAPH TRAVERSAL
        EDIT FOR CUSTOM RELATIONSHIP QUERIES
        """
        if grant_id not in self.graph:
            return []
        
        related = []
        visited = set()
        
        # Breadth-first search
        queue = [(grant_id, 0, [grant_id])]  # (node, depth, path)
        
        while queue and len(related) < max_results:
            current, depth, path = queue.pop(0)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Check if current is a grant node
            if current != grant_id and self.graph.nodes[current].get('type') == 'grant':
                # Calculate relationship strength
                strength = self._calculate_relationship_strength(grant_id, current, path)
                
                related.append({
                    "grant_id": current,
                    "depth": depth,
                    "path": path,
                    "relationship_strength": strength,
                    "node_data": self.graph.nodes[current]
                })
            
            # Explore neighbors if within depth limit
            if depth < max_depth:
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1, path + [neighbor]))
        
        # Sort by relationship strength
        related.sort(key=lambda x: x["relationship_strength"], reverse=True)
        
        return related[:max_results]
    
    def _calculate_relationship_strength(self, source: str, target: str, 
                                        path: List[str]) -> float:
        """Calculate relationship strength - EDIT FOR CUSTOM STRENGTH CALCULATION"""
        if len(path) < 2:
            return 0.0
        
        # Calculate product of edge weights along path
        strength = 1.0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1], {})
            weight = edge_data.get('weight', 0.5)
            strength *= weight
        
        # Adjust for path length
        strength *= (1.0 / len(path))
        
        return strength

class GraphAugmentedRAG:
    """
    KNOWLEDGE GRAPH-AUGMENTED RAG SYSTEM
    EDIT FOR CUSTOM GRAPH-RAG INTEGRATION
    """
    
    def __init__(self, knowledge_graph: nx.Graph, 
                 vector_search_func = None):
        self.knowledge_graph = knowledge_graph
        self.vector_search = vector_search_func
        
    def hybrid_retrieve(self, query: str, 
                       vector_top_k: int = 5,
                       graph_top_k: int = 5) -> List[Dict]:
        """
        HYBRID RETRIEVAL: VECTOR + GRAPH
        EDIT FOR CUSTOM HYBRID STRATEGY
        """
        results = []
        
        # Step 1: Vector search (if available)
        if self.vector_search:
            vector_results = self.vector_search(query, top_k=vector_top_k)
            results.extend(vector_results)
            
            # Step 2: Graph expansion for top vector results
            for result in vector_results[:3]:  # Expand top 3
                grant_id = result.get('grant_id')
                if grant_id and grant_id in self.knowledge_graph:
                    related_grants = self.knowledge_graph.find_related_grants(
                        grant_id, max_depth=2, max_results=graph_top_k
                    )
                    
                    # Add graph-based results
                    for related in related_grants:
                        results.append({
                            **related['node_data'],
                            'source': 'graph_expansion',
                            'original_grant': grant_id,
                            'relationship_strength': related['relationship_strength'],
                            'path_depth': related['depth']
                        })
        
        # Deduplicate and rank
        unique_results = self._deduplicate_results(results)
        ranked_results = self._rank_results(unique_results, query)
        
        return ranked_results[:vector_top_k + graph_top_k]
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate results - EDIT FOR CUSTOM DEDUPLICATION"""
        seen = set()
        unique = []
        
        for result in results:
            grant_id = result.get('grant_id')
            if grant_id and grant_id not in seen:
                seen.add(grant_id)
                unique.append(result)
        
        return unique
    
    def _rank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Rank results - EDIT FOR CUSTOM RANKING"""
        for result in results:
            # Calculate combined score
            vector_score = result.get('similarity_score', 0)
            graph_score = result.get('relationship_strength', 0)
            
            # Weighted combination (adjust weights as needed)
            result['combined_score'] = 0.7 * vector_score + 0.3 * graph_score
        
        return sorted(results, key=lambda x: x.get('combined_score', 0), reverse=True)
    
    def explain_retrieval(self, result: Dict) -> str:
        """
        GENERATE EXPLANATION FOR RETRIEVAL DECISION
        EDIT FOR CUSTOM EXPLANATION GENERATION
        """
        explanation_parts = []
        
        if result.get('source') == 'graph_expansion':
            original = result.get('original_grant', '')
            strength = result.get('relationship_strength', 0)
            depth = result.get('path_depth', 0)
            
            explanation_parts.append(
                f"Found via graph relationship to {original} "
                f"(strength: {strength:.2f}, path depth: {depth})"
            )
        
        if 'similarity_score' in result:
            explanation_parts.append(
                f"Semantic similarity: {result['similarity_score']:.3f}"
            )
        
        if 'relationship_strength' in result:
            explanation_parts.append(
                f"Graph relationship strength: {result['relationship_strength']:.3f}"
            )
        
        return "; ".join(explanation_parts)

# ============ VISUALIZATION FOR PHASE 5 ============

def visualize_knowledge_graph(knowledge_graph: nx.Graph, 
                             analysis_results: Dict,
                             sample_queries: List[str] = None,
                             save: bool = True):
    """
    VISUALIZE KNOWLEDGE GRAPH AND ANALYSIS
    EDIT FOR CUSTOM VISUALIZATIONS
    """
    if not KG_AVAILABLE:
        print("⚠️  NetworkX not available for visualization")
        return
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase 5: Knowledge Graph Integration', fontsize=16)
        
        # 1. Graph overview (simplified)
        ax = axes[0, 0]
        
        # Create a simplified visualization
        pos = nx.spring_layout(knowledge_graph, seed=42)
        
        # Color nodes by type
        node_colors = []
        for node in knowledge_graph.nodes():
            node_type = knowledge_graph.nodes[node].get('type', 'unknown')
            if node_type == 'grant':
                node_colors.append('lightblue')
            elif node_type == 'institution':
                node_colors.append('lightgreen')
            elif node_type == 'year':
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightgray')
        
        nx.draw_networkx_nodes(knowledge_graph, pos, ax=ax, 
                              node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(knowledge_graph, pos, ax=ax, 
                              alpha=0.3, width=1)
        
        # Add labels for important nodes
        important_nodes = []
        if 'centrality' in analysis_results:
            for centrality_type in ['degree_top_5', 'betweenness_top_5']:
                if centrality_type in analysis_results['centrality']:
                    for node, _ in analysis_results['centrality'][centrality_type]:
                        important_nodes.append(node)
        
        labels = {node: node[:10] for node in important_nodes[:5]}
        nx.draw_networkx_labels(knowledge_graph, pos, labels, ax=ax, font_size=8)
        
        ax.set_title('Knowledge Graph Overview')
        ax.axis('off')
        
        # 2. Node type distribution
        ax = axes[0, 1]
        if 'node_types' in analysis_results:
            types = list(analysis_results['node_types'].keys())
            counts = list(analysis_results['node_types'].values())
            
            ax.bar(range(len(types)), counts, color='skyblue')
            ax.set_xlabel('Node Type')
            ax.set_ylabel('Count')
            ax.set_title('Node Type Distribution')
            ax.set_xticks(range(len(types)))
            ax.set_xticklabels(types, rotation=45, ha='right')
        
        # 3. Edge type distribution
        ax = axes[0, 2]
        if 'edge_types' in analysis_results:
            types = list(analysis_results['edge_types'].keys())
            counts = list(analysis_results['edge_types'].values())
            
            ax.bar(range(len(types)), counts, color='lightgreen')
            ax.set_xlabel('Edge Type')
            ax.set_ylabel('Count')
            ax.set_title('Edge Type Distribution')
            ax.set_xticks(range(len(types)))
            ax.set_xticklabels(types, rotation=45, ha='right')
        
        # 4. Centrality analysis
        ax = axes[1, 0]
        if 'centrality' in analysis_results and 'degree_top_5' in analysis_results['centrality']:
            top_nodes = [n[:15] for n, _ in analysis_results['centrality']['degree_top_5']]
            centrality_values = [v for _, v in analysis_results['centrality']['degree_top_5']]
            
            ax.barh(range(len(top_nodes)), centrality_values, color='salmon')
            ax.set_yticks(range(len(top_nodes)))
            ax.set_yticklabels(top_nodes)
            ax.set_xlabel('Degree Centrality')
            ax.set_title('Top 5 Grants by Degree Centrality')
        
        # 5. Community structure
        ax = axes[1, 1]
        if 'community_structure' in analysis_results:
            comm_data = analysis_results['community_structure']
            
            if 'num_communities' in comm_data and 'modularity' in comm_data:
                metrics = ['Communities', 'Modularity']
                values = [comm_data['num_communities'], comm_data['modularity']]
                
                bars = ax.bar(metrics, values, color=['lightblue', 'lightgreen'])
                ax.set_ylabel('Value')
                ax.set_title('Community Structure Analysis')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 6. System summary
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        summary_data = []
        if 'basic_stats' in analysis_results:
            stats = analysis_results['basic_stats']
            summary_data.extend([
                ['Nodes', str(stats.get('nodes', 0))],
                ['Edges', str(stats.get('edges', 0))],
                ['Density', f"{stats.get('density', 0):.4f}"],
                ['Components', str(stats.get('connected_components', 0))]
            ])
        
        if 'community_structure' in analysis_results:
            comm = analysis_results['community_structure']
            summary_data.extend([
                ['Communities', str(comm.get('num_communities', 0))],
                ['Modularity', f"{comm.get('modularity', 0):.3f}"]
            ])
        
        if summary_data:
            table = ax.table(cellText=summary_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
        
        ax.set_title('Graph Statistics Summary')
        
        plt.tight_layout()
        
        if save:
            plt.savefig('phase5_knowledge_graph.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")

# ============ MAIN EXECUTION ============

def run_phase5_tests():
    """
    MAIN FUNCTION TO RUN PHASE 5 TESTS
    EDIT FOR CUSTOM TEST FLOW
    """
    print("\n" + "="*70)
    print("🚀 STARTING PHASE 5: KNOWLEDGE GRAPH INTEGRATION")
    print("="*70)
    
    if not KG_AVAILABLE:
        print("❌ Knowledge graph libraries not available. Installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                  "networkx", "python-louvain"])
            import networkx as nx
            from community import community_louvain
            global KG_AVAILABLE
            KG_AVAILABLE = True
        except:
            print("❌ Failed to install knowledge graph libraries")
            return {"error": "Knowledge graph libraries not available"}
    
    # 1. Load grant data
    print("\n📊 STEP 1: LOADING GRANT DATA")
    print("-" * 50)
    
    try:
        # Try to load from previous phases
        grants_df = pd.read_csv('api_fetched_grants.csv')
        print(f"✅ Loaded {len(grants_df)} grants from API data")
    except:
        try:
            grants_df = pd.read_csv('document_chunks_database.csv')
            # Extract grant-level data from chunks
            grants_df = grants_df.drop_duplicates('grant_id').copy()
            print(f"✅ Loaded {len(grants_df)} grants from chunk database")
        except:
            print("⚠️  No grant data found. Creating sample data...")
            grants_df = _create_sample_grants(50)
    
    # 2. Build knowledge graph
    print("\n🏗️  STEP 2: BUILDING KNOWLEDGE GRAPH")
    print("-" * 50)
    
    kg_builder = KnowledgeGraphBuilder()
    knowledge_graph = kg_builder.build_from_grants(grants_df)
    
    # Analyze graph
    analysis_results = kg_builder.analyze_graph()
    
    print(f"\n📈 Graph Analysis Results:")
    print(json.dumps(analysis_results, indent=2))
    
    # 3. Test graph traversal
    print("\n🔍 STEP 3: TESTING GRAPH TRAVERSAL")
    print("-" * 50)
    
    # Test with sample grants
    test_grant_ids = list(grants_df['grant_id'].head(3)) if 'grant_id' in grants_df.columns else []
    
    traversal_results = {}
    for grant_id in test_grant_ids[:3]:
        print(f"\n🔗 Finding related grants for: {grant_id}")
        related = kg_builder.find_related_grants(grant_id, max_depth=2, max_results=5)
        
        traversal_results[grant_id] = {
            "total_related": len(related),
            "top_related": [
                {
                    "grant": r["grant_id"],
                    "depth": r["depth"],
                    "strength": r["relationship_strength"]
                }
                for r in related[:3]
            ]
        }
        
        print(f"  Found {len(related)} related grants")
        for r in related[:3]:
            print(f"  • {r['grant_id']} (depth: {r['depth']}, strength: {r['relationship_strength']:.3f})")
    
    # 4. Test graph-augmented RAG
    print("\n🤖 STEP 4: TESTING GRAPH-AUGMENTED RAG")
    print("-" * 50)
    
    # Create mock vector search function
    def mock_vector_search(query: str, top_k: int = 5):
        """Mock vector search for testing - REPLACE WITH REAL IMPLEMENTATION"""
        # In real implementation, connect to your vector database
        return [
            {
                'grant_id': f"R01MD{100000 + i}",
                'title': f"Grant about {query.split()[0]} prevention",
                'similarity_score': 0.9 - (i * 0.1),
                'abstract': f"Sample abstract about {query}"
            }
            for i in range(top_k)
        ]
    
    # Create graph-augmented RAG system
    graph_rag = GraphAugmentedRAG(kg_builder, mock_vector_search)
    
    # Test queries
    test_queries = [
        "diabetes prevention in FQHCs",
        "behavioral health integration grants"
    ]
    
    hybrid_results = {}
    for query in test_queries:
        print(f"\n🔍 Hybrid retrieval for: '{query}'")
        results = graph_rag.hybrid_retrieve(query, vector_top_k=3, graph_top_k=2)
        
        hybrid_results[query] = {
            "total_results": len(results),
            "vector_results": sum(1 for r in results if r.get('source') != 'graph_expansion'),
            "graph_results": sum(1 for r in results if r.get('source') == 'graph_expansion'),
            "top_result": results[0].get('grant_id', 'N/A') if results else 'N/A'
        }
        
        print(f"  Found {len(results)} total results")
        print(f"  Vector results: {hybrid_results[query]['vector_results']}")
        print(f"  Graph results: {hybrid_results[query]['graph_results']}")
        
        # Show explanations for top results
        for i, result in enumerate(results[:2]):
            explanation = graph_rag.explain_retrieval(result)
            print(f"  Result {i+1}: {result.get('grant_id', 'N/A')} - {explanation}")
    
    # 5. Visualization
    print("\n📊 STEP 5: GENERATING VISUALIZATIONS")
    print("-" * 50)
    
    visualize_knowledge_graph(knowledge_graph, analysis_results, test_queries)
    
    # 6. Save results
    results = {
        "phase": "phase5_knowledge_graph",
        "timestamp": datetime.now().isoformat(),
        "graph_stats": analysis_results.get("basic_stats", {}),
        "traversal_results": traversal_results,
        "hybrid_rag_results": hybrid_results,
        "node_count": knowledge_graph.number_of_nodes(),
        "edge_count": knowledge_graph.number_of_edges(),
        "test_queries": test_queries
    }
    
    with open("phase5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save graph for future use
    try:
        nx.write_gml(knowledge_graph, "knowledge_graph.gml")
        print("💾 Saved knowledge graph to 'knowledge_graph.gml'")
    except:
        pass
    
    print("\n" + "="*70)
    print("✅ PHASE 5 COMPLETE!")
    print("="*70)
    print("\n📁 Results saved to:")
    print("  • phase5_results.json")
    print("  • phase5_knowledge_graph.png")
    print("  • knowledge_graph.gml")
    print("\n📈 Knowledge Graph Summary:")
    print(f"  • Nodes: {knowledge_graph.number_of_nodes()}")
    print(f"  • Edges: {knowledge_graph.number_of_edges()}")
    print(f"  • Grant nodes: {analysis_results.get('node_types', {}).get('grant', 0)}")
    print(f"  • Edge types: {len(analysis_results.get('edge_types', {}))}")
    print("\n🎉 ALL 5 PHASES COMPLETE!")
    
    return results

def _create_sample_grants(num_grants: int = 50) -> pd.DataFrame:
    """Create sample grants for testing"""
    grants = []
    institutions = [
        "University of Chicago", "Northwestern University", "Johns Hopkins",
        "UCLA", "University of Michigan", "University of Washington"
    ]
    
    institutes = ["NIMHD", "NIMH", "NCI", "NHLBI", "NIA", "NIDDK"]
    
    for i in range(num_grants):
        focus_area = ["diabetes", "mental health", "cancer", "cardiovascular", 
                     "aging", "substance use"][i % 6]
        
        grants.append({
            "grant_id": f"R01MD{100000 + i}",
            "title": f"{focus_area.title()} Intervention in FQHC Settings",
            "abstract": f"This study examines {focus_area} interventions in Federally "
                       f"Qualified Health Centers serving underserved populations. "
                       f"The research focuses on implementation science and health equity.",
            "year": 2022 + (i % 3),
            "institute": institutes[i % len(institutes)],
            "institution": institutions[i % len(institutions)],
            "focus_area": focus_area,
            "is_fqhc_focused": True
        })
    
    return pd.DataFrame(grants)

# ============================================================================
# 🏃‍♂️ RUN PHASE 5 TESTS
# ============================================================================

if __name__ == "__main__":
    # Run tests
    results = run_phase5_tests()