# ============================================================================
# 📁 phase5_evaluation.py - COMPLETE WORKING VERSION
# ============================================================================

import networkx as nx
import numpy as np
import json
from collections import defaultdict

class GraphGrantEvaluator:
    """
    Evaluates how well the knowledge graph finds relevant grants
    for RFP/target application matching
    """
    
    def __init__(self, graph):
        self.graph = graph
        self.grants = [n for n in graph.nodes if graph.nodes[n].get('type') == 'grant']
        
        # Create analyzer as instance variable
        self.analyzer = GraphAnalyzer(graph)
    
    def find_grants_by_combo(self, condition, intervention, population):
        """Find grants that match all three concepts"""
        
        cond_node = f"COND_{condition}"
        int_node = f"INT_{intervention}"
        pop_node = f"POP_{population}"
        
        if cond_node not in self.graph or int_node not in self.graph or pop_node not in self.graph:
            return []
        
        cond_grants = set(self.graph.neighbors(cond_node))
        int_grants = set(self.graph.neighbors(int_node))
        pop_grants = set(self.graph.neighbors(pop_node))
        
        # Grants that match all three
        matches = cond_grants & int_grants & pop_grants
        
        # Filter to only grant nodes
        return [g for g in matches if self.graph.nodes[g].get('type') == 'grant']
    
    def create_test_rfps(self, num_rfps=10):
        """Create realistic RFP scenarios from actual grant combinations"""
        
        rfps = []
        
        # Find grants that combine multiple concepts
        conditions = ['diabetes', 'hypertension', 'cancer', 'depression']
        interventions = ['CHW', 'telehealth', 'navigation', 'screening']
        populations = ['Latino', 'rural', 'pediatric', 'low_income']
        
        for condition in conditions:
            for intervention in interventions:
                for population in populations:
                    
                    # Find grants that match this combo
                    matching = self.find_grants_by_combo(condition, intervention, population)
                    
                    if len(matching) >= 2:  # At least 2 examples
                        rfps.append({
                            'rfp_id': f"RFP_{condition}_{intervention}_{population}",
                            'condition': condition,
                            'intervention': intervention,
                            'population': population,
                            'known_matching_grants': matching[:5],  # Ground truth
                            'expected_count': len(matching)
                        })
                    
                    if len(rfps) >= num_rfps:
                        break
                if len(rfps) >= num_rfps:
                    break
            if len(rfps) >= num_rfps:
                break
        
        return rfps
    
    def evaluate_rfp_matching(self, rfp):
        """How well does the graph find grants for this RFP?"""
        
        # Query the graph using analyzer
        results = self.analyzer.rfp_graph_query(
            conditions=[rfp['condition']],
            interventions=[rfp['intervention']],
            populations=[rfp['population']],
            fqhc_only=True
        )
        
        found_grants = [r['grant_id'] for r in results]
        ground_truth = rfp['known_matching_grants']
        
        # Metrics
        metrics = {
            'rfp_id': rfp['rfp_id'],
            'found_count': len(found_grants),
            'expected_count': rfp['expected_count'],
            
            # Recall: what % of known grants were found?
            'recall': len(set(found_grants) & set(ground_truth)) / len(ground_truth) if ground_truth else 0,
            
            # Precision: what % of found grants are in ground truth?
            'precision': len(set(found_grants) & set(ground_truth)) / len(found_grants) if found_grants else 0,
            
            # Novelty: how many new grants found?
            'novel_grants': len(set(found_grants) - set(ground_truth)),
            
            # Diversity of found grants
            'diversity': self._calculate_diversity(found_grants)
        }
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0
        
        return metrics
    
    def _calculate_diversity(self, grant_ids):
        """Calculate diversity of institutes and years"""
        if not grant_ids:
            return 0
        
        institutes = set()
        years = set()
        
        for g in grant_ids:
            data = self.graph.nodes[g]
            institutes.add(data.get('institute', 'Unknown'))
            years.add(data.get('year', 0))
        
        # Normalized diversity score (0-1)
        inst_diversity = len(institutes) / len(grant_ids) if grant_ids else 0
        year_diversity = len(years) / len(grant_ids) if grant_ids else 0
        
        return (inst_diversity + year_diversity) / 2
    
    def run_full_evaluation(self):
        """Complete evaluation of graph-based grant discovery"""
        
        print("\n" + "="*70)
        print("📊 PHASE 5: GRAPH-BASED GRANT DISCOVERY EVALUATION")
        print("="*70)
        
        # 1. Coverage
        print("\n1️⃣  COVERAGE ANALYSIS")
        print("-" * 50)
        coverage = self._analyze_coverage()
        
        # 2. Create test RFPs
        print("\n2️⃣  CREATING TEST RFPs")
        print("-" * 50)
        test_rfps = self.create_test_rfps(num_rfps=8)
        print(f"   Created {len(test_rfps)} test RFP scenarios")
        
        # 3. Evaluate each RFP
        print("\n3️⃣  RFP MATCHING PERFORMANCE")
        print("-" * 50)
        
        all_metrics = []
        for rfp in test_rfps:
            metrics = self.evaluate_rfp_matching(rfp)
            all_metrics.append(metrics)
            
            print(f"\n   📋 {rfp['rfp_id']}")
            print(f"      Found: {metrics['found_count']} grants (expected {metrics['expected_count']})")
            print(f"      Precision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f} | F1: {metrics['f1']:.2f}")
            print(f"      Novel grants: {metrics['novel_grants']} | Diversity: {metrics['diversity']:.2f}")
        
        # 4. Aggregate results
        print("\n4️⃣  OVERALL PERFORMANCE")
        print("-" * 50)
        
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_metrics])
        avg_novelty = np.mean([m['novel_grants'] for m in all_metrics])
        avg_diversity = np.mean([m['diversity'] for m in all_metrics])
        
        print(f"   Average Precision: {avg_precision:.3f}")
        print(f"   Average Recall:    {avg_recall:.3f}")
        print(f"   Average F1 Score:  {avg_f1:.3f}")
        print(f"   Average Novelty:   {avg_novelty:.1f} new grants per query")
        print(f"   Average Diversity: {avg_diversity:.3f}")
        
        # 5. Compare to Phase 4 (different goals)
        print("\n5️⃣  COMPARISON TO PHASE 4")
        print("-" * 50)
        print("   Phase 4 (Hybrid Search):")
        print("     • Goal: Find specific chunks by similarity")
        print("     • Metric: P@5 = 0.920")
        print("     • Output: Ranked text chunks")
        print("\n   Phase 5 (Knowledge Graph):")
        print("     • Goal: Discover related grants by concept")
        print("     • Metric: F1 = {:.3f}".format(avg_f1))
        print("     • Output: Connected grant IDs for exploration")
        print("     • Novelty: Finds {:.1f} grants not in training".format(avg_novelty))
        
        return {
            'coverage': coverage,
            'rfp_metrics': all_metrics,
            'aggregate': {
                'precision': float(avg_precision),
                'recall': float(avg_recall),
                'f1': float(avg_f1),
                'novelty': float(avg_novelty),
                'diversity': float(avg_diversity)
            }
        }
    
    def _analyze_coverage(self):
        """Analyze graph coverage of concepts"""
        
        conditions = [n for n in self.graph.nodes if self.graph.nodes[n].get('type') == 'condition']
        interventions = [n for n in self.graph.nodes if self.graph.nodes[n].get('type') == 'intervention']
        populations = [n for n in self.graph.nodes if self.graph.nodes[n].get('type') == 'population']
        
        grants_with_conditions = set()
        for c in conditions:
            grants_with_conditions.update([
                n for n in self.graph.neighbors(c) 
                if self.graph.nodes[n].get('type') == 'grant'
            ])
        
        grants_with_interventions = set()
        for i in interventions:
            grants_with_interventions.update([
                n for n in self.graph.neighbors(i) 
                if self.graph.nodes[n].get('type') == 'grant'
            ])
        
        grants_with_populations = set()
        for p in populations:
            grants_with_populations.update([
                n for n in self.graph.neighbors(p) 
                if self.graph.nodes[n].get('type') == 'grant'
            ])
        
        total_grants = len(self.grants)
        
        coverage = {
            'conditions': len(conditions),
            'interventions': len(interventions),
            'populations': len(populations),
            'grants_with_conditions': len(grants_with_conditions),
            'pct_with_conditions': len(grants_with_conditions)/total_grants*100,
            'grants_with_interventions': len(grants_with_interventions),
            'pct_with_interventions': len(grants_with_interventions)/total_grants*100,
            'grants_with_populations': len(grants_with_populations),
            'pct_with_populations': len(grants_with_populations)/total_grants*100,
        }
        
        print(f"   Conditions: {coverage['conditions']} types covering {coverage['pct_with_conditions']:.1f}% of grants")
        print(f"   Interventions: {coverage['interventions']} types covering {coverage['pct_with_interventions']:.1f}% of grants")
        print(f"   Populations: {coverage['populations']} types covering {coverage['pct_with_populations']:.1f}% of grants")
        
        return coverage


class GraphAnalyzer:
    """
    Graph analysis functions (copied from phase5_knowledge_graph.py)
    """
    
    def __init__(self, graph):
        self.graph = graph
    
    def rfp_graph_query(self, conditions=None, interventions=None, 
                        populations=None, fqhc_only=True):
        """Find grants matching RFP criteria"""
        
        if conditions is None:
            conditions = []
        if interventions is None:
            interventions = []
        if populations is None:
            populations = []
        
        scores = defaultdict(float)
        
        # Score by conditions
        for cond in conditions:
            cond_node = f"COND_{cond}"
            if cond_node in self.graph:
                for grant in self.graph.neighbors(cond_node):
                    if self.graph.nodes[grant].get('type') == 'grant':
                        scores[grant] += 1.5
        
        # Score by interventions
        for inter in interventions:
            int_node = f"INT_{inter}"
            if int_node in self.graph:
                for grant in self.graph.neighbors(int_node):
                    if self.graph.nodes[grant].get('type') == 'grant':
                        scores[grant] += 1.0
        
        # Score by populations
        for pop in populations:
            pop_node = f"POP_{pop}"
            if pop_node in self.graph:
                for grant in self.graph.neighbors(pop_node):
                    if self.graph.nodes[grant].get('type') == 'grant':
                        scores[grant] += 1.0
        
        # Filter to FQHC if requested
        if fqhc_only and "FQHC_HUB" in self.graph:
            fqhc_grants = {
                n for n in self.graph.neighbors("FQHC_HUB")
                if self.graph.nodes[n].get('type') == 'grant'
            }
            scores = {k: v for k, v in scores.items() if k in fqhc_grants}
        
        # Format results
        results = []
        for grant_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            node_data = dict(self.graph.nodes.get(grant_id, {}))
            results.append({
                'grant_id': grant_id,
                'match_score': round(score, 2),
                'institute': node_data.get('institute', 'Unknown'),
                'year': node_data.get('year', 'Unknown'),
                'is_fqhc_focused': bool(node_data.get('is_fqhc_focused', False))
            })
        
        return results


# ============ RUN THE EVALUATION ============

if __name__ == "__main__":
    # Load your graph
    print("Loading knowledge graph...")
    graph = nx.read_gml('phase5_knowledge_graph.gml')
    print(f"✅ Loaded graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Create evaluator
    evaluator = GraphGrantEvaluator(graph)
    
    # Run full evaluation
    results = evaluator.run_full_evaluation()
    
    # Save results
    with open('phase5_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Phase 5 evaluation complete! Results saved to phase5_evaluation_results.json")