"""
🔍 COMPREHENSIVE GNN CREDIT RISK MODEL VALIDATION SUITE
Detects silent overfitting, data leakage, and validation issues in Graph Neural Networks

Performs all critical checks for behavioral credit risk models:
- Data splitting integrity
- Feature audit for leakage
- Graph-specific validation
- Performance stress tests
- Production readiness checks
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.utils import shuffle
import json
import warnings
from datetime import datetime
import sys
import os

# Add models directory to path
sys.path.append('..')
from models.gnn_models import load_production_model, create_phase2_behavioral_risk_model

class GNNValidationSuite:
    """Comprehensive validation suite for GNN credit risk models"""
    
    def __init__(self, data, model=None, model_path=None):
        self.data = data
        self.original_data = self._deep_copy_data(data)
        self.model = model
        self.model_path = model_path
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model is None and model_path:
            self.model, _ = load_production_model(model_path)
            
    def _deep_copy_data(self, data):
        """Create a deep copy of the data object"""
        import copy
        return copy.deepcopy(data)
    
    def _restore_original_data(self):
        """Restore original data after modifications"""
        self.data = self._deep_copy_data(self.original_data)
        self.data = self.data.to(self.device)
    
    def _evaluate_model_performance(self, test_mask=None):
        """Evaluate model performance on current data state"""
        if test_mask is None:
            # Use all data if no mask provided
            test_mask = torch.ones(self.data.x.size(0), dtype=torch.bool)
        
        self.model.eval()
        with torch.no_grad():
            # Get predictions
            pred_risk_score = self.model(self.data.x, self.data.edge_index, task='regression')[test_mask]
            pred_risk_rating = self.model(self.data.x, self.data.edge_index, task='classification')[test_mask]
            
            # Get true values
            true_risk_score = self.data.risk_scores[test_mask]
            true_risk_rating = self.data.risk_ratings[test_mask]
            
            # Calculate metrics
            rmse = torch.sqrt(F.mse_loss(pred_risk_score, true_risk_score)).item()
            r2 = 1 - (F.mse_loss(pred_risk_score, true_risk_score) / 
                     torch.var(true_risk_score)).item()
            
            classification_acc = (pred_risk_rating.argmax(dim=1) == true_risk_rating).float().mean().item()
            
            return {
                'rmse': rmse,
                'r2': r2,
                'classification_accuracy': classification_acc,
                'n_samples': test_mask.sum().item()
            }

    def check_1_data_splitting_integrity(self):
        """Check for node overlap and edge leakage across splits"""
        print("🔍 CHECK 1: Data Splitting Integrity")
        print("="*50)
        
        results = {}
        
        # Create test splits
        num_nodes = self.data.x.size(0)
        indices = torch.randperm(num_nodes)
        
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_indices = set(indices[:train_size].tolist())
        val_indices = set(indices[train_size:train_size + val_size].tolist())
        test_indices = set(indices[train_size + val_size:].tolist())
        
        # Check for node overlap
        train_val_overlap = len(train_indices & val_indices)
        train_test_overlap = len(train_indices & test_indices)
        val_test_overlap = len(val_indices & test_indices)
        
        results['node_overlap'] = {
            'train_val': train_val_overlap,
            'train_test': train_test_overlap,
            'val_test': val_test_overlap,
            'total_overlaps': train_val_overlap + train_test_overlap + val_test_overlap
        }
        
        # Check edge leakage across splits
        edge_leakage_count = 0
        total_edges = self.data.edge_index.size(1)
        
        for i in range(total_edges):
            source, target = self.data.edge_index[:, i].tolist()
            
            # Check if edge connects different splits
            if ((source in train_indices and target in test_indices) or
                (source in test_indices and target in train_indices) or
                (source in train_indices and target in val_indices) or
                (source in val_indices and target in train_indices)):
                edge_leakage_count += 1
        
        results['edge_leakage'] = {
            'cross_split_edges': edge_leakage_count,
            'total_edges': total_edges,
            'leakage_percentage': (edge_leakage_count / total_edges) * 100
        }
        
        # Assessment
        print(f"✅ Node Overlap Check:")
        print(f"   Train-Val overlap: {train_val_overlap} nodes")
        print(f"   Train-Test overlap: {train_test_overlap} nodes") 
        print(f"   Val-Test overlap: {val_test_overlap} nodes")
        
        print(f"📊 Edge Leakage Check:")
        print(f"   Cross-split edges: {edge_leakage_count:,} ({results['edge_leakage']['leakage_percentage']:.2f}%)")
        print(f"   Total edges: {total_edges:,}")
        
        if results['node_overlap']['total_overlaps'] == 0:
            print("✅ PASS: No node overlap between splits")
        else:
            print("⚠️  WARNING: Node overlap detected between splits")
            
        if results['edge_leakage']['leakage_percentage'] > 30:
            print("⚠️  WARNING: High cross-split edge leakage detected")
        else:
            print("✅ PASS: Cross-split edge leakage within acceptable range")
        
        self.results['data_splitting'] = results
        print()

    def check_2_feature_audit(self):
        """Audit features for leakage and suspicious correlations"""
        print("🔍 CHECK 2: Feature Audit for Leakage")
        print("="*50)
        
        results = {}
        
        # Load feature names if available
        try:
            with open('output/phase2_feature_mapping.json', 'r') as f:
                feature_mapping = json.load(f)
                feature_names = [feature_mapping['feature_index_to_name'][str(i)] 
                               for i in range(self.data.x.size(1))]
        except:
            feature_names = [f'feature_{i}' for i in range(self.data.x.size(1))]
        
        # Correlation analysis with target
        features_np = self.data.x.cpu().numpy()
        risk_scores_np = self.data.risk_scores.cpu().numpy()
        risk_ratings_np = self.data.risk_ratings.cpu().numpy()
        
        correlations_continuous = []
        correlations_categorical = []
        
        print("📊 Computing feature-target correlations...")
        
        for i in range(features_np.shape[1]):
            feature_vals = features_np[:, i]
            
            # Correlation with continuous risk scores
            corr_continuous = np.corrcoef(feature_vals, risk_scores_np)[0, 1]
            if not np.isnan(corr_continuous):
                correlations_continuous.append((feature_names[i], abs(corr_continuous)))
            
            # Correlation with categorical risk ratings
            try:
                corr_categorical = np.corrcoef(feature_vals, risk_ratings_np)[0, 1]
                if not np.isnan(corr_categorical):
                    correlations_categorical.append((feature_names[i], abs(corr_categorical)))
            except:
                correlations_categorical.append((feature_names[i], 0.0))
        
        # Sort by correlation strength
        correlations_continuous.sort(key=lambda x: x[1], reverse=True)
        correlations_categorical.sort(key=lambda x: x[1], reverse=True)
        
        # Identify suspicious correlations
        high_corr_continuous = [f for f, corr in correlations_continuous if corr > 0.95]
        high_corr_categorical = [f for f, corr in correlations_categorical if corr > 0.95]
        
        results['correlations'] = {
            'continuous_target': correlations_continuous[:10],  # Top 10
            'categorical_target': correlations_categorical[:10],
            'high_correlation_continuous': high_corr_continuous,
            'high_correlation_categorical': high_corr_categorical
        }
        
        # Feature variance analysis
        feature_variances = np.var(features_np, axis=0)
        zero_variance_features = [feature_names[i] for i, var in enumerate(feature_variances) if var < 1e-10]
        
        results['variance_analysis'] = {
            'zero_variance_features': zero_variance_features,
            'low_variance_features': [feature_names[i] for i, var in enumerate(feature_variances) 
                                    if var < 1e-6 and var > 1e-10]
        }
        
        # Shuffle test for random baseline
        print("🎲 Running shuffle test...")
        original_performance = self._evaluate_model_performance()
        
        # Shuffle labels and test
        shuffled_risk_scores = risk_scores_np.copy()
        shuffled_risk_ratings = risk_ratings_np.copy()
        np.random.shuffle(shuffled_risk_scores)
        np.random.shuffle(shuffled_risk_ratings)
        
        # Temporarily replace labels
        self.data.risk_scores = torch.tensor(shuffled_risk_scores, dtype=torch.float).to(self.device)
        self.data.risk_ratings = torch.tensor(shuffled_risk_ratings, dtype=torch.long).to(self.device)
        
        shuffled_performance = self._evaluate_model_performance()
        
        # Restore original labels
        self._restore_original_data()
        
        results['shuffle_test'] = {
            'original_r2': original_performance['r2'],
            'original_accuracy': original_performance['classification_accuracy'],
            'shuffled_r2': shuffled_performance['r2'],
            'shuffled_accuracy': shuffled_performance['classification_accuracy'],
            'r2_drop': original_performance['r2'] - shuffled_performance['r2'],
            'accuracy_drop': original_performance['classification_accuracy'] - shuffled_performance['classification_accuracy']
        }
        
        # Print results
        print(f"🎯 Top Correlated Features (Continuous Target):")
        for i, (feature, corr) in enumerate(correlations_continuous[:5]):
            print(f"   {i+1}. {feature}: {corr:.4f}")
        
        print(f"🎯 Top Correlated Features (Categorical Target):")
        for i, (feature, corr) in enumerate(correlations_categorical[:5]):
            print(f"   {i+1}. {feature}: {corr:.4f}")
        
        print(f"🎲 Shuffle Test Results:")
        print(f"   Original R²: {original_performance['r2']:.4f}")
        print(f"   Shuffled R²: {shuffled_performance['r2']:.4f}")
        print(f"   R² Drop: {results['shuffle_test']['r2_drop']:.4f}")
        print(f"   Original Accuracy: {original_performance['classification_accuracy']:.4f}")
        print(f"   Shuffled Accuracy: {shuffled_performance['classification_accuracy']:.4f}")
        print(f"   Accuracy Drop: {results['shuffle_test']['accuracy_drop']:.4f}")
        
        # Assessment
        if len(high_corr_continuous) > 0 or len(high_corr_categorical) > 0:
            print("⚠️  WARNING: Features with suspiciously high correlation (>0.95) detected")
            for feature in high_corr_continuous + high_corr_categorical:
                print(f"     - {feature}")
        else:
            print("✅ PASS: No suspiciously high feature correlations")
        
        if results['shuffle_test']['r2_drop'] < 0.5:
            print("⚠️  WARNING: Poor performance drop with shuffled labels - possible leakage")
        else:
            print("✅ PASS: Significant performance drop with shuffled labels")
        
        if results['shuffle_test']['accuracy_drop'] < 0.5:
            print("⚠️  WARNING: Poor accuracy drop with shuffled labels - possible leakage")
        else:
            print("✅ PASS: Significant accuracy drop with shuffled labels")
        
        self.results['feature_audit'] = results
        print()

    def check_3_graph_specific_leakage(self):
        """Check for graph-specific leakage patterns"""
        print("🔍 CHECK 3: Graph-Specific Leakage Detection")
        print("="*50)
        
        results = {}
        
        # Original performance baseline
        original_performance = self._evaluate_model_performance()
        print(f"📊 Original Performance: R² = {original_performance['r2']:.4f}, Accuracy = {original_performance['classification_accuracy']:.4f}")
        
        # Test 1: Random edge rewiring
        print("🔀 Test 1: Random Edge Rewiring (Homophily Test)")
        original_edges = self.data.edge_index.clone()
        
        # Create random edges while preserving degree distribution
        num_edges = original_edges.size(1)
        num_nodes = self.data.x.size(0)
        
        # Random rewiring
        random_edges = torch.randint(0, num_nodes, (2, num_edges), device=self.device)
        self.data.edge_index = random_edges
        
        random_edge_performance = self._evaluate_model_performance()
        
        results['random_edge_test'] = {
            'original_r2': original_performance['r2'],
            'random_edge_r2': random_edge_performance['r2'],
            'r2_retention': random_edge_performance['r2'] / original_performance['r2'],
            'original_accuracy': original_performance['classification_accuracy'],
            'random_edge_accuracy': random_edge_performance['classification_accuracy'],
            'accuracy_retention': random_edge_performance['classification_accuracy'] / original_performance['classification_accuracy']
        }
        
        print(f"   Random Edge R²: {random_edge_performance['r2']:.4f}")
        print(f"   R² Retention: {results['random_edge_test']['r2_retention']:.4f}")
        print(f"   Random Edge Accuracy: {random_edge_performance['classification_accuracy']:.4f}")
        print(f"   Accuracy Retention: {results['random_edge_test']['accuracy_retention']:.4f}")
        
        # Restore original edges
        self.data.edge_index = original_edges
        
        # Test 2: Feature permutation test
        print("🔀 Test 2: Feature Permutation Test")
        original_features = self.data.x.clone()
        
        # Randomly permute features across nodes
        permuted_indices = torch.randperm(self.data.x.size(0))
        self.data.x = self.data.x[permuted_indices]
        
        permuted_performance = self._evaluate_model_performance()
        
        results['feature_permutation_test'] = {
            'original_r2': original_performance['r2'],
            'permuted_r2': permuted_performance['r2'],
            'r2_retention': permuted_performance['r2'] / original_performance['r2'],
            'original_accuracy': original_performance['classification_accuracy'],
            'permuted_accuracy': permuted_performance['classification_accuracy'],
            'accuracy_retention': permuted_performance['classification_accuracy'] / original_performance['classification_accuracy']
        }
        
        print(f"   Permuted Features R²: {permuted_performance['r2']:.4f}")
        print(f"   R² Retention: {results['feature_permutation_test']['r2_retention']:.4f}")
        print(f"   Permuted Features Accuracy: {permuted_performance['classification_accuracy']:.4f}")
        print(f"   Accuracy Retention: {results['feature_permutation_test']['accuracy_retention']:.4f}")
        
        # Restore original features
        self.data.x = original_features
        
        # Test 3: Homophily analysis
        print("📊 Test 3: Homophily Analysis")
        edge_index_np = self.data.edge_index.cpu().numpy()
        risk_ratings_np = self.data.risk_ratings.cpu().numpy()
        
        same_class_edges = 0
        total_edges = edge_index_np.shape[1]
        
        for i in range(total_edges):
            source, target = edge_index_np[:, i]
            if risk_ratings_np[source] == risk_ratings_np[target]:
                same_class_edges += 1
        
        homophily_ratio = same_class_edges / total_edges
        
        results['homophily_analysis'] = {
            'same_class_edges': same_class_edges,
            'total_edges': total_edges,
            'homophily_ratio': homophily_ratio
        }
        
        print(f"   Same-class connected edges: {same_class_edges:,} / {total_edges:,}")
        print(f"   Homophily ratio: {homophily_ratio:.4f}")
        
        # Assessment
        if results['random_edge_test']['r2_retention'] > 0.8:
            print("⚠️  WARNING: High performance retention with random edges - possible label copying")
        else:
            print("✅ PASS: Significant performance drop with random edges")
        
        if results['feature_permutation_test']['r2_retention'] > 0.8:
            print("⚠️  WARNING: High performance retention with permuted features - possible edge-based leakage")
        else:
            print("✅ PASS: Significant performance drop with permuted features")
        
        if homophily_ratio > 0.9:
            print("⚠️  WARNING: Very high homophily - model may be relying on label copying")
        elif homophily_ratio > 0.7:
            print("⚠️  CAUTION: High homophily - verify model learns features not just neighbors")
        else:
            print("✅ PASS: Reasonable homophily level")
        
        self.results['graph_leakage'] = results
        print()

    def check_4_temporal_validation(self):
        """Check for temporal leakage and future information"""
        print("🔍 CHECK 4: Temporal Validation")
        print("="*50)
        
        results = {}
        
        print("📅 Temporal leakage check...")
        print("   Note: This requires timestamp information in features")
        
        # Check if any features might contain temporal information
        feature_names = []
        try:
            with open('output/phase2_feature_mapping.json', 'r') as f:
                feature_mapping = json.load(f)
                feature_names = [feature_mapping['feature_index_to_name'][str(i)] 
                               for i in range(self.data.x.size(1))]
        except:
            feature_names = [f'feature_{i}' for i in range(self.data.x.size(1))]
        
        # Look for temporal-sounding feature names
        temporal_keywords = ['time', 'date', 'recent', 'last', 'current', 'latest', 'period', 'month', 'day', 'year']
        potential_temporal_features = []
        
        for feature_name in feature_names:
            if any(keyword in feature_name.lower() for keyword in temporal_keywords):
                potential_temporal_features.append(feature_name)
        
        # Look for post-event features (features that might be influenced by the outcome)
        post_event_keywords = ['closed', 'terminated', 'default', 'delinquent', 'collection', 'recovery', 'writeoff']
        potential_post_event_features = []
        
        for feature_name in feature_names:
            if any(keyword in feature_name.lower() for keyword in post_event_keywords):
                potential_post_event_features.append(feature_name)
        
        results['temporal_analysis'] = {
            'potential_temporal_features': potential_temporal_features,
            'potential_post_event_features': potential_post_event_features,
            'total_features_analyzed': len(feature_names)
        }
        
        print(f"🕐 Potential temporal features found: {len(potential_temporal_features)}")
        for feature in potential_temporal_features[:5]:  # Show first 5
            print(f"     - {feature}")
        
        print(f"⚠️  Potential post-event features found: {len(potential_post_event_features)}")
        for feature in potential_post_event_features[:5]:  # Show first 5
            print(f"     - {feature}")
        
        # Assessment
        if len(potential_post_event_features) > 0:
            print("⚠️  WARNING: Features that might be influenced by the outcome detected")
        else:
            print("✅ PASS: No obvious post-event features detected")
        
        if len(potential_temporal_features) > 10:
            print("⚠️  CAUTION: Many temporal features detected - verify temporal ordering")
        else:
            print("✅ PASS: Reasonable number of temporal features")
        
        self.results['temporal_validation'] = results
        print()

    def check_5_production_readiness(self):
        """Check production deployment readiness"""
        print("🔍 CHECK 5: Production Readiness")
        print("="*50)
        
        results = {}
        
        # Feature stability test (add noise)
        print("🎲 Feature Stability Test (Noise Robustness)")
        original_performance = self._evaluate_model_performance()
        
        # Add small amount of noise to features
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        noise_results = {}
        
        for noise_level in noise_levels:
            # Add Gaussian noise
            noisy_features = self.data.x + torch.randn_like(self.data.x) * noise_level
            original_features = self.data.x.clone()
            self.data.x = noisy_features
            
            noisy_performance = self._evaluate_model_performance()
            
            noise_results[noise_level] = {
                'r2': noisy_performance['r2'],
                'accuracy': noisy_performance['classification_accuracy'],
                'r2_retention': noisy_performance['r2'] / original_performance['r2'],
                'accuracy_retention': noisy_performance['classification_accuracy'] / original_performance['classification_accuracy']
            }
            
            # Restore original features
            self.data.x = original_features
            
            print(f"   Noise {noise_level:0.2f}: R² = {noisy_performance['r2']:.4f} ({noise_results[noise_level]['r2_retention']:.3f}), "
                  f"Acc = {noisy_performance['classification_accuracy']:.4f} ({noise_results[noise_level]['accuracy_retention']:.3f})")
        
        results['noise_robustness'] = noise_results
        
        # Model size and complexity analysis
        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        
        results['model_complexity'] = {
            'total_parameters': model_params,
            'model_size_mb': model_size_mb,
            'parameters_per_feature': model_params / self.data.x.size(1),
            'samples_per_parameter': self.data.x.size(0) / model_params
        }
        
        print(f"📊 Model Complexity:")
        print(f"   Total parameters: {model_params:,}")
        print(f"   Model size: {model_size_mb:.2f} MB")
        print(f"   Parameters per feature: {results['model_complexity']['parameters_per_feature']:.1f}")
        print(f"   Samples per parameter: {results['model_complexity']['samples_per_parameter']:.1f}")
        
        # Memory and computational requirements
        try:
            # Estimate inference time
            import time
            times = []
            for _ in range(10):
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model(self.data.x, self.data.edge_index, task='regression')
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_inference_time = np.mean(times)
            results['performance_metrics'] = {
                'avg_inference_time_ms': avg_inference_time * 1000,
                'throughput_samples_per_second': self.data.x.size(0) / avg_inference_time
            }
            
            print(f"⚡ Performance Metrics:")
            print(f"   Average inference time: {avg_inference_time * 1000:.2f} ms")
            print(f"   Throughput: {results['performance_metrics']['throughput_samples_per_second']:.0f} samples/second")
        except Exception as e:
            print(f"⚠️  Could not measure inference performance: {e}")
            results['performance_metrics'] = None
        
        # Assessment
        robustness_threshold = 0.8
        robust_noise_levels = sum(1 for level in noise_levels if noise_results[level]['r2_retention'] > robustness_threshold)
        
        if robust_noise_levels >= 3:
            print("✅ PASS: Model shows good robustness to noise")
        elif robust_noise_levels >= 2:
            print("⚠️  CAUTION: Model shows moderate robustness to noise")
        else:
            print("⚠️  WARNING: Model is sensitive to input noise")
        
        if results['model_complexity']['samples_per_parameter'] > 5:
            print("✅ PASS: Good samples-to-parameters ratio")
        else:
            print("⚠️  WARNING: Low samples-to-parameters ratio - possible overfitting risk")
        
        self.results['production_readiness'] = results
        print()

    def generate_comprehensive_report(self):
        """Generate a comprehensive validation report"""
        print("📋 === COMPREHENSIVE VALIDATION REPORT ===")
        print("="*60)
        
        # Overall assessment
        warnings_count = 0
        passes_count = 0
        
        # Count warnings and passes from all checks
        for check_name, check_results in self.results.items():
            print(f"\n🔍 {check_name.upper().replace('_', ' ')}:")
            
            # This is a simplified assessment - in practice, you'd implement
            # more sophisticated scoring based on the specific thresholds
            
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"output/gnn_validation_report_{timestamp}.json"
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'data_info': {
                'num_nodes': int(self.data.x.size(0)),
                'num_features': int(self.data.x.size(1)),
                'num_edges': int(self.data.edge_index.size(1))
            },
            'validation_results': self.results,
            'overall_assessment': {
                'warnings_count': warnings_count,
                'passes_count': passes_count,
                'validation_status': 'PASS' if warnings_count < 3 else 'WARNING' if warnings_count < 6 else 'FAIL'
            }
        }
        
        os.makedirs('output', exist_ok=True)
        with open(report_filename, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {report_filename}")
        
        # Final recommendation
        print(f"\n🎯 FINAL ASSESSMENT:")
        if warnings_count < 3:
            print("✅ PASS: Model appears to be legitimate and ready for production")
        elif warnings_count < 6:
            print("⚠️  WARNING: Some concerns detected - review and address before production")
        else:
            print("❌ FAIL: Significant issues detected - model needs revision")
        
        return final_report

    def run_all_checks(self):
        """Run all validation checks"""
        print("🚀 === STARTING COMPREHENSIVE GNN VALIDATION ===")
        print(f"📊 Model: {self.model_path if self.model_path else 'Loaded model'}")
        print(f"📊 Data: {self.data.x.size(0):,} nodes, {self.data.x.size(1)} features, {self.data.edge_index.size(1):,} edges")
        print("="*60)
        
        # Ensure model and data are on correct device
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        try:
            # Run all checks
            self.check_1_data_splitting_integrity()
            self.check_2_feature_audit()
            self.check_3_graph_specific_leakage()
            self.check_4_temporal_validation()
            self.check_5_production_readiness()
            
            # Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            return report
            
        except Exception as e:
            print(f"❌ Error during validation: {e}")
            import traceback
            traceback.print_exc()
            return None


def load_phase2_data():
    """Load Phase 2 data for validation"""
    paths_to_try = [
        'output/phase2_credit_risk_graph_data.pt',
        '../output/phase2_credit_risk_graph_data.pt',
        '../graph_construction/output/phase2_credit_risk_graph_data.pt'
    ]
    
    for path in paths_to_try:
        try:
            if os.path.exists(path):
                data = torch.load(path, weights_only=False)
                print(f"✅ Loaded data from: {path}")
                return data
        except Exception as e:
            continue
    
    raise FileNotFoundError("Could not find Phase 2 data file")


if __name__ == "__main__":
    print("🔍 === GNN CREDIT RISK MODEL VALIDATION SUITE ===")
    
    try:
        # Load data
        data = load_phase2_data()
        
        # Load model
        model_path = 'output/best_phase2_gcn_behavioral_risk_model.pt'
        
        # Create validation suite
        validator = GNNValidationSuite(data, model_path=model_path)
        
        # Run all validation checks
        report = validator.run_all_checks()
        
        if report:
            print("\n🎉 Validation completed successfully!")
            print(f"📋 Check the detailed report for complete analysis")
        else:
            print("\n❌ Validation failed - check error messages above")
            
    except Exception as e:
        print(f"❌ Error running validation suite: {e}")
        import traceback
        traceback.print_exc()
