import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import numpy as np

class CreditRiskGCN(nn.Module):
    """
    Graph Convolutional Network for Credit Risk Assessment
    Updated for 66 input features including enhanced graph metrics
    """
    def __init__(self, num_features=66, hidden_dim=128, num_classes=5):
        super(CreditRiskGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # Dual prediction heads for multi-task learning
        self.risk_regressor = nn.Linear(hidden_dim // 2, 1)      # Risk score (0-1)
        self.rating_classifier = nn.Linear(hidden_dim // 2, num_classes)  # Risk categories (5 classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index, task='regression'):
        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        if task == 'regression':
            # Risk score prediction (continuous 0-1)
            out = self.risk_regressor(x)
            return torch.sigmoid(out).squeeze()
        elif task == 'classification':
            # Risk rating prediction (categorical)
            out = self.rating_classifier(x)
            return F.log_softmax(out, dim=1)
        else:
            raise ValueError("Task must be 'regression' or 'classification'")

class CreditRiskGAT(nn.Module):
    """
    Graph Attention Network for Credit Risk Assessment
    Uses attention to identify important transaction relationships
    """
    def __init__(self, num_features=66, hidden_dim=128, heads=8, num_classes=5):
        super(CreditRiskGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=True, dropout=0.6)
        self.conv3 = GATConv(hidden_dim, hidden_dim // 2, heads=1, concat=True, dropout=0.6)
        
        # Dual prediction heads
        self.risk_regressor = nn.Linear(hidden_dim // 2, 1)
        self.rating_classifier = nn.Linear(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(0.6)
        
    def forward(self, x, edge_index, task='regression'):
        # Attention-based graph convolution
        x = self.dropout(x)
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv3(x, edge_index))
        
        if task == 'regression':
            out = self.risk_regressor(x)
            return torch.sigmoid(out).squeeze()
        elif task == 'classification':
            out = self.rating_classifier(x)
            return F.log_softmax(out, dim=1)
        else:
            raise ValueError("Task must be 'regression' or 'classification'")

class PersonalizedPageRankGCN(nn.Module):
    """
    Enhanced GCN with Personalized PageRank for Risk Propagation
    Specifically leverages the PPR features from high-risk borrowers
    """
    def __init__(self, num_features=66, hidden_dim=128, num_classes=5):
        super(PersonalizedPageRankGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # Enhanced with PPR feature processing (you have 10 PPR features from high-risk seeds)
        self.ppr_processor = nn.Linear(10, hidden_dim // 4)
        self.feature_fusion = nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 2)
        
        # Prediction heads
        self.risk_regressor = nn.Linear(hidden_dim // 2, 1)
        self.rating_classifier = nn.Linear(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index, task='regression'):
        # Standard GNN processing
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv3(h, edge_index))
        
        # Extract PPR features (assuming they're the last 10 features)
        ppr_features = x[:, -10:]  # Last 10 features are PPR from high-risk seeds
        ppr_processed = F.relu(self.ppr_processor(ppr_features))
        
        # Fuse standard GNN features with PPR features
        combined = torch.cat([h, ppr_processed], dim=1)
        fused = F.relu(self.feature_fusion(combined))
        
        if task == 'regression':
            out = self.risk_regressor(fused)
            return torch.sigmoid(out).squeeze()
        elif task == 'classification':
            out = self.rating_classifier(fused)
            return F.log_softmax(out, dim=1)
        else:
            raise ValueError("Task must be 'regression' or 'classification'")

class CreditRiskEnsemble(nn.Module):
    """
    Ensemble of GCN and GAT for robust credit risk assessment
    """
    def __init__(self, num_features=66, hidden_dim=128, num_classes=5):
        super(CreditRiskEnsemble, self).__init__()
        self.gcn = CreditRiskGCN(num_features, hidden_dim, num_classes)
        self.gat = CreditRiskGAT(num_features, hidden_dim, num_classes=num_classes)
        
        # Ensemble combination layers
        self.ensemble_regressor = nn.Linear(2, 1)
        self.ensemble_classifier = nn.Linear(2 * num_classes, num_classes)
        
    def forward(self, x, edge_index, task='regression'):
        gcn_out = self.gcn(x, edge_index, task=task)
        gat_out = self.gat(x, edge_index, task=task)
        
        if task == 'regression':
            # Combine regression outputs
            combined = torch.stack([gcn_out, gat_out], dim=1)
            return torch.sigmoid(self.ensemble_regressor(combined)).squeeze()
        elif task == 'classification':
            # Combine classification outputs
            combined = torch.cat([gcn_out, gat_out], dim=1)
            return F.log_softmax(self.ensemble_classifier(combined), dim=1)

def create_credit_risk_model(model_type='gcn', num_features=66, hidden_dim=128, num_classes=5):
    """
    Factory function to create credit risk models
    
    Args:
        model_type: 'gcn', 'gat', 'ppr_gcn', or 'ensemble'
        num_features: Number of input features (66 based on your output)
        hidden_dim: Hidden layer dimension
        num_classes: Number of risk rating classes (5 based on your data)
    """
    if model_type == 'gcn':
        return CreditRiskGCN(num_features, hidden_dim, num_classes)
    elif model_type == 'gat':
        return CreditRiskGAT(num_features, hidden_dim, num_classes=num_classes)
    elif model_type == 'ppr_gcn':
        return PersonalizedPageRankGCN(num_features, hidden_dim, num_classes)
    elif model_type == 'ensemble':
        return CreditRiskEnsemble(num_features, hidden_dim, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test model creation with your actual data dimensions
    num_features = 66  # Based on your feature engineering output
    num_classes = 5    # Based on your risk profile distribution
    
    models = {
        'gcn': create_credit_risk_model('gcn', num_features),
        'gat': create_credit_risk_model('gat', num_features), 
        'ppr_gcn': create_credit_risk_model('ppr_gcn', num_features),
        'ensemble': create_credit_risk_model('ensemble', num_features)
    }
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name.upper()} parameters: {params:,}")
    
    # Test with your actual dimensions
    batch_size = 10
    x = torch.randn(batch_size, num_features)
    edge_index = torch.randint(0, batch_size, (2, 20))
    
    # Test GCN
    gcn_model = models['gcn']
    risk_scores = gcn_model(x, edge_index, task='regression')
    risk_ratings = gcn_model(x, edge_index, task='classification')
    
    print(f"\nGCN outputs:")
    print(f"Risk scores shape: {risk_scores.shape}")
    print(f"Risk ratings shape: {risk_ratings.shape}")
    print(f"Risk scores range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
    
    print("\n✅ All models ready for your credit risk assessment system!")
