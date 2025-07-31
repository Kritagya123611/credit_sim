import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
import numpy as np
import sys
import os
sys.path.append('..')
from models.gnn_models import create_credit_risk_model

def load_graph_data(path='../output/graph_data.pt'):
    import torch_geometric.data.data
    allowed_classes = [torch_geometric.data.data.DataEdgeAttr]
    with torch.serialization.safe_globals(allowed_classes):
        data = torch.load(path, weights_only=False)
    return data

def train_credit_risk_model(model_type='gcn', epochs=300, lr=0.01):
    print(f"Training {model_type.upper()} model for credit risk assessment")

    # Load dataset with PyTorch 2.6+ safe deserialization
    try:
        data = load_graph_data()
        print(f"Loaded graph data: {data.x.shape[0]} nodes, {data.x.shape[1]} features")
        print(f"Risk score range: [{data.risk_scores.min():.3f}, {data.risk_scores.max():.3f}]")
        print(f"Risk rating classes: {data.risk_ratings.unique().tolist()}")
    except Exception as e:
        print("Error loading graph data:", e)
        return None

    # Create train, validation, and test splits (60/20/20)
    num_nodes = data.x.size(0)
    indices = torch.randperm(num_nodes)

    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    # Initialize model
    model = create_credit_risk_model(model_type=model_type,
                                    num_features=data.x.size(1),
                                    num_classes=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # ✅ FIXED: Removed verbose=True parameter
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20)

    best_val_rmse = float('inf')
    patience, patience_counter = 50, 0

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass for both tasks
        pred_risk_score = model(data.x, data.edge_index, task='regression')
        pred_risk_rating = model(data.x, data.edge_index, task='classification')

        # Compute multi-task loss
        loss_risk_score = F.mse_loss(pred_risk_score[train_mask], data.risk_scores[train_mask])
        loss_risk_rating = F.nll_loss(pred_risk_rating[train_mask], data.risk_ratings[train_mask])

        # Combined loss: weight classification less
        loss = loss_risk_score + 0.3 * loss_risk_rating

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_pred = model(data.x, data.edge_index, task='regression')[val_mask]
                val_true = data.risk_scores[val_mask]
                val_rmse = torch.sqrt(F.mse_loss(val_pred, val_true)).item()
                
                # ✅ ENHANCED: More informative logging since verbose is removed
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:03d}: Loss={loss.item():.4f}, Val RMSE={val_rmse:.4f}, LR={current_lr:.6f}")
                
                scheduler.step(val_rmse)
                
                # Early stopping
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    patience_counter = 0
                    torch.save(model.state_dict(), f'../output/best_{model_type}_model.pt')
                    print(f"    → New best model saved! RMSE: {val_rmse:.4f}")
                else:
                    patience_counter += 10
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

    # Load best model weights
    model.load_state_dict(torch.load(f'../output/best_{model_type}_model.pt'))

    # Final evaluation on test set
    test_rmse = evaluate(model, data, test_mask)
    print(f"Test RMSE (Risk Score Regression): {test_rmse:.4f}")
    detailed_evaluation(model, data, test_mask)
    return model

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, task='regression')[mask]
        true = data.risk_scores[mask]
        rmse = torch.sqrt(F.mse_loss(pred, true)).item()
    return rmse

def detailed_evaluation(model, data, mask):
    model.eval()
    with torch.no_grad():
        pred_risk_score = model(data.x, data.edge_index, task='regression')[mask].cpu().numpy()
        pred_risk_rating = model(data.x, data.edge_index, task='classification')[mask].cpu().numpy()

        true_risk_score = data.risk_scores[mask].cpu().numpy()
        true_risk_rating = data.risk_ratings[mask].cpu().numpy()

        # Regression metrics
        rmse = np.sqrt(mean_squared_error(true_risk_score, pred_risk_score))
        mae = np.mean(np.abs(true_risk_score - pred_risk_score))
        r2 = 1 - np.var(true_risk_score - pred_risk_score) / np.var(true_risk_score)

        print(f"\n{'='*50}")
        print(f"FINAL EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"\nRegression Metrics (Risk Score Prediction):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")

        # Classification metrics
        pred_risk_classes = np.argmax(pred_risk_rating, axis=1)
        target_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        
        print(f"\nClassification Metrics (Risk Rating Prediction):")
        print(classification_report(true_risk_rating, pred_risk_classes, target_names=target_names))

        # Binary high-risk detection (risk > 0.7)
        high_risk_true = (true_risk_score > 0.7).astype(int)
        high_risk_pred_scores = pred_risk_score
        high_risk_pred_binary = (pred_risk_score > 0.7).astype(int)
        
        if len(np.unique(high_risk_true)) > 1:
            auc = roc_auc_score(high_risk_true, high_risk_pred_scores)
            precision = np.sum((high_risk_pred_binary == 1) & (high_risk_true == 1)) / max(np.sum(high_risk_pred_binary == 1), 1)
            recall = np.sum((high_risk_pred_binary == 1) & (high_risk_true == 1)) / max(np.sum(high_risk_true == 1), 1)
            f1 = 2 * precision * recall / max((precision + recall), 1e-10)
            
            print(f"\nHigh-Risk Detection (threshold > 0.7):")
            print(f"  AUC:       {auc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
            # Additional insights
            total_high_risk = np.sum(high_risk_true)
            total_predicted_high_risk = np.sum(high_risk_pred_binary)
            print(f"  True high-risk borrowers: {total_high_risk}")
            print(f"  Predicted high-risk: {total_predicted_high_risk}")
        else:
            print("\nHigh-Risk Detection: Insufficient variation in risk scores for binary classification")

if __name__ == "__main__":
    # Train models sequentially
    results = {}
    
    for model_name in ['gcn', 'gat', 'ppr_gcn']:
        print("\n" + "="*60)
        print(f"Training model: {model_name.upper()}")
        print("="*60)
        
        try:
            trained_model = train_credit_risk_model(model_type=model_name, epochs=300, lr=0.01)
            if trained_model:
                print(f"\n✅ {model_name.upper()} model training completed successfully!")
                results[model_name] = "SUCCESS"
            else:
                print(f"\n❌ {model_name.upper()} model training failed!")
                results[model_name] = "FAILED"
        except Exception as e:
            print(f"\n❌ {model_name.upper()} model crashed: {str(e)}")
            results[model_name] = f"ERROR: {str(e)}"
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for model, status in results.items():
        print(f"{model.upper()}: {status}")
