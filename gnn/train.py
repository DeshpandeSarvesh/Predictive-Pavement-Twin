import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from gnn.dataset import load_pavement_dataset
from gnn.model import PavementPredictorGNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
# PHASE 4: TRAINING & BENCHMARKING PIPELINE
# =====================================================================
# This script trains the GNN, trains a static Random Forest baseline,
# and evaluates their ability to predict future pavement condition.
# =====================================================================

def train_and_benchmark(epochs=150, lr=0.002):
    # 1. Create output folders
    os.makedirs("models", exist_ok=True)
    
    # 2. Load the formatted dataset
    data_list, num_features = load_pavement_dataset()
    
    # 3. Train/Test Split (Temporal Split)
    # We use the first 80% of graph snapshots for training, and evaluate on the final 20%
    num_samples = len(data_list)
    split_idx = int(num_samples * 0.8)
    
    train_data = data_list[:split_idx]
    test_data = data_list[split_idx:]
    
    print(f"Data split: {len(train_data)} training graphs, {len(test_data)} testing graphs.")
    
    # =================================================================
    # CRITERION A: TRAINING THE GRAPH NEURAL NETWORK (GNN)
    # =================================================================
    print("\nTraining the Spatial Graph Attention Network (GNN)...")
    model = PavementPredictorGNN(in_channels=num_features, hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in train_data:
            optimizer.zero_grad()
            # Run GNN forward pass
            pred = model(data).squeeze()
            target = data.y.squeeze()
            
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d} | Average MSE Loss: {epoch_loss / len(train_data):.4f}")
            
    # Save the trained GNN weights
    model_save_path = "models/gnn_weights.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"GNN Model weights saved to: {model_save_path}")

    # =================================================================
    # CRITERION B: TRAINING THE STATIC NON-GRAPH BASELINE (Random Forest)
    # =================================================================
    print("\nTraining the Static Isolation Model (Random Forest)...")
    
    # Flatten the graph nodes into flat feature matrices for scikit-learn
    # We compile all nodes across all training time steps
    train_x_flat = []
    train_y_flat = []
    for data in train_data:
        train_x_flat.append(data.x.numpy())
        train_y_flat.append(data.y.numpy())
        
    train_x_flat = np.vstack(train_x_flat)  # Shape: [Time * Nodes, 42]
    train_y_flat = np.vstack(train_y_flat).squeeze()  # Shape: [Time * Nodes]
    
    # Train the Random Forest
    rf_baseline = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_baseline.fit(train_x_flat, train_y_flat)
    print("Static Random Forest model successfully trained!")

    # =================================================================
    # CRITERION C: EVALUATION & BENCHMARKING
    # =================================================================
    print("\nEvaluating models on the Test Set (Future snapshot predictions)...")
    model.eval()
    
    gnn_preds_all = []
    static_preds_all = []
    targets_all = []
    
    with torch.no_grad():
        for data in test_data:
            # GNN inference
            gnn_pred = model(data).numpy().squeeze()
            # Static model inference
            static_pred = rf_baseline.predict(data.x.numpy())
            
            target = data.y.numpy().squeeze()
            
            gnn_preds_all.append(gnn_pred)
            static_preds_all.append(static_pred)
            targets_all.append(target)
            
    gnn_preds_all = np.concatenate(gnn_preds_all)
    static_preds_all = np.concatenate(static_preds_all)
    targets_all = np.concatenate(targets_all)
    
    # Scale back to original 0-100 PCI values for accurate physical error metrics
    gnn_preds_all = gnn_preds_all * 100.0
    static_preds_all = static_preds_all * 100.0
    targets_all = targets_all * 100.0
    
    # Compute standard evaluation metrics
    # 1. GNN Metrics
    gnn_mae = mean_absolute_error(targets_all, gnn_preds_all)
    gnn_rmse = np.sqrt(mean_squared_error(targets_all, gnn_preds_all))
    gnn_r2 = r2_score(targets_all, gnn_preds_all)
    
    # 2. Static Model Metrics
    static_mae = mean_absolute_error(targets_all, static_preds_all)
    static_rmse = np.sqrt(mean_squared_error(targets_all, static_preds_all))
    static_r2 = r2_score(targets_all, static_preds_all)
    
    print("\n" + "="*50)
    print("                MODEL PERFORMANCE COMPARISON")
    print("="*50)
    print(f"1. Spatio-Temporal GNN (Ours):")
    print(f"   - Root Mean Squared Error (RMSE): {gnn_rmse:.4f} PCI")
    print(f"   - Mean Absolute Error (MAE):     {gnn_mae:.4f} PCI")
    print(f"   - Coefficient of Determination (R²): {gnn_r2:.4f}")
    print("-"*50)
    print(f"2. Static Isolation Model (Random Forest Baseline):")
    print(f"   - Root Mean Squared Error (RMSE): {static_rmse:.4f} PCI")
    print(f"   - Mean Absolute Error (MAE):     {static_mae:.4f} PCI")
    print(f"   - Coefficient of Determination (R²): {static_r2:.4f}")
    print("="*50)
    
    # Save the trained Random Forest baseline model for use in the dashboard if needed
    with open("models/static_baseline.pkl", "wb") as f:
        pickle.dump(rf_baseline, f)

if __name__ == "__main__":
    train_and_benchmark(epochs=150)
