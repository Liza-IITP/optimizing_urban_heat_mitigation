import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

class UrbanHeatPINN(nn.Module):
    def __init__(self, input_dim):
        super(UrbanHeatPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def calculate_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

def physics_informed_loss(y_pred, y_true, ndvi, albedo, bah, tah, ndwi):
    mse_loss = nn.MSELoss()(y_pred, y_true)
    
    T_kelvin = y_pred + 273.15
    sigma = 5.67e-8
    emissivity = 0.97
    LW_out = emissivity * sigma * (T_kelvin ** 4)
    
    # Beer-Lambert AOD attenuation — Delhi peak summer AOD ≈ 0.55
    SW_in = 800.0 * torch.exp(torch.tensor(-0.55))
    LW_in = 350.0
    R_net = (1.0 - albedo) * SW_in + LW_in - LW_out
    
    Q_f = bah + tah
    
    # Urban sensible heat: 50 W/m²/K is physically correct for low-roughness rooftops
    T_air_kelvin = 313.15
    H = 50.0 * (T_kelvin - T_air_kelvin)
    
    # NDWI added to Latent Heat Flux — water bodies contribute strongly via evaporation
    LE = 300.0 * ndvi + 500.0 * torch.clamp(ndwi, min=0.0)
    
    G = 0.1 * R_net
    
    seb_residual = (R_net + Q_f) - (H + LE + G)
    physics_penalty = torch.mean(seb_residual ** 2)
    
    lambda_phy = 0.001  # Reduced: physics regularises, MSE drives convergence
    total_loss = mse_loss + lambda_phy * physics_penalty
    
    return total_loss

def main():
    print("--- Starting PINN Training V4 (Kundu et al. 2026 Optimization) ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'delhi_thermal_features.csv')
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    print("1. Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    
    # Mocking new features as per Kundu et al. (2026) requirements
    if 'NDWI' not in df.columns:
        print("NDWI not found. Mocking NDWI values...")
        np.random.seed(42)
        df['NDWI'] = np.random.uniform(-0.1, 0.5, size=len(df))
        
    if 'Zone_Core' not in df.columns:
        print("Zone_Core not found. Deriving spatial classification from BAH...")
        # Core: Densely packed areas (BAH > median), Peri-Urban: otherwise
        df['Zone_Core'] = (df['BAH'] > df['BAH'].median()).astype(int)
    
    feature_cols = ['NDVI', 'Albedo', 'BAH', 'TAH', 'NDWI', 'Zone_Core']
    target_col = 'LST_Celsius'
    
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    # Extract unscaled features for the physics loss
    ndvi_train = torch.FloatTensor(X_train[:, 0:1]).to(device)
    albedo_train = torch.FloatTensor(X_train[:, 1:2]).to(device)
    bah_train = torch.FloatTensor(X_train[:, 2:3]).to(device)
    tah_train = torch.FloatTensor(X_train[:, 3:4]).to(device)
    ndwi_train = torch.FloatTensor(X_train[:, 4:5]).to(device)
    
    dataset = TensorDataset(X_train_t, y_train_t, ndvi_train, albedo_train, bah_train, tah_train, ndwi_train)
    batch_size = 4096
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("2. Initializing Physics-Informed Neural Network (V4)...")
    model = UrbanHeatPINN(input_dim=len(feature_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    print(f"3. Training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y, batch_ndvi, batch_albedo, batch_bah, batch_tah, batch_ndwi in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = physics_informed_loss(predictions, batch_y, batch_ndvi, batch_albedo, batch_bah, batch_tah, batch_ndwi)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(dataloader.dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {epoch_loss:.4f}")
            
    print("4. Evaluating PINN V4...")
    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t)
        y_pred = y_pred_t.cpu().numpy()
        
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_r, _ = pearsonr(y_test.flatten(), y_pred.flatten())
    bias = calculate_bias(y_test, y_pred)
    
    print("\n" + "="*45)
    print("🏆 FINAL MODEL COMPARISON (V4 SPATIAL PINN)")
    print("="*45)
    print(f"Metric         | PINN V4")
    print(f"---------------+---------")
    print(f"R²             | {r2:.4f}")
    print(f"RMSE (°C)      | {rmse:.4f}")
    print(f"Pearson's r    | {pearson_r:.4f}")
    print(f"Bias (°C)      | {bias:.4f}")
    print("="*45)
    
    save_path = os.path.join(model_dir, 'pinn_delhi_v4.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler_X.mean_,
        'scaler_scale': scaler_X.scale_,
        'feature_cols': feature_cols
    }, save_path)
    print(f"\nSuccess! Trained PINN V4 weights and scaler saved to {save_path}")

if __name__ == "__main__":
    main()
