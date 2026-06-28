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

# Define the Physics-Informed Neural Network architecture
class UrbanHeatPINN(nn.Module):
    def __init__(self, input_dim):
        super(UrbanHeatPINN, self).__init__()
        # A deep network to capture complex non-linear urban heat dynamics
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

def physics_informed_loss(y_pred, y_true, ndvi, albedo, bah, tah, aod):
    """
    Custom Loss = Standard MSE + Physics Penalty
    The physics penalty enforces the complete Surface Energy Balance:
    R_net + Q_f = H + LE + G
    """
    # 1. Data-driven Loss (MSE)
    mse_loss = nn.MSELoss()(y_pred, y_true)
    
    # 2. Physics-Informed Loss (Surface Energy Balance)
    T_kelvin = y_pred + 273.15
    sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2 K^4)
    emissivity = 0.97  # Assumed average urban emissivity
    
    # Calculate Outgoing Longwave Radiation (W/m^2)
    LW_out = emissivity * sigma * (T_kelvin ** 4)
    
    # Incoming Shortwave is dynamically attenuated by Aerosol Optical Depth (AOD)
    # Using Beer-Lambert approximation
    SW_in_top = 800.0
    SW_in = SW_in_top * torch.exp(-aod)
    
    LW_in = 350.0
    
    R_net = (1.0 - albedo) * SW_in + LW_in - LW_out
    
    # Anthropogenic Heat proxy acting as a W/m^2 addition
    Q_f = bah + tah  
    
    # Sensible Heat Flux (H)
    # Roughly proportional to (LST - Air Temp). Assuming Air Temp = 40C (313.15K)
    T_air_kelvin = 313.15
    H = 20.0 * (T_kelvin - T_air_kelvin)
    
    # Latent Heat Flux (LE)
    # Assumed to scale proportionally with NDVI (evapotranspiration)
    # Max LE ~ 300 W/m^2 for dense vegetation
    LE = 300.0 * ndvi
    
    # Ground Heat Flux (G)
    # Assumed as a fraction of R_net during peak daytime
    G = 0.1 * R_net
    
    # Physics Constraint: Enforce Complete Balance (R_net + Q_f = H + LE + G)
    seb_residual = (R_net + Q_f) - (H + LE + G)
    physics_penalty = torch.mean(seb_residual ** 2)
    
    # Combine losses
    lambda_phy = 0.01  # Scaling weight for the physics constraint
    total_loss = mse_loss + lambda_phy * physics_penalty
    
    return total_loss

def main():
    print("--- Starting PINN Training V2 (Pillar 3) ---")
    
    # 1. Paths and Data Loading
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'delhi_thermal_features.csv')
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
        
    print("1. Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    
    # Ensure AOD is in the dataset, mock it if necessary for the pipeline to run
    if 'AOD' not in df.columns:
        print("AOD not found in data. Mocking AOD values for demonstration...")
        df['AOD'] = np.random.uniform(0.1, 0.8, size=len(df))
    
    # Use the top 5 engineered features including AOD
    feature_cols = ['NDVI', 'Albedo', 'BAH', 'TAH', 'AOD']
    target_col = 'LST_Celsius'
    
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    # Split data (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Setup PyTorch Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert scaled inputs to Tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    # Extract unscaled features specifically for the physics loss function
    # Index mapping: NDVI=0, Albedo=1, BAH=2, TAH=3, AOD=4
    ndvi_train = torch.FloatTensor(X_train[:, 0:1]).to(device)
    albedo_train = torch.FloatTensor(X_train[:, 1:2]).to(device)
    bah_train = torch.FloatTensor(X_train[:, 2:3]).to(device)
    tah_train = torch.FloatTensor(X_train[:, 3:4]).to(device)
    aod_train = torch.FloatTensor(X_train[:, 4:5]).to(device)
    
    # Create DataLoader
    dataset = TensorDataset(X_train_t, y_train_t, ndvi_train, albedo_train, bah_train, tah_train, aod_train)
    batch_size = 4096
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("2. Initializing Physics-Informed Neural Network (V2)...")
    model = UrbanHeatPINN(input_dim=len(feature_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    print(f"3. Training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y, batch_ndvi, batch_albedo, batch_bah, batch_tah, batch_aod in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_X)
            
            # Compute PINN loss (Data MSE + Physics SEB constraint)
            loss = physics_informed_loss(predictions, batch_y, batch_ndvi, batch_albedo, batch_bah, batch_tah, batch_aod)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(dataloader.dataset)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {epoch_loss:.4f}")
            
    print("4. Evaluating PINN V2...")
    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t)
        y_pred = y_pred_t.cpu().numpy()
        
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_r, _ = pearsonr(y_test.flatten(), y_pred.flatten())
    bias = calculate_bias(y_test, y_pred)
    
    print("\n" + "="*45)
    print("🏆 FINAL MODEL COMPARISON (PILLAR 3 V2)")
    print("="*45)
    print(f"Metric         | PINN V2")
    print(f"---------------+---------")
    print(f"R²             | {r2:.4f}")
    print(f"RMSE (°C)      | {rmse:.4f}")
    print(f"Pearson's r    | {pearson_r:.4f}")
    print(f"Bias (°C)      | {bias:.4f}")
    print("="*45)
    
    # 5. Save the Model
    save_path = os.path.join(model_dir, 'pinn_delhi_v2.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler_X.mean_,
        'scaler_scale': scaler_X.scale_,
        'feature_cols': feature_cols
    }, save_path)
    
    print(f"\nSuccess! Trained PINN V2 weights and scaler saved to {save_path}")
    print("Ready for Day 4 Scenarios!")

if __name__ == "__main__":
    main()
