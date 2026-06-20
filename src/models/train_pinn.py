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

def physics_informed_loss(y_pred, y_true, albedo, bah, tah):
    """
    Custom Loss = Standard MSE + Physics Penalty
    The physics penalty enforces a simplified Surface Energy Balance.
    """
    # 1. Data-driven Loss (MSE)
    mse_loss = nn.MSELoss()(y_pred, y_true)
    
    # 2. Physics-Informed Loss (Surface Energy Balance)
    # CRITICAL FIX: Convert predicted LST from Celsius to Kelvin before Stefan-Boltzmann calc
    T_kelvin = y_pred + 273.15
    sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2 K^4)
    emissivity = 0.97  # Assumed average urban emissivity
    
    # Calculate Outgoing Longwave Radiation (W/m^2)
    LW_out = emissivity * sigma * (T_kelvin ** 4)
    
    # Simplified Available Energy (R_net + Q_f)
    # Assuming clear-sky summer conditions for Delhi:
    # Incoming Shortwave (SW_in ~ 800 W/m^2) and Longwave (LW_in ~ 350 W/m^2)
    SW_in = 800.0
    LW_in = 350.0
    
    R_net = (1.0 - albedo) * SW_in + LW_in - LW_out
    
    # Anthropogenic Heat proxy acting as a W/m^2 addition
    Q_f = bah + tah  
    
    available_energy = R_net + Q_f
    
    # Physics Constraint: During peak summer day, Available Energy must be strictly positive.
    # We penalize the model if it predicts a temperature so wildly high that available energy goes negative.
    physics_penalty = torch.mean(torch.relu(-available_energy))
    
    # Combine losses
    lambda_phy = 0.01  # Scaling weight for the physics constraint
    total_loss = mse_loss + lambda_phy * physics_penalty
    
    return total_loss

def main():
    print("--- Starting PINN Training (Pillar 3) ---")
    
    # 1. Paths and Data Loading
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'delhi_thermal_features.csv')
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
        
    print("1. Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    
    # Use the top 4 engineered features as specified
    feature_cols = ['NDVI', 'Albedo', 'BAH', 'TAH']
    target_col = 'LST_Celsius'
    
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    # Split data (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features (Standardization is crucial for stable Neural Network training)
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
    
    # Extract unscaled Albedo, BAH, TAH specifically for the physics loss function
    # Index mapping: NDVI=0, Albedo=1, BAH=2, TAH=3
    albedo_train = torch.FloatTensor(X_train[:, 1:2]).to(device)
    bah_train = torch.FloatTensor(X_train[:, 2:3]).to(device)
    tah_train = torch.FloatTensor(X_train[:, 3:4]).to(device)
    
    # Create DataLoader
    dataset = TensorDataset(X_train_t, y_train_t, albedo_train, bah_train, tah_train)
    batch_size = 4096 # Large batch size to handle the 1.5M rows efficiently
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("2. Initializing Physics-Informed Neural Network...")
    model = UrbanHeatPINN(input_dim=len(feature_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20 # Kept relatively small for rapid experimentation
    print(f"3. Training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y, batch_albedo, batch_bah, batch_tah in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_X)
            
            # Compute PINN loss (Data MSE + Physics SEB constraint)
            loss = physics_informed_loss(predictions, batch_y, batch_albedo, batch_bah, batch_tah)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(dataloader.dataset)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {epoch_loss:.4f}")
            
    print("4. Evaluating PINN against LightGBM Baseline...")
    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t)
        y_pred = y_pred_t.cpu().numpy()
        
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_r, _ = pearsonr(y_test.flatten(), y_pred.flatten())
    bias = calculate_bias(y_test, y_pred)
    
    print("\n" + "="*45)
    print("🏆 FINAL MODEL COMPARISON (PILLAR 3)")
    print("="*45)
    print(f"Metric         | LightGBM Baseline | PINN")
    print(f"---------------+-------------------+---------")
    print(f"R²             | 0.4112            | {r2:.4f}")
    print(f"RMSE (°C)      | 1.9871            | {rmse:.4f}")
    print(f"Pearson's r    | 0.6406            | {pearson_r:.4f}")
    print(f"Bias (°C)      | -0.0051           | {bias:.4f}")
    print("="*45)
    
    # 5. Save the Model
    save_path = os.path.join(model_dir, 'pinn_delhi.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler_X.mean_,
        'scaler_scale': scaler_X.scale_,
        'feature_cols': feature_cols
    }, save_path)
    
    print(f"\nSuccess! Trained PINN weights and scaler saved to {save_path}")
    print("Ready for Day 4 Scenarios!")

if __name__ == "__main__":
    main()
