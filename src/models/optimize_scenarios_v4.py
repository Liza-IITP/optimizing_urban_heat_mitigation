import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import warnings
warnings.filterwarnings('ignore')

class UrbanHeatPINN(nn.Module):
    def __init__(self, input_dim):
        super(UrbanHeatPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

class CoolingOptimizationV4(Problem):
    def __init__(self, pinn_model, base_features, scaler_mean, scaler_scale):
        self.pinn_model = pinn_model
        # base_features = [NDVI, Albedo, BAH, TAH, NDWI, Zone_Core]
        self.base_features = base_features  
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.N = len(base_features)

        # Variables: N NDVI values, N Albedo values, N NDWI values
        xl = np.zeros(3 * self.N)
        xu = np.zeros(3 * self.N)
        
        orig_ndvi = base_features[:, 0]
        orig_albedo = base_features[:, 1]
        orig_ndwi = base_features[:, 4]
        zone_core = base_features[:, 5]

        # NDVI bounds: up to 0.60 everywhere
        xl[:self.N] = np.clip(orig_ndvi, 0.10, 0.60)
        xu[:self.N] = 0.60
        
        # Albedo bounds:
        # Zone_Core == 1: allow up to 0.65
        # Zone_Core == 0: strict constraint, e.g., max 0.35
        max_albedo = np.where(zone_core == 1, 0.65, 0.35)
        xl[self.N:2*self.N] = np.clip(orig_albedo, 0.15, max_albedo)
        xu[self.N:2*self.N] = max_albedo
        
        # NDWI bounds:
        # Zone_Core == 1: max increase of 0.05
        # Zone_Core == 0: up to 0.50
        max_ndwi = np.where(zone_core == 1, orig_ndwi + 0.05, 0.50)
        # Ensure max_ndwi doesn't exceed 0.50 globally just in case
        max_ndwi = np.clip(max_ndwi, a_min=orig_ndwi, a_max=0.50)
        xl[2*self.N:] = np.clip(orig_ndwi, -0.1, max_ndwi)
        xu[2*self.N:] = max_ndwi

        super().__init__(n_var=3*self.N, n_obj=1, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        new_ndvi = X[:, :self.N]
        new_albedo = X[:, self.N:2*self.N]
        new_ndwi = X[:, 2*self.N:]

        orig_ndvi = self.base_features[:, 0]
        orig_albedo = self.base_features[:, 1]
        orig_ndwi = self.base_features[:, 4]

        ndvi_change = new_ndvi - orig_ndvi
        albedo_change = new_albedo - orig_albedo
        ndwi_change = new_ndwi - orig_ndwi
        total_change = np.sum(ndvi_change + albedo_change + ndwi_change, axis=1)
        
        # Budget: average 0.20 change allowed per pixel
        budget = self.N * 0.20 
        g1 = total_change - budget 

        BAH = np.tile(self.base_features[:, 2], (pop_size, 1))
        TAH = np.tile(self.base_features[:, 3], (pop_size, 1))
        Zone_Core = np.tile(self.base_features[:, 5], (pop_size, 1))
        
        features = np.stack([new_ndvi, new_albedo, BAH, TAH, new_ndwi, Zone_Core], axis=2)
        features_flat = features.reshape(-1, 6)

        features_scaled = (features_flat - self.scaler_mean) / self.scaler_scale
        features_t = torch.FloatTensor(features_scaled)

        with torch.no_grad():
            pred_lst_t = self.pinn_model(features_t)
            pred_lst = pred_lst_t.numpy().reshape(pop_size, self.N)

        f1 = np.mean(pred_lst, axis=1)

        out["F"] = f1
        out["G"] = g1

def main():
    print("--- Starting Pillar 4: NSGA-II Scenario Optimization V4 (Spatial Planning) ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    data_path = os.path.join(base_dir, 'data', 'processed', 'delhi_thermal_features.csv')
    df = pd.read_csv(data_path)
    
    # Mocking new features if not present
    if 'NDWI' not in df.columns:
        np.random.seed(42)
        df['NDWI'] = np.random.uniform(-0.1, 0.5, size=len(df))
    if 'Zone_Core' not in df.columns:
        df['Zone_Core'] = (df['BAH'] > df['BAH'].median()).astype(int)
    
    model_path = os.path.join(base_dir, 'models', 'pinn_delhi_v4.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model V4 not found at {model_path}. Please run train_pinn_v4.py first.")
        
    checkpoint = torch.load(model_path, weights_only=False)
    
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    feature_cols = checkpoint['feature_cols']
    
    model = UrbanHeatPINN(input_dim=len(feature_cols))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("1. Isolating the top 100 extreme UHI hotspots...")
    df_sorted = df.sort_values(by='LST_Celsius', ascending=False)
    hotspots = df_sorted.head(100).copy()
    
    base_features = hotspots[feature_cols].values

    base_features_scaled = (base_features - scaler_mean) / scaler_scale
    with torch.no_grad():
        baseline_lst = model(torch.FloatTensor(base_features_scaled)).numpy().flatten()
    mean_baseline_lst = np.mean(baseline_lst)
    
    print(f"-> Baseline Average LST of Hotspots: {mean_baseline_lst:.2f} °C")

    print("2. Initializing NSGA-II V4 with Spatial Constraints (Zone_Core)...")
    problem = CoolingOptimizationV4(model, base_features, scaler_mean, scaler_scale)
    algorithm = NSGA2(pop_size=50)
    termination = get_termination("n_gen", 40)

    print("3. Hunting the Pareto Front for spatially bounded optimal interventions...")
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=42,
                   return_least_infeasible=True,
                   verbose=False)

    if res.F is None:
        best_lst = mean_baseline_lst
        optimal_X = np.zeros(300)
    else:
        best_lst = res.F[0] if isinstance(res.F, (list, np.ndarray)) else res.F
        optimal_X = res.X[0] if isinstance(res.X[0], (list, np.ndarray)) else res.X

    hotspots['Optimized_LST'] = 0.0
    # Recalculate per pixel for the best solution to get detailed delta T
    best_features = np.copy(base_features)
    best_features[:, 0] = optimal_X[:100]  # NDVI
    best_features[:, 1] = optimal_X[100:200]  # Albedo
    best_features[:, 4] = optimal_X[200:300]  # NDWI
    
    best_features_scaled = (best_features - scaler_mean) / scaler_scale
    with torch.no_grad():
        final_lst = model(torch.FloatTensor(best_features_scaled)).numpy().flatten()
    hotspots['Optimized_LST'] = final_lst
    hotspots['Delta_T'] = baseline_lst - final_lst
    
    # Calculate breakdown by Zone_Core
    core_hotspots = hotspots[hotspots['Zone_Core'] == 1]
    peri_hotspots = hotspots[hotspots['Zone_Core'] == 0]
    
    mean_delta_t_core = core_hotspots['Delta_T'].mean() if len(core_hotspots) > 0 else 0.0
    mean_delta_t_peri = peri_hotspots['Delta_T'].mean() if len(peri_hotspots) > 0 else 0.0

    print("\n" + "="*50)
    print("🌍 OPTIMIZATION COMPLETE (V4 SPATIAL PLANNING)")
    print("="*50)
    print(f"Pre-Intervention Hotspot LST:  {mean_baseline_lst:.2f} °C")
    print(f"Post-Intervention Hotspot LST: {np.mean(final_lst):.2f} °C")
    print(f"Total Cooling Achieved (ΔT):   {mean_baseline_lst - np.mean(final_lst):.2f} °C")
    print("--------------------------------------------------")
    print(f"ΔT in Dense Core (Zone 1):     {mean_delta_t_core:.2f} °C (Count: {len(core_hotspots)})")
    print(f"ΔT in Peri-Urban (Zone 0):     {mean_delta_t_peri:.2f} °C (Count: {len(peri_hotspots)})")
    print("="*50)
    
    output_path = os.path.join(base_dir, 'data', 'processed', 'optimal_scenario_v4.csv')
    hotspots.to_csv(output_path, index=False)
    print(f"\nOptimal pixel interventions saved to {output_path}")

if __name__ == "__main__":
    main()
