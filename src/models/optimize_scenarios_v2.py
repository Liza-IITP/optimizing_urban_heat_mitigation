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

# 1. Recreate the exact PINN architecture to load the trained weights
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

# 2. Define the Evolutionary Optimization Problem
class CoolingOptimization(Problem):
    def __init__(self, pinn_model, base_features, scaler_mean, scaler_scale):
        self.pinn_model = pinn_model
        self.base_features = base_features  # [NDVI, Albedo, BAH, TAH, AOD]
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.N = len(base_features)

        # Variables: N NDVI values followed by N Albedo values
        xl = np.zeros(2 * self.N)
        xu = np.zeros(2 * self.N)
        
        orig_ndvi = base_features[:, 0]
        orig_albedo = base_features[:, 1]
        orig_bah = base_features[:, 2] # BAH is index 2

        # Interventions bounds (can only improve the environment)
        # Greening: current NDVI up to 0.60
        xl[:self.N] = np.clip(orig_ndvi, 0.10, 0.60)
        xu[:self.N] = 0.60
        
        # Cool Roofs Morphological Constraint:
        # If BAH > 80 (densely packed informal settlements), max Albedo modification is 0.35
        # Otherwise, max Albedo is 0.65
        max_albedo = np.where(orig_bah > 80.0, 0.35, 0.65)
        
        # In case the original albedo is already higher than max_albedo, we cap the lower bound
        xl[self.N:] = np.clip(orig_albedo, 0.15, max_albedo)
        xu[self.N:] = max_albedo

        super().__init__(n_var=2*self.N, n_obj=1, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        new_ndvi = X[:, :self.N]
        new_albedo = X[:, self.N:]

        orig_ndvi = self.base_features[:, 0]
        orig_albedo = self.base_features[:, 1]

        # CONSTRAINT: Intervention Budget
        # We cap the total combined change to simulate a realistic municipal budget
        ndvi_change = new_ndvi - orig_ndvi
        albedo_change = new_albedo - orig_albedo
        total_change = np.sum(ndvi_change + albedo_change, axis=1)
        
        # Max average change of 0.20 per pixel allowed across the grid
        budget = self.N * 0.20 
        g1 = total_change - budget 

        # OBJECTIVE: Minimize predicted LST
        BAH = np.tile(self.base_features[:, 2], (pop_size, 1))
        TAH = np.tile(self.base_features[:, 3], (pop_size, 1))
        
        # If AOD is present (V2 model has 5 features)
        if self.base_features.shape[1] > 4:
            AOD = np.tile(self.base_features[:, 4], (pop_size, 1))
            features = np.stack([new_ndvi, new_albedo, BAH, TAH, AOD], axis=2)
            features_flat = features.reshape(-1, 5)
        else:
            features = np.stack([new_ndvi, new_albedo, BAH, TAH], axis=2)
            features_flat = features.reshape(-1, 4)

        # Scale and predict using the PINN
        features_scaled = (features_flat - self.scaler_mean) / self.scaler_scale
        features_t = torch.FloatTensor(features_scaled)

        with torch.no_grad():
            pred_lst_t = self.pinn_model(features_t)
            pred_lst = pred_lst_t.numpy().reshape(pop_size, self.N)

        # Minimize the mean temperature of the hotspot sample
        f1 = np.mean(pred_lst, axis=1)

        out["F"] = f1
        out["G"] = g1

def main():
    print("--- Starting Pillar 4: NSGA-II Scenario Optimization V2 ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Load Data
    data_path = os.path.join(base_dir, 'data', 'processed', 'delhi_thermal_features.csv')
    df = pd.read_csv(data_path)
    
    # Ensure AOD is mocked if it's missing, since V2 needs it
    if 'AOD' not in df.columns:
        print("AOD not found in data. Mocking AOD values for demonstration...")
        df['AOD'] = np.random.uniform(0.1, 0.8, size=len(df))
    
    # Load PINN Model V2 & Scalers
    model_path = os.path.join(base_dir, 'models', 'pinn_delhi_v2.pth')
    if not os.path.exists(model_path):
        print(f"Model V2 not found at {model_path}. Trying V1...")
        model_path = os.path.join(base_dir, 'models', 'pinn_delhi.pth')
        
    checkpoint = torch.load(model_path, weights_only=False)
    
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    feature_cols = checkpoint.get('feature_cols', ['NDVI', 'Albedo', 'BAH', 'TAH'])
    
    model = UrbanHeatPINN(input_dim=len(feature_cols))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("1. Isolating the top 100 extreme UHI hotspots for targeted intervention...")
    df_sorted = df.sort_values(by='LST_Celsius', ascending=False)
    hotspots = df_sorted.head(100).copy()
    
    base_features = hotspots[feature_cols].values

    # Calculate Baseline LST
    base_features_scaled = (base_features - scaler_mean) / scaler_scale
    with torch.no_grad():
        baseline_lst = model(torch.FloatTensor(base_features_scaled)).numpy()
    mean_baseline_lst = np.mean(baseline_lst)
    
    print(f"-> Baseline Average LST of Hotspots: {mean_baseline_lst:.2f} °C")

    print("2. Initializing NSGA-II Evolutionary Algorithm V2...")
    problem = CoolingOptimization(model, base_features, scaler_mean, scaler_scale)
    algorithm = NSGA2(pop_size=50)
    termination = get_termination("n_gen", 40)

    print("3. Hunting the Pareto Front for optimal LCZ interventions...")
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=42,
                   return_least_infeasible=True,
                   verbose=False)

    if res.F is None:
        print("Warning: No feasible solution found, returning least infeasible.")
        best_lst = mean_baseline_lst # Fallback
        optimal_X = np.zeros(200) # Fallback
    else:
        best_lst = res.F[0] if isinstance(res.F, (list, np.ndarray)) else res.F
        optimal_X = res.X[0] if isinstance(res.X[0], (list, np.ndarray)) else res.X

    temperature_drop = mean_baseline_lst - best_lst

    print("\n" + "="*50)
    print("🌍 OPTIMIZATION COMPLETE (PILLAR 4 V2)")
    print("="*50)
    print(f"Pre-Intervention Hotspot LST:  {mean_baseline_lst:.2f} °C")
    print(f"Post-Intervention Hotspot LST: {best_lst:.2f} °C")
    print(f"Total Cooling Achieved (ΔT):   {temperature_drop:.2f} °C")
    print("="*50)
    
    # Save the optimal intervention strategy
    hotspots['Optimized_NDVI'] = optimal_X[:100]
    hotspots['Optimized_Albedo'] = optimal_X[100:]
    hotspots['Delta_T'] = baseline_lst.flatten() - best_lst
    
    output_path = os.path.join(base_dir, 'data', 'processed', 'optimal_scenario_v2.csv')
    hotspots.to_csv(output_path, index=False)
    print(f"\nOptimal pixel interventions saved to {output_path}")

if __name__ == "__main__":
    main()
