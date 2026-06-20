import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def calculate_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

def main():
    print("--- Starting LightGBM Training and SHAP Analysis ---")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    
    # Create output directory for figures if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)
    
    data_path = os.path.join(processed_dir, 'delhi_thermal_features.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run build_features.py first.")
        
    print("1. Loading processed data...")
    df = pd.read_csv(data_path)
    
    # We predict LST based on engineered features. 
    target_col = 'LST_Celsius'
    feature_cols = ['NDVI', 'Albedo', 'BAH', 'TAH', 'IAH', 'MAH']
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    
    print("2. Splitting data into train and test sets...")
    # Using an 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("3. Training LightGBM Regressor...")
    # Using default hyperparameters for the baseline model
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("4. Evaluating Model...")
    y_pred = model.predict(X_test)
    
    # Calculate exact metrics from literature
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_r, _ = pearsonr(y_test, y_pred)
    bias = calculate_bias(y_test, y_pred)
    
    print("--- Model Performance Metrics ---")
    print(f"R² (Coefficient of Determination): {r2:.4f}")
    print(f"RMSE (Root Mean Square Error):     {rmse:.4f} °C")
    print(f"Pearson's r:                       {pearson_r:.4f}")
    print(f"Bias:                              {bias:.4f} °C")
    
    print("5. Generating SHAP Explanations...")
    # We sample the test set for SHAP to avoid extreme memory/time usage on >1.5M pixels
    shap_sample = X_test.sample(n=min(10000, len(X_test)), random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(shap_sample)
    
    # 5a. Global Importance (Bar Chart)
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    global_importance_path = os.path.join(figures_dir, 'shap_global_importance.png')
    plt.savefig(global_importance_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Global Importance (Bar Chart) -> {global_importance_path}")
    
    # 5b. Local Importance (Beeswarm Plot)
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    local_importance_path = os.path.join(figures_dir, 'shap_local_importance.png')
    plt.savefig(local_importance_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Local Importance (Beeswarm)   -> {local_importance_path}")
    
    print("--- Day 2 LightGBM & SHAP Analysis Complete! ---")

if __name__ == "__main__":
    main()
