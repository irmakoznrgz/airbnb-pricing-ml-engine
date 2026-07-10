import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import load_raw_data, save_processed_data
from preprocess import run_preprocessing_pipeline

def main():
    raw_data_path = "data/raw/listings.csv"
    df = load_raw_data(raw_data_path)
    
    df_clean = run_preprocessing_pipeline(df)
    df_clean.columns = df_clean.columns.str.replace(r'\W+', '_', regex=True)
    processed_data_path = "data/processed/cleaned_listings.csv"
    save_processed_data(df_clean, processed_data_path)
  
    X = df_clean.drop(columns=['price'])
    y = df_clean['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training Set Size: {X_train.shape}")
    print(f"Test Set Size: {X_test.shape}")
    
    models = {
        "Ridge Regression (Baseline)": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "LGBMR Regressor": LGBMRegressor(
        n_estimators=500,learning_rate=0.1,
        random_state=42, verbosity=-1)
    }
    
    print("\n--- Model Performance Results ---")
    best_model_name = ""
    best_rmse = float('inf')

    for name, model in models.items():

        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"RMSE (Average Deviation): {rmse:.2f} ")
        print(f"R²: {r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse 
            best_model_name = name

        print("\n" + "="*40)
    print(f"WINNING MODEL: {best_model_name}")
    print(f"LOWEST ERROR (RMSE): {best_rmse:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()