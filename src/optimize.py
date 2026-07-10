import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold, cross_val_score
from lightgbm import LGBMRegressor
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_hyperparameters(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),

        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),

        'num_leaves': trial.suggest_int('num_leaves', 20, 150),

        'max_depth': trial.suggest_int('max_depth', 3, 15),

        'subsample': trial.suggest_float('subsample', 0.5, 1.0),

        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),

        'verbosity': -1,

        'random_state': 42
    }
    
    model = LGBMRegressor(**params)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    
    return -scores.mean()

def main():
    processed_data_path = "data/processed/cleaned_listings.csv"
    df = pd.read_csv(processed_data_path)
    
    X = df.drop(columns=['price'])
    y = df['price']
    
    study = optuna.create_study(direction="minimize")
    
    study.optimize(lambda trial: optimize_hyperparameters(trial, X, y), n_trials=20)
    
    print("\n--- Optimization Complete ---")
    print("Lowest RMSE:", study.best_value)
    print("Best Parameters:", study.best_params)
    
    print("\nBest model is being trained and saved...")
    best_params = study.best_params
    best_params['verbosity'] = -1
    best_params['random_state'] = 42
    
    final_model = LGBMRegressor(**best_params)
    final_model.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, 'models/best_lgbm_model.pkl')
    print("Model successfully saved to 'models/best_lgbm_model.pkl'!")

if __name__ == "__main__":
    main()