"""Hyperparameter optimization for baseline models.

Provides HPO functions to find optimal hyperparameters for:
- LSTM: hidden_dim, num_layers, dropout, learning_rate, batch_size
- Random Forest: n_estimators, max_depth, min_samples_split, max_features
- Fine-tuned Chronos: hidden_dim, dropout, learning_rate
"""

from __future__ import annotations

import json
import numpy as np
import optuna
from pathlib import Path
from loguru import logger

from src.benchmark.baseline_models import (
    LSTMPredictor,
    RandomForestRegressor_Wrapper,
    FineTunedChronosPredictor,
)


def run_lstm_hpo(
    close_windows_train: np.ndarray,
    targets_train: np.ndarray,
    close_windows_val: np.ndarray,
    targets_val: np.ndarray,
    target_h: int,
    n_trials: int = 30,
    device: str = "cpu",
) -> dict:
    """Optimize LSTM hyperparameters using Optuna.
    
    Searched hyperparameters:
        hidden_dim ∈ [32, 256] (horizon-dependent: 1D→shallow, 20D→deep)
        num_layers ∈ [1, 4]
        dropout ∈ [0.0, 0.5]
        learning_rate ∈ [1e-4, 1e-2] (log-uniform)
        batch_size ∈ [16, 64]
    
    Args:
        close_windows_train: (N_train, seq_len) training price windows
        targets_train: (N_train,) training targets
        close_windows_val: (N_val, seq_len) validation price windows
        targets_val: (N_val,) validation targets
        target_h: Prediction horizon (1, 5, or 20)
        n_trials: Number of Optuna trials
        device: Device to use ("cpu" or "cuda")
    
    Returns:
        dict: Best hyperparameters
    """
    logger.info("═══ LSTM HPO ({}D, {} trials) ═══", target_h, n_trials)
    
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
        
        try:
            # Create model
            model = LSTMPredictor(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device=device,
            )
            
            # Train with early stopping
            history = model.fit(
                close_windows_train, targets_train,
                close_windows_val, targets_val,
                epochs=100,
                batch_size=batch_size,
                learning_rate=lr,
                patience=10,
            )
            
            best_val_loss = history["best_val_loss"]
            logger.debug(
                "LSTM trial {}: hidden_dim={}, layers={}, dropout={:.2f}, lr={:.1e}, batch={} → loss={:.6f}",
                trial.number, hidden_dim, num_layers, dropout, lr, batch_size, best_val_loss
            )
            
            return best_val_loss
        
        except Exception as e:
            logger.warning("LSTM trial {} failed: {}", trial.number, str(e))
            return float("inf")
    
    # Run optimization
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    logger.info("Best LSTM params: {}", best_params)
    
    return best_params


def run_rf_hpo(
    close_windows_train: np.ndarray,
    targets_train: np.ndarray,
    close_windows_val: np.ndarray,
    targets_val: np.ndarray,
    target_h: int,
    n_trials: int = 20,
) -> dict:
    """Optimize Random Forest hyperparameters using Optuna.
    
    Searched hyperparameters:
        n_estimators ∈ [50, 300]
        max_depth ∈ [5, 30] (horizon-dependent: 1D→shallow, 20D→deep)
        min_samples_split ∈ [2, 20]
        max_features ∈ ["sqrt", "log2"]
    
    Args:
        close_windows_train: (N_train, seq_len) training price windows
        targets_train: (N_train,) training targets
        close_windows_val: (N_val, seq_len) validation price windows
        targets_val: (N_val,) validation targets
        target_h: Prediction horizon (1, 5, or 20)
        n_trials: Number of Optuna trials
    
    Returns:
        dict: Best hyperparameters
    """
    logger.info("═══ Random Forest HPO ({}D, {} trials) ═══", target_h, n_trials)
    
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        
        try:
            # Create and train model
            model = RandomForestRegressor_Wrapper(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
            )
            model.fit(close_windows_train, targets_train)
            
            # Evaluate on validation
            preds_val = model.predict(close_windows_val)
            mse = np.mean((preds_val - targets_val) ** 2)
            
            logger.debug(
                "RF trial {}: n_trees={}, depth={}, min_split={}, max_feat={} → MSE={:.6f}",
                trial.number, n_estimators, max_depth, min_samples_split, max_features, mse
            )
            
            return mse
        
        except Exception as e:
            logger.warning("RF trial {} failed: {}", trial.number, str(e))
            return float("inf")
    
    # Run optimization
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    logger.info("Best Random Forest params: {}", best_params)
    
    return best_params


def run_finetuned_chronos_hpo(
    chronos_predictor,
    close_windows_train: np.ndarray,
    targets_train: np.ndarray,
    close_windows_val: np.ndarray,
    targets_val: np.ndarray,
    target_h: int,
    n_trials: int = 15,
    device: str = "cpu",
) -> dict:
    """Optimize Fine-tuned Chronos hyperparameters using Optuna.
    
    Searched hyperparameters:
        hidden_dim ∈ [64, 256]
        dropout ∈ [0.1, 0.5]
        learning_rate ∈ [1e-5, 1e-3] (log-uniform, low for fine-tuning)
    
    Args:
        chronos_predictor: ChronosMarketPredictor instance
        close_windows_train: (N_train, seq_len) training price windows
        targets_train: (N_train,) training targets
        close_windows_val: (N_val, seq_len) validation price windows
        targets_val: (N_val,) validation targets
        target_h: Prediction horizon (1, 5, or 20)
        n_trials: Number of Optuna trials
        device: Device to use
    
    Returns:
        dict: Best hyperparameters
    """
    logger.info("═══ Fine-tuned Chronos HPO ({}D, {} trials) ═══", target_h, n_trials)
    
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=32)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        
        try:
            # Create model
            model = FineTunedChronosPredictor(
                chronos_predictor,
                hidden_dim=hidden_dim,
                dropout=dropout,
                device=device,
            )
            
            # Train with early stopping
            history = model.fit(
                close_windows_train, targets_train,
                close_windows_val, targets_val,
                epochs=25,
                batch_size=32,
                learning_rate=lr,
                patience=5,
            )
            
            best_val_loss = history["best_val_loss"]
            logger.debug(
                "Fine-tuned Chronos trial {}: hidden={}, dropout={:.2f}, lr={:.1e} → loss={:.6f}",
                trial.number, hidden_dim, dropout, lr, best_val_loss
            )
            
            return best_val_loss
        
        except Exception as e:
            logger.warning("Fine-tuned Chronos trial {} failed: {}", trial.number, str(e))
            return float("inf")
    
    # Run optimization
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    logger.info("Best Fine-tuned Chronos params: {}", best_params)
    
    return best_params


def load_or_run_baseline_hpo(
    hpo_dir: Path,
    close_windows_train: np.ndarray,
    targets_train: np.ndarray,
    close_windows_val: np.ndarray,
    targets_val: np.ndarray,
    chronos_predictor,
    target_h: int,
    device: str = "cpu",
    force_rerun: bool = False,
) -> dict:
    """Load cached HPO params or run HPO if not cached.
    
    Args:
        hpo_dir: Directory to cache HPO results
        close_windows_train: Training data
        targets_train: Training targets
        close_windows_val: Validation data
        targets_val: Validation targets
        chronos_predictor: ChronosMarketPredictor for fine-tuned version
        target_h: Prediction horizon
        device: Device for training
        force_rerun: Force re-running HPO even if cached
    
    Returns:
        dict: {
            "lstm": {...},
            "rf": {...},
            "finetuned_chronos": {...}
        }
    """
    hpo_dir.mkdir(parents=True, exist_ok=True)
    cache_file = hpo_dir / f"best_baseline_params_{target_h}d.json"
    
    # Try to load from cache
    if cache_file.exists() and not force_rerun:
        logger.info("Loading cached baseline HPO params ({}D) from {}", target_h, cache_file)
        with open(cache_file) as f:
            return json.load(f)
    
    logger.info("Running baseline HPO for {}D", target_h)
    
    # Run LSTM HPO
    lstm_params = run_lstm_hpo(
        close_windows_train, targets_train,
        close_windows_val, targets_val,
        target_h=target_h,
        n_trials=30,
        device=device,
    )
    
    # Run Random Forest HPO
    rf_params = run_rf_hpo(
        close_windows_train, targets_train,
        close_windows_val, targets_val,
        target_h=target_h,
        n_trials=20,
    )
    
    # Run Fine-tuned Chronos HPO
    chronos_ft_params = run_finetuned_chronos_hpo(
        chronos_predictor,
        close_windows_train, targets_train,
        close_windows_val, targets_val,
        target_h=target_h,
        n_trials=15,
        device=device,
    )
    
    # Save all params
    all_params = {
        "lstm": lstm_params,
        "rf": rf_params,
        "finetuned_chronos": chronos_ft_params,
    }
    
    with open(cache_file, "w") as f:
        json.dump(all_params, f, indent=2)
    
    logger.info("Baseline HPO complete. Cached to {}", cache_file)
    
    return all_params
