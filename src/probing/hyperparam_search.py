import optuna
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from probes import MLPProbe, ProbeTrainer
from typing import Tuple
from functools import partial

def objective(trial, dataloaders: Tuple[DataLoader, DataLoader], num_epochs=30, task_type="regression"):
    sample = dataloaders[0].dataset[0]
    input_dim = sample['features'].shape[1]
    output_dim = sample['targets'].shape[1] if len(sample['targets'].shape) > 1 else 1
    
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_dims = []
    
    for i in range(n_layers):
        hidden_dim = trial.suggest_int(f'dim{i}_size', 16, 128)
        hidden_dims.append(hidden_dim)
    
    model = MLPProbe(input_dim, output_dim, hidden_dims, task_type=task_type)
    
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    best_val_loss = 0
    
    # TODO: train function goes here
    
    # trainer = ProbeTrainer(model)
    # early_stopping_patience = 15
    # best_val_loss = float("inf")
    # patience_counter = 0
    
    # for i in range(num_epochs):
    #     trainer.train_epoch(dataloaders[0], optimizer)
    #     val_metrics = trainer.evaluate(dataloaders[1])
    #     val_loss = val_metrics["loss"]
    
    #     # Early stopping
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1
    #     if patience_counter >= early_stopping_patience:
    #         break
        
    return best_val_loss
    
def hyperparam_search_mlp_probe(dataloaders: Tuple[DataLoader, DataLoader], n_trials=50, num_epochs=30, task_type="regression"):
    """
    Runs a hyperparameter search for an MLP probe

    Args:
        dataloaders (Tuple[DataLoader, DataLoader]): dataloaders[0] is the training set, dataloaders[1] is the validation set
        n_trials (int, optional) Number of hyperparameter search trials. Defaults to 50.
        num_epochs(int, optional) Number of epochs to train each model. Defaults to 30.
        task_type (str, optional) Defaults to "regression".

    Returns:
        The hyperparameters of the best model
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, dataloaders=dataloaders, task_type=task_type), n_trials=n_trials)
    
    return study.best_trial.params