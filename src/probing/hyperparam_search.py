import optuna
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from probes import MLPProbe, ProbeTrainer
from typing import Tuple
from functools import partial

def objective(
    trial, 
    dataloaders: Tuple[DataLoader, DataLoader],
    num_epochs=30, 
    early_stopping_patience=15,
    use_scheduler=False,
    task_type="regression", 
    probe_type: str = None, 
    layer: int = None,
    wandb_enabled: bool = False
) -> float:
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
    
    scheduler = None
    
    if use_scheduler:
        t_max = trial.suggest_int("t_max", 10, num_epochs)
        scheduler = CosineAnnealingLR(optimizer, t_max=t_max)
    
    trainer = ProbeTrainer(model)
    _, best_val_loss = trainer.train(
        num_epochs, 
        optimizer, 
        scheduler, 
        early_stopping_patience, 
        dataloaders[0], 
        dataloaders[1], 
        probe_type, 
        layer, 
        wandb_enabled)
        
    return best_val_loss
    
def hyperparam_search_mlp_probe(
    dataloaders: Tuple[DataLoader, DataLoader], 
    n_trials=50, 
    num_epochs=30,
    early_stopping_patience=15,
    use_scheduler=False,
    task_type="regression",
    probe_type: str = None, 
    layer: int = None,
    wandb_enabled: bool = False):
    """
    Runs a hyperparameter search for an MLP probe

    Args:
        dataloaders (Tuple[DataLoader, DataLoader]): dataloaders[0] is the training set, dataloaders[1] is the validation set
        n_trials (int, optional) Number of hyperparameter search trials. Defaults to 50.
        Refer to ProbeTrainer.train for remaining args

    Returns:
        The hyperparameters of the best model
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        partial(
            objective, 
            dataloaders=dataloaders,
            num_epochs=num_epochs, 
            early_stopping_patience=early_stopping_patience,
            use_scheduler=use_scheduler,
            task_type=task_type,
            probe_type=probe_type,
            layer=layer,
            wandb_enabled=wandb_enabled), 
        n_trials=n_trials)
    
    return study.best_trial.params