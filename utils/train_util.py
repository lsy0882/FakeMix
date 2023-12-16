import os
import torch
from utils import system_util

# Call logger for monitoring (in terminal)
logger = system_util.get_logger(__name__)

def load_latent_checkpoint(checkpoint_dir, model, optimizer, location='cpu'):
    """
    Load the latest checkpoint (model state and optimizer state) from a given directory.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        model (torch.nn.Module): The model into which the checkpoint's model state should be loaded.
        optimizer (torch.optim.Optimizer): The optimizer into which the checkpoint's optimizer state should be loaded.
        location (str, optional): Device location for loading the checkpoint. Defaults to 'cpu'.

    Returns:
        int: The epoch number associated with the loaded checkpoint. 
             If no checkpoint is found, returns 0 as the starting epoch.

    Notes:
        - The checkpoint file is expected to have keys: 'model_state_dict', 'optimizer_state_dict', and 'epoch'.
        - If there are multiple checkpoint files in the directory, the one with the highest epoch number is loaded.
    """
    # List all .pkl files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir)]

    # If there are no checkpoint files, return 0 as the starting epoch
    if not checkpoint_files:
        return 0
    else:
        # Extract the epoch numbers from the file names and find the latest (max)
        epochs = [int(f.split('.')[1]) for f in checkpoint_files]
        latest_checkpoint_file = os.path.join(checkpoint_dir, checkpoint_files[epochs.index(max(epochs))])

        # Load the checkpoint into the model & optimizer
        logger.info(f"Loaded Pretrained model from {latest_checkpoint_file} .....")
        checkpoint_dict = torch.load(latest_checkpoint_file, map_location=location)
        model.load_state_dict(checkpoint_dict.get('model_state_dict') or checkpoint_dict.get('state_dict')) # Depend on weight file's key!!
        for param_group in optimizer.param_groups:
            try:
                param_group['lr'] = checkpoint_dict['optimizer_state_dict']['param_groups'][0]['lr']
            except:
                logger.info("\t There are no optimizer_state_dict key")
        
        # Retrun latent epoch
        return checkpoint_dict['epoch']
    
def save_checkpoint_per_nth(nth, epoch, model, optimizer, checkpoint_path, wandb_run):
    """
    Save the state of the model and optimizer every nth epoch to a checkpoint file.
    Additionally, log and save the checkpoint file using wandb.

    Args:
        nth (int): Interval for which checkpoints should be saved.
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        wandb_run (wandb.wandb_run.Run): The current wandb run to log and save the checkpoint.

    Returns:
        None
    """
    if epoch % nth == 0:
        # Save the state of the model and optimizer to a checkpoint file
        torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    },
                    os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        
        # Log and save the checkpoint file using wandb
        wandb_run.save(os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
    
def step_scheduler(scheduler, **kwargs):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(kwargs.get('val_loss'))
    elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler.step()
    elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler.step()
    # Add another schedulers
    else:
        raise ValueError(f"Unknown scheduler type: {type(scheduler)}")

def print_parameters_count(model):
    total_parameters = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_parameters += param_count
        print(f"{name}: {param_count}")
    print(f"Total parameters: {total_parameters / 1e6}M")