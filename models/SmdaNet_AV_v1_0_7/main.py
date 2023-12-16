import os
from .dataloader import get_dataloaders
from .model import SmdaNet
from .trainer import Trainer
from .tester import Tester
from utils import system_util, implement_util

# Call logger for monitoring (in terminal)
logger = system_util.get_logger(__name__)

def main(args):
    ''' Build Setting '''
    # Call root setting file
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")
    yaml_dict = system_util.parse_yaml(yaml_path)
    
    # Call wandb and configuration
    wandb_run = system_util.wandb_setup(yaml_dict)
    config = wandb_run.config
    
    # Call DataLoader [train / valid / test / etc...]
    datasets, dataloaders = get_dataloaders(args, config["dataset"], config["dataloader"])

    ''' Build Model '''
    # Call network model
    model = SmdaNet(num_classes=config["model"]["num_classes"])
    system_util.log_model_information_to_wandb(wandb_run, model, os.path.dirname(os.path.abspath(__file__))) # Record artifact on wandb for logging model architecture

    ''' Build Engine '''    
    # Call Implement [criterion / optimizer / scheduler]
    criterion = implement_util.get_criterion(config["criterion"])
    optimizer = implement_util.get_optimizer(config["optimizer"], model.parameters())
    scheduler = implement_util.get_scheduler(config["scheduler"], optimizer)
    
    # Call & Run Engine
    gpuid = tuple(map(int, config["engine"]["gpuid"].split(',')))
    if args.mode == "train":
        engine = Trainer(config, model, datasets, dataloaders, criterion, optimizer, scheduler, gpuid, wandb_run)
    elif args.mode == "test":
        engine = Tester(config, model, datasets, dataloaders, criterion, optimizer, scheduler, gpuid, wandb_run)

    engine.run()