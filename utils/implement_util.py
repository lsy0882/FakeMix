import torch
from utils import system_util

# Call logger for monitoring (in terminal)
logger = system_util.get_logger(__name__)

def get_distance(config):
    try:
        if config["name"] == "custom":
            print("Insert your custom distance :)")
            return None # Change "None" to your distance instance
        else:
            distance_class = getattr(torch.nn, config["name"])
            distance_args = config.get(config["name"], {})
            return distance_class(**distance_args)
    except AttributeError:
        raise NotImplementedError(f"Distance {config['name']} not implemented!")

def get_criterion(config):
    """
    Returns a loss function (criterion) based on the provided configuration.

    This function reads the specified criterion name from the given configuration.
    If the criterion name is "custom", the function prompts the user to insert their custom loss function.
    Otherwise, it fetches the corresponding criterion class from PyTorch's `torch.nn` module and returns its instance.

    Args:
        config (dict): Dictionary containing the criterion configuration. Expected to have a "name" key which 
            specifies the name of the desired criterion.

    Returns:
        torch.nn.Module: An instance of the specified loss function.

    Raises:
        NotImplementedError: If the desired criterion is not found in PyTorch's `torch.nn` module.

    Example:
        >>> criterion_config = {"name": "CrossEntropyLoss"}
        >>> criterion = get_criterion(criterion_config)
    """
    try:
        if config["name"] == "custom":
            print("Insert your custom loss function :)")
            return None # Change "None" to your criterion instance
        else:
            criterion_class = getattr(torch.nn, config["name"])
            return criterion_class()
    except AttributeError:
        raise NotImplementedError(f"Criterion {config['name']} not implemented!")

def get_optimizer(config, parameters_policy):
    """
    Returns an optimizer based on the provided configuration and parameters.

    This function reads the specified optimizer name from the given configuration.
    If the optimizer name is "custom", the function prompts the user to insert their custom optimizer.
    Otherwise, it fetches the corresponding optimizer class from PyTorch's `torch.optim` module, 
    configures it with the provided parameters and any additional arguments specified in the config, 
    and returns its instance.

    Args:
        config (dict): Dictionary containing the optimizer configuration. Expected to have a "name" key which 
            specifies the name of the desired optimizer. Additional keys can be provided to specify optimizer arguments.
        parameters_policy (iterable): Parameters of the model to be optimized.

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer.

    Raises:
        NotImplementedError: If the desired optimizer is not found in PyTorch's `torch.optim` module.

    Example:
        >>> optimizer_config = {"name": "Adam", "Adam": {"lr": 0.001}}
        >>> parameters = model.parameters()
        >>> optimizer = get_optimizer(optimizer_config, parameters)
    """
    try:
        if config["name"] == "custom":
            print("Insert your custom optimizer :)")
            return None # Change "None" to your optimizer instance
        else:
            optimizer_class = getattr(torch.optim, config["name"])
            optimizer_args = config.get(config["name"], {})
            return optimizer_class(parameters_policy, **optimizer_args)
    except AttributeError:
        raise NotImplementedError(f"Optimizer {config['name']} not implemented!")

def get_scheduler(config, optimizer):
    """
    Returns a learning rate scheduler based on the provided configuration and optimizer.

    This function reads the specified scheduler name from the given configuration.
    If the scheduler name is "custom", the function prompts the user to insert their custom scheduler.
    Otherwise, it fetches the corresponding scheduler class from PyTorch's `torch.optim.lr_scheduler` module, 
    configures it with the provided optimizer and any additional arguments specified in the config, 
    and returns its instance.

    Args:
        config (dict): Dictionary containing the scheduler configuration. Expected to have a "name" key which 
            specifies the name of the desired scheduler. Additional keys can be provided to specify scheduler arguments.
        optimizer (torch.optim.Optimizer): An instance of the optimizer for which the scheduler is being fetched.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: An instance of the specified learning rate scheduler.

    Raises:
        NotImplementedError: If the desired scheduler is not found in PyTorch's `torch.optim.lr_scheduler` module.

    Example:
        >>> scheduler_config = {"name": "StepLR", "StepLR": {"step_size": 10, "gamma": 0.5}}
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = get_scheduler(scheduler_config, optimizer)
    """
    try:
        if config["name"] == "custom":
            print("Insert your custom scheduler :)")
            return None # Change "None" to your scheduler instance
        else:
            scheduler_class = getattr(torch.optim.lr_scheduler, config["name"])
            scheduler_args = config.get(config["name"], {})
            return scheduler_class(optimizer, **scheduler_args, verbose=True)
    except AttributeError:
        raise NotImplementedError(f"Scheduler {config['name']} not implemented!")
    
    
'''
    <<<  scheduler build example  >>>
    
    class WarmupConstantSchedule(th.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to "init_lr" over `warmup_steps` training steps.
        Keeps learning rate schedule equal to "init_lr". after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            return 1.0

        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)
'''