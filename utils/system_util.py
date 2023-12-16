import os
import yaml
import logging
import wandb


def parse_yaml(path):
    """
    Parse and return the contents of a YAML file.

    Args:
        path (str): Path to the YAML file to be parsed.

    Returns:
        dict: A dictionary containing the parsed contents of the YAML file.

    Raises:
        FileNotFoundError: If the provided path does not point to an existing file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Could not find configure files...{}".format(path))
    with open(path, 'r') as f:
        config_dict = yaml.full_load(f)
    return config_dict

def get_logger(name, format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s"):
    """
    Configures and returns a logger instance with the given name and format string.

    This function sets up a logger with a stream handler that outputs to the console.
    The logger is set to the INFO level by default.

    Args:
        name (str): Name of the logger.
        format_str (str, optional): The format string for the log messages. 
            Defaults to "%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s".

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = get_logger("MyLogger")
        >>> logger.info("This is an info message.")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def wandb_setup(yaml_dict):
    """
    Configures and initializes a Weights & Biases (wandb) run using the parameters provided in the YAML configuration.

    This function sets up logging, tracking, and visualization capabilities for a machine learning experiment
    using wandb. The function reads the wandb-specific configuration from the provided YAML dictionary and
    initializes a new wandb run with those settings.

    Args:
        yaml_dict (dict): Dictionary containing the wandb configuration loaded from a YAML file.

    Returns:
        wandb.wandb_run.Run: An instance of a wandb run, which provides methods and properties to 
            interact with the experiment being tracked.

    Example:
        >>> config = parse_yaml("config.yaml")
        >>> run = wandb_setup(config)
    """
    wandb.login(key=yaml_dict['wandb']['login']['key'])
    # Create your wandb run
    run = wandb.init(
        entity                 = yaml_dict['wandb']['init']['entity'],
        project                = yaml_dict['wandb']['init']['project'],
        save_code              = yaml_dict['wandb']['init']['save_code'],
        # group                  = yaml_dict['wandb']['init']['group'],
        job_type               = yaml_dict['wandb']['init']['job_type'],
        tags                   = yaml_dict['wandb']['init']['tags'],
        name                   = yaml_dict['wandb']['init']['name'],
        notes                  = yaml_dict['wandb']['init']['notes'],
        dir                    = yaml_dict['wandb']['init']['dir'],
        resume                 = yaml_dict['wandb']['init']['resume'],
        reinit                 = yaml_dict['wandb']['init']['reinit'],
        magic                  = yaml_dict['wandb']['init']['magic'],
        # config_exclude_keys    = yaml_dict['wandb']['init']['config_exclude_keys'],
        # config_include_keys    = yaml_dict['wandb']['init']['config_include_keys'],
        anonymous              = yaml_dict['wandb']['init']['anonymous'],
        mode                   = yaml_dict['wandb']['init']['mode'],
        allow_val_change       = yaml_dict['wandb']['init']['allow_val_change'],
        force                  = yaml_dict['wandb']['init']['force'],
        sync_tensorboard       = yaml_dict['wandb']['init']['sync_tensorboard'],
        monitor_gym            = yaml_dict['wandb']['init']['monitor_gym'],
        id                     = yaml_dict['wandb']['init']['name'], ### same with "name"
        config                 = yaml_dict['wandb']['init']['config']
    )
    return run

def log_model_information_to_wandb(wandb_run, model, root_path):
    """
    Logs the architecture of the given model to Weights & Biases (wandb) as an artifact.

    This function creates a new wandb artifact of type "model" to store the architecture
    of the provided PyTorch model. The architecture details, including the names and types of all modules
    in the model, are written to a file named "model_arch.txt". This file is then attached to the created artifact.
    Finally, the artifact is logged to the provided wandb run.

    Args:
        wandb_run (wandb.wandb_run.Run): The current wandb run to which the artifact should be logged.
        model (torch.nn.Module): The PyTorch model whose architecture should be logged.

    Example:
        >>> wandb_run = wandb.init()
        >>> model = SomePyTorchModel()
        >>> log_model_architecture_to_wandb(wandb_run, model)
    """
    # Ref: https://docs.wandb.ai/ref/python/artifact
    artifact_arch = wandb.Artifact(
            "model-architecture",
            type="model",
            description="Architecture of the trained model",
            metadata={"framework": "pytorch"}
        )    
    with artifact_arch.new_file("model_arch.txt", mode="w") as f:
        for name, module in model.named_modules():
            f.write(f"{name}: {type(module).__name__}\n")
    wandb_run.log_artifact(artifact_arch)
    
    artifact_dirfiles = wandb.Artifact(
        "my_directory_files", 
        type="data",
        description="All files and subdirectories from the specified directory"
        ) 
    def add_files_to_artifact(dir_path, artifact):
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                artifact.add_file(item_path, name=item_path[len(dir_path)+1:])
            elif os.path.isdir(item_path):
                add_files_to_artifact(item_path, artifact)
    add_files_to_artifact(root_path, artifact_dirfiles)
    wandb_run.log_artifact(artifact_dirfiles)