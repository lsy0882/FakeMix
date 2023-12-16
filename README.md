# Multimodal DeepFake detection (MDFD)
<br>

## Guide
### 0. Base
* Code rules
  * Rule 1: Maintain the framework displayed in the Code Architecture Diagram.
  * Rule 2: Record various parameters and variables in the config.yaml file for each model.
  * Rule 3: Even if the same instances or methods are used in the <model_class_name>.py file within different model directories, implement each in their respective folders.
* Branch rules
  * Rule 1: Work on a branch created with the initials of your name, which will be called your own branch.
  * Rule 2: After completing the work, push to your own branch.
  * Rule 3: Once the push to your own branch is completed, make a pull request to the main branch.
  * Rule 4: When making a pull request, briefly write down the key points about what has been added/changed/deleted.
```shell
# Git clone / push / request / pull example 
(c.f. name: Lee Sangyoun -> branchname: lsy)

1. Clone
   cd <your_dir_path>
   git clone -b <branchname> https://github.com/lsy0882/MDFD.git
   git checkout <branchname>
2. Push
   (Update code)
   (if "git branch" == <branchname>: pass, else: git checkout <branchname>)
   git add .
   git commit -m "Update vX.X.X"
   git push origin <branchname>
3. Pull request
   1) Navigate to the repository page and click on the 'Pull requests' tab.
   2) Click the 'New Pull Request' button.
   3) Select 'main' as the 'base' branch and '<branchname>' as the 'compare' branch.
   4) Click the 'Create Pull Request' button.
   5) Briefly explain the key points about the added/modified/deleted sections.
4. Pull
   (Someone merge main branch)
   (if "git branch" == <branchname>: pass, else: git checkout <branchname>)
   git stash
   git pull origin main
   (Option)
     git stash pop
     If a conflict occurs, resolve the conflict in the affected parts, then re-stage the modified files and commit.
     Conflict example:
       <<<<<<< HEAD
       ... (current branch code)
       =======
       ... (stash code)
       >>>>>>> Stashed changes
     git add <filename>
     git commit -m "Resolved conflicts between stash and current branch"
   git push origin <branchname>
```
* Explanation of keys and values within config.yaml
```yaml
wandb:
  login: 
    key: "2e1b50dc43a56f36434c3853c7be5775a467ad72" ### Login key / Don't change
  init: ### Ref: https://docs.wandb.ai/ref/python/init
    project: "Multimodal_Deepfake_Detection" ### Dont't change
    entity: "leesy" ### Your wandb profile name (=id)
    save_code: true ### Don't change
    group: "" ### Don't change / Ref: https://docs.wandb.ai/guides/runs/grouping
    job_type: "training" ### "data-preprocessing", "training", "testing", etc...
    tags: ["Hw1p2Net", "Small"] ### [Network, Size, etc...]
    name: "Hw1p2Net_Small_v1.0.0" ### "Network"_"Size"_"Version" | Version policy: v{Architecture change}_{Method/Block change}_{Layer/Minor change}
    notes: "Testing wandb setting" ### Insert changes(plz write details)
    dir: "./wandb" ### Don't change
    resume: "auto" ### Don't change
    reinit: false ### Don't change
    magic: null ### Don't change
    config_exclude_keys: [] ### Don't change
    config_include_keys: [] ### Don't change
    anonymous: null ### Don't change
    mode: "online" ### Don't change
    allow_val_change: true ### Don't change
    force: false ### Don't change
    sync_tensorboard: false ### Don't change
    monitor_gym: false ### Don't change
    config: ### Record and use all parameters and variables
      dataset:
        name: "hw1p2"
        modality: "audio"
        data_path: "/home/lsy/hw1p2"
        context: 20
      dataloader:
        batch_size: 1024
        pin_memory: true
        num_workers: 0
      model:
        # input_size: 100
        # output_size: 100
        options:
          test_ignore_this_var: true
      criterion: ### Ref: https://pytorch.org/docs/stable/nn.html#loss-functions
        name: "CrossEntropyLoss" ### Choose a torch.nn's class(=attribute) e.g. ["CrossEntropyLoss", "MSELoss", "Custom", ...] / You can build your criterion :)
      optimizer: ### Ref: https://pytorch.org/docs/stable/optim.html#algorithms
        name: "Adamw" ### Choose a torch.optim's class(=attribute) e.g. ["Adam", "Adamw", "SGD", ...] / You can build your optimizer :)
        Adam: ### Add or modify instance & args using reference link
          lr: 1.0e-3
          weight_decay: 1.0e-2
        AdamW:
          lr: 1.0e-3
          weight_decay: 1.0e-2
        SGD:
          lr: 1.0e-3
          momentum: 0.9
          weight_decay: 1.0e-2
        Custom:
          custom_arg1:
          custom_arg2:
      scheduler: ### Ref(& find "How to adjust learning rate"): https://pytorch.org/docs/stable/optim.html#algorithms
        name: "StepLR" ### Choose a torch.optim.lr_scheduler's class(=attribute) e.g. ["StepLR", "ReduceLROnPlateau", "Custom"] / You can build your scheduler :)
        StepLR: ### Add or modify instance & args using reference link
          step_size: 5
          gamma: 0.9
        ReduceLROnPlateau:
          mode: "min"
          min_lr: 1.0e-10
          factor: 0.9
          patience: 5
        Custom:
          custom_arg1:
          custom_arg2:
      trainer:
        epoch: 100
        gpuid: 0 ### int or list ; 0(single-gpu) or [0, 1] (multi-gpu)
```
<br>

### 1. Conda Virtual Environment Setup
* It is assumed that git and conda installation and setup have been completed.
```shell
(Caution) Install the version of python & torch that matches your OS/GPU environment.
(Note) The version of torch used for the experiment is as follows.

cd <your_dir_path>/MDFD
conda create -n mdfd python=3.9
conda activate mdfd
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```
<br>

### 2. 학습데이터 준비 및 전처리(Preprocess)
* 데이터셋 Samples
  ![Samples](images/samples.png)
  
* 데이터셋 Tree 구조
