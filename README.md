<<<<<<< Updated upstream
# FakeMix

## Guide
### 0. FakeMix benchmark download link
https://www.dropbox.com/scl/fi/nsf5dphlb1te6m91j8kn2/data.zip?rlkey=xacwyculfvvussat7p7cl45v5&st=k6tbyti1&dl=0
<br>

### 1. Instructions for Preparing FakeMix Benchmark Data <br>(Optional; You can easily download the dataset using the link provided above.)

#### Step 1: Download FakeAVCeleb
First, download the FakeAVCeleb dataset.

#### Step 2: Data Preprocessing
1. Open `.../FakeMix/preprocessor/data_preprocess.py`.
2. Set the `data_directory` variable to the root directory of FakeAVCeleb.
3. Set the `output_directory` variable to the path where you want to save the preprocessed data.
4. After modifying the variables, run the following command:
   ```bash
   python .../FakeMix/preprocessor/data_preprocess.py
   ```

#### Step 3: Mixing Clips
1. Open `.../FakeMix/preprocessor/mix_clips.py`.
2. Set the `dataset_base_dir` variable to the same path as the `output_directory` from the previous step.
3. Set the `output_base_dir` variable to the path where you want to save the mixed data.
4. After modifying the variables, run the following command (ensure not to change the seed value for `random.seed()`):
   ```bash
   python .../FakeMix/preprocessor/mix_clips.py
   ```

#### Step 4: Understanding the FakeMix Benchmark Data
The mixed data obtained using `.../FakeMix/preprocessor/mix_clips.py` is the FakeMix benchmark data. It is organized as follows:

- **FakeVideo-FakeAudio**: Directory containing videos where each video has 1-second clips of FakeVideo-FakeAudio or RealVideo-RealAudio appearing in random order.
- **FakeVideo-RealAudio**: Directory containing videos where each video has 1-second clips of FakeVideo-RealAudio or RealVideo-RealAudio appearing in random order.
- **RealVideo-FakeAudio**: Directory containing videos where each video has 1-second clips of RealVideo-FakeAudio or RealVideo-RealAudio appearing in random order.

#### Annotations
For each created video (e.g., `abc.mp4`) or audio file (e.g., `abc.wav`), there is an annotation file with the same name saved in JSON format containing frame-by-frame label information (e.g., `abc.json`).
=======
# Multimodal DeepFake detection (MDFD)
<br>

## Guide
### 0. Base
* Code rules
  * Code architecture diagram ![Code Architecture Diagram](https://github.com/lsy0882/MDFD/releases/download/0.0.1/Code_Architecture_Diagram.png)
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
>>>>>>> Stashed changes
<br>

### 2. Testing Baseline_1_Unimodal_Video_Xception and Baseline_1_Unimodal_Audio_Xception

First, ensure you have a [wandb](https://www.wandb.com/) account as this experiment logs results using wandb.

#### Steps 1: Set Up Wandb Account
Create a wandb account and get your API key.

#### Steps 2: Configure the Experiment
- Open the `.../FakeMix/run.sh` file.
- Set the `--model` parameter to the directory name in `.../FakeMix/models` where you want to run the experiment. For example:
  ```bash
  --model Baseline_1_Unimodal_Audio_Xception
  ```
- Set the `--mode` parameter to `test`. For example:
  ```bash
  --mode test
  ```
- If you want to use FakeMix for training, set the `--mode` parameter to `train`.

#### Steps 3: Update Configurations
- In the experiment directory, open the `configs.yaml` file. For example:
  ```bash
  .../FakeMix/models/Baseline_1_Unimodal_Audio_Xception/configs.yaml
  ```
- Update the `wandb-login-key` value with your wandb login API key.
- Update the `wandb-init-entity` value with your wandb profile name or team name.
- Set the `wandb-init-config-dataset-data_path` value to the root path of the FakeMix dataset. For example:
  ```yaml
  data_path: "/home/lsy/laboratory/Research/FakeMix/data"
  ```
- Set the `wandb-init-config-engine-gpuid` value to the GPU ID you want to use for the experiment. For example, to use GPU 0:
  ```yaml
  gpuid: "0"
  ```

#### Steps 4: Run the Experiment
- Execute the experiment by running:
  ```bash
  sh .../FakeMix/run.sh
  ```

#### Steps 5. Check Results
- After the test is completed, the output will display `Average each accuracy` and `Total accuracy`. These values represent the TA and FDM metrics, respectively.
- Each experiment directory will also have an `each_file_record.xlsx` file, which shows the accuracy for each video clip, indicating which clips were correctly classified.

By following these steps, you can test the Baseline_1_Unimodal_Video_Xception and Baseline_1_Unimodal_Audio_Xception models and analyze the results using wandb and the provided Excel files.
<br>

### 3. Testing Baseline_2_Multimodal_AVAD

#### Step 1: Download AVAD Checkpoint
- Download Audio-visual synchronization model checkpoint `sync_model.pth`[link](https://drive.google.com/file/d/1BxaPiZmpiOJDsbbq8ZIDHJU7--RJE7Br/view?usp=sharing) at .../FakeMix/models/Baseline_2_Multimodal_AVAD/ 

#### Step 2: Generate Test Data Paths File
- Open `.../FakeMix/models/Baseline_2_Multimodal_AVAD/make_data_path_to_textfile.py`.
- Set the `root_directory` variable to the path of the FakeMix test directory.
- Set the `output_file` variable to the desired path where you want to save the `.txt` file.
- Run the following command to create a `.txt` file containing paths to the test data:
  ```bash
  python .../FakeMix/models/Baseline_2_Multimodal_AVAD/make_data_path_to_textfile.py
  ```

#### Step 3: Run Detection
- Navigate to the model directory:
  ```bash
  cd .../FakeMix/models/Baseline_2_Multimodal_AVAD
  ```
- Run the detection command:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python detect.py --test_video_path "/home/lsy/laboratory/Research/FakeMix/models/Baseline_2_Multimodal_AVAD/tools_for_FakeMix/FakeMIx_mp4_paths.txt" --device cuda:0 --max-len 50 --n_workers 18 --bs 1 --lam 0 --output_dir /home/lsy/laboratory/Research/FakeMix/models/Baseline_2_Multimodal_AVAD
  ```
- Set the command arguments as follows:
  - `CUDA_VISIBLE_DEVICES=n`: Set `n` to the GPU ID you want to use.
  - `test_video_path`: Path to the `.txt` file created in the previous step.
  - `device`: Set to the same value as `CUDA_VISIBLE_DEVICES`.
  - `max-len`: Maximum sequence length considered by AVAD. Set to 50 for this experiment.
  - `n_workers`: Number of CPU workers for the dataloader. Adjust based on your CPU specifications.
  - `bs`: Batch size. Set to 1 for testing.
  - `lam`: Fixed value.
  - `output_dir`: Path where the output results will be saved (e.g., `.../FakeMix/models/Baseline_2_Multimodal_AVAD`).

#### Step 4: Evaluate Results
- After the process completes, a `testing_scores.json` file will be created in `.../FakeMix/models/Baseline_2_Multimodal_AVAD`. This JSON file contains evaluation results for each test video and audio clip, including probability scores for detecting deepfakes per second.
- Open `.../FakeMix/models/Baseline_2_Multimodal_AVAD/calculate_our_metric.py`.
- Set the `file_path` variable to the path of the `testing_scores.json` file.
- Run the following command to calculate the metrics:
  ```bash
  python .../FakeMix/models/Baseline_2_Multimodal_AVAD/calculate_our_metric.py
  ```
- Once executed, you will obtain the TA and FDM evaluation results.
- Additionally, in the directory specified by the `file_path` variable, you will find neatly recorded JSON files detailing the evaluation results for each video clip.

By following these steps, you can effectively test the Baseline_2_Multimodal_AVAD model and analyze the results.


## Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@inproceedings{TBD,
  title={WWW: Where, Which and Whatever Enhancing Interpretability in Multimodal Deepfake Detection},
  author={TBD},
  booktitle={TBD,
  year={2024},
  organization={IJCAI 2024 Workshop}
}
