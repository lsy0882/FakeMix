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
