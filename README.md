# Efficient Multi-Person Motion Prediction by Lightweight Spatial and Temporal Interactions (ICCV 2025)

**Paper:** [arxiv]() (Coming Soon)

## Quick Start

### Environment Setup
- Python 3.8
```bash
pip install -r requirements.txt
```
We recommend using VS Code. If using other IDEs like PyCharm, you may encounter path-related issues.

### Hardware Requirements
We use a single NVIDIA 3090 GPU. To fully reproduce our results, we recommend using the same GPU.

### Path Configuration
1. Ensure your current working directory is **EMPMP**. If different, modify the `C.repo_name` variable in `src/baseline_3dpw/config.py` to match your working directory name. **Note: All folders starting with "baseline" and all folders starting with "models" have almost identical code formats and are relatively independent, so you also need to modify config.py in other baseline folders.**
2. Set `PYTHONPATH` to your working directory in `.vscode/settings.json`.

## Data Preparation

### Download Dataset Files

#### Dataset Files from GitHub Release
Download the following dataset files from our [GitHub Releases](../../releases):
- `mupots_120_3persons.npy`
- `somof_test.pt` 
- `test_3_120_mocap.npy`
- `train_3_120_mocap.npy`

Place these files directly in the `data/` directory.

#### Pretrained Model Files from GitHub Release
Download the following pretrained model files from our [GitHub Releases](../../releases):
- `pt_norc.pth`
- `pt_rc.pth`

Place these files in the `pt_ckpts/` directory.

#### 3DPW Dataset Files
Download the 3DPW dataset from the official [website](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

After downloading, extract and place the files in the `data/3dpw/` directory structure as shown below.

### Data Structure (@/data)


```
data/
├── mupots_120_3persons.npy          
├── somof_test.pt   #use training set for test                 
├── test_3_120_mocap.npy            
├── train_3_120_mocap.npy           
└── 3dpw/            # use test set for training               
    └── sequenceFiles/
        └── test/
```


## Reproduce Results

### Quick Start (Batch Execution)
Run all experiments at once using:
```bash
bash run_all.sh
```

### Individual Experiments
1. To reproduce **Mocap30to30** setting results, run:
   ```bash
   python src/baseline_h36m_30to30_pips/train.py
   ```
2. To reproduce **Mupots30to30** setting results, run:
   ```bash
   python src/baseline_h36m_30to30/train_no_traj.py
   ```
3. To reproduce **Mocap15to15** setting results, run:
   ```bash
   python src/baseline_h36m_15to15/train.py
   ```
4. To reproduce **Mupots15to15** setting results, run:
   ```bash
   python src/baseline_h36m_15to15/train_no_traj.py
   ```
5. To reproduce **3dpw_norc** setting results, run:
   ```bash
   python src/baseline_3dpw/train_norc.py
   ```
   **Note: Uncomment line 224 in src/models_dual_inter_traj_3dpw/mlp.py**
6. To reproduce **3dpw_rc** setting results, run:
   ```bash
   python src/baseline_3dpw/train_rc.py
   ```
   **Note: Uncomment line 223 in src/models_dual_inter_traj_3dpw/mlp.py**
7. To reproduce **Mocap15to45** setting results, run:
   ```bash
   python src/baseline_h36m_15to45/train.py
   ```
8. To reproduce **3dpw_rc(pretrain)** setting results, run:
   ```bash
   python src/baseline_3dpw_big/train_rc.py
   ```

9. To reproduce **3dpw_norc(pretrain)** setting results, run:
   ```bash
   python src/baseline_3dpw_big/train_norc.py
   ```


### Expected Results
- Experimental results will be saved in the `exprs/` folder with appropriate naming conventions based on the experiment settings.

### Important Notes
- The **first value** of each metric represents the **average**:
  1. In our paper, we **truncate data to one decimal place** (the same operation is also applied to **other models to ensure fairness**).