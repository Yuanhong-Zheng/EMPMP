# 🚀 [ICCV 2025] Efficient Multi-Person Motion Prediction by Lightweight Spatial and Temporal Interactions

<p align="center">
  <a href="https://arxiv.org/abs/2507.09446">
    <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper">
  </a>
  <a href="https://iccv2025.thecvf.com/">
    <img src="https://img.shields.io/badge/Conference-ICCV%202025-blue" alt="Conference">
  </a>
</p>

## 🎯 Quick Start

### 🔧 Environment Setup
- **Python Version:** 3.8
```bash
pip install -r requirements.txt
```

### 💻 Hardware Requirements
We use a single **NVIDIA 3090 GPU**. To fully reproduce our results, we recommend using the same GPU.

### 📁 Path Configuration
Ensure your current working directory is **EMPMP**. If different, modify the `C.repo_name` variable in `src/baseline_3dpw/config.py` to match your working directory name. 

> **⚠️ Note:** All folders starting with "baseline" and all folders starting with "models" have almost identical code formats and are relatively independent, so you also need to modify config.py in other baseline folders.

---

## 📊 Data Preparation

### 📥 Download Dataset Files

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

### 📂 Data Structure (@/data)

```
data/
├── mupots_120_3persons.npy        
├── somof_test.pt   # use training set for test               
├── test_3_120_mocap.npy          
├── train_3_120_mocap.npy         
└── 3dpw/            # use test set for training             
    └── sequenceFiles/
        └── test/
```
---

## 🔬 Reproduce Results

### ⚡ Quick Start (Batch Execution)
Run all experiments at once using:
```bash
bash run_all.sh
```

### 🧪 Individual Experiments

| **Setting** | **Command** |
|-------------|-------------|
| **Mocap30to30** | `python src/baseline_h36m_30to30_pips/train.py` |
| **Mupots30to30** | `python src/baseline_h36m_30to30/train_no_traj.py` |
| **Mocap15to15** | `python src/baseline_h36m_15to15/train.py` |
| **Mupots15to15** | `python src/baseline_h36m_15to15/train_no_traj.py` |
| **3dpw_norc** | `python src/baseline_3dpw/train_norc.py` |
| **3dpw_rc** | `python src/baseline_3dpw/train_rc.py` |
| **Mocap15to45** | `python src/baseline_h36m_15to45/train.py` |
| **3dpw_rc(pretrain)** | `python src/baseline_3dpw_big/train_rc.py` |
| **3dpw_norc(pretrain)** | `python src/baseline_3dpw_big/train_norc.py` |

### 📋 Important Notes
- The **first value** of each metric represents the **average**
- In our paper, we **truncate data to one decimal place** (the same operation is also applied to **other models to ensure fairness**)

## 📚 Citation

If you find our work helpful, please cite our paper:

```
@article{arxiv250709446,
  title={Efficient Multi-Person Motion Prediction by Lightweight Spatial and Temporal Interactions},
  author={Anonymous},
  journal={arXiv preprint arXiv:2507.09446},
  year={2025},
  archivePrefix={arXiv},
  eprint={2507.09446},
  primaryClass={cs.CV}
}
```
