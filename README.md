# spgvla: Simple Progress Guidance For Vision Language Model

## Overview


This repo is based on [lehome-challenge](https://github.com/lehome-official/lehome-challenge) and [kai0](https://github.com/OpenDriveLab/kai0), thanks for their excellent work and useful repo.

## Hardware

GPU: 1 L40, CPU: 14 core 120G. CUDA: 12.8


## Start

Because this repo is based on [lehome-challenge](https://github.com/lehome-official/lehome-challenge), you should read lehome-challenge firstly, prepare venv and copy Assets and third_party folder(too large file).

## Training and Evaluation

### Training

```bash
grep -q "export HF_ENDPOINT=" ~/.bashrc || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc


nohup python scripts/lerobot_train.py --config_path=configs/train_spgvla.yaml > train_logs/spgvla_exp.log 2>&1 &


```

### Evaluation

```bash
bash scripts/run_eval.sh
```


## Results and Analysis
This repo chooses the lehome challenge official dataset for testing, the metric is success rate(SR)

### Results

[spgvla model checkpoint] todo   


| experiments setting           | top long | top short | pants long | pants short | mean SR |
|-------------------------------|----------|-----------|------------|-------------|---------|
| baseline(SmolVLA)              | 61.67%   | 10%       | 31.67%     | 76.67%      | 45%     |
| baseline+spg                   | 55%      | 21.67%    | 45%        | 80%         | 50.4%   |
| baseline+spg+bs64              | 63.33%   | 25%       | 33.33%     | 88.33%      | 52.5%   |
| baseline+spg+bs64+wm           | 70%      | 25%       | 45%        | 86.67%      | 56.67%  |


## Others

If you are interested in spgvla, you can try to use spgvla to your model, and if you have any questions or suggestions, you can raise an issue.



## Acknowledgements

- [lehome-challenge](https://github.com/lehome-official/lehome-challenge)

- [kai0](https://github.com/OpenDriveLab/kai0)
