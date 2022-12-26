![tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

# CVBOX-DETECTMASK
Python Face Mask Detection for Suratthani School CVBOX Project
## Installation

1. Download the program from the "Releases" Tab (see the release note for which file to download)
2. Install the required libraries

```bash
  pip3 install -r requirements.txt
```
    
## Model Training (Skip if downloaded DetectMask.zip or DetectMask_Arduino.zip)

```bash
  python3 train.py --dataset dataset
```
## Usage - Face Mask Detection

1. Set you configurations in the config.py
2. Run the face mask detection script
```bash
  python3 detectmask.py
```
