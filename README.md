# MLOps

## TrashNet Image Classification

The project for the MLOps course. In this project, we train a deep learning model to classify waste images into 6 categories using the TrashNet dataset.


## Installation

Clone the repository and set up the environment:

```
git clone https://github.com/Theng1/Mlops.git
cd Mlops
```

## Data

To download the dataset:

```
dvc pull
```

## CLI

Train and evaluate model from terminal:

```
python3 -m venv mlops_env
source mlops_env/bin/activate
poetry install
pre-commit run -a
python trash_classifier/train.py
python trash_classifier/infer.py
```


## Useful links

- TrashNet dataset: https://github.com/garythung/trashnet
- DVC documentation: https://dvc.org/doc
- TorchVision models: https://pytorch.org/vision/stable/models.html
