# Dataset

## Source

The dataset used in this project is a publicly available NLP benchmark dataset containing labelled English sentences.

**File expected:** `data/dataset.csv`

## Format

| Column | Description |
|--------|-------------|
| `sentence` | Raw English text |
| `label` | `question` or `sentence` |

## How to obtain

The original dataset file used during development: `1646977175-5e748a2d5fc288e9f69c5f86.csv`

You can find similar open datasets for question classification on:
- [Kaggle — Question vs Statement datasets](https://www.kaggle.com/)
- [HuggingFace Datasets](https://huggingface.co/datasets)

After downloading, rename the file to `dataset.csv` and place it in this `data/` folder. Update the file path in the notebook accordingly.

## Note

The raw dataset file is not committed to this repository to keep the repo lightweight. The notebook and source code are fully reproducible once the dataset is in place.
