# Multilingual Toxicity Detection

This project aims to develop an automatic toxicity prediction model for multilingual comments using deep learning methods. The model is fine-tuned on a toxicity detection dataset provided by the course staff, with potential augmentation from publicly available datasets such as the Jigsaw dataset.

## Project Overview
We utilize **Toxic-BERT** from Hugging Face to build a multilingual toxicity detection model. Our approach includes:
- Experimenting with **BERT-based models** for text classification.
- Evaluating model performance using **precision, accuracy, recall, and F1-score**.
- Applying **fine-tuning strategies**, including:
  - Improved **data preprocessing**.
  - Modifying model architecture (e.g., adding **CNN layers**).
  - Expanding datasets for better generalization.
  - Refining label categories.
- Exploring **machine translation models** to enhance toxicity detection in low-resource languages like Finnish and German.

## Repository Structure
```
├── src
│   ├── data_loaders.py    # Data loading utilities
│   ├── models.py          # Model definitions (BERT-based classifier)
│   ├── preprocessing.py   # Preprocessing functions (tokenization, cleaning, etc.)
│   ├── utils.py           # Helper functions
│
├── train.py               # Training script
├── test.py                # Inference and evaluation script
├── config.json            # Configuration file with training parameters
├── results/               # Stores training outputs (checkpoints, logs, predictions)
│
└── README.md              # Project documentation
```

## Installation & Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Training the Model
To train the BERT-based toxicity classifier, run:
```bash
python train.py
```
This script will:
- Load and preprocess data.
- Initialize the BERT classifier.
- Train the model and save checkpoints in the `results/` directory.

## Testing & Evaluation
To perform inference on the test dataset:
```bash
python test.py
```
This script will:
- Load the trained model.
- Perform predictions on the test dataset.
- Save results in `results/test.tsv`.

## Configuration
Modify `config.json` to adjust training parameters such as:
```json
{
    "bert_name": "bert-base-uncased",
    "epochs": 10,
    "batch_size": 16,
    "max_length": 256,
    "learning_rate": 2e-5
}
```

## Acknowledgments
This project is part of a competition on multilingual toxicity detection. We extend our thanks to the course staff for providing the dataset and guidance.

<!-- ## License
MIT License -->

