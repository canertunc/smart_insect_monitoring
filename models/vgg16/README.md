# Insect Classification with Fine-tuned VGG16

This project implements an insect classification model using a fine-tuned VGG16 architecture. The model is trained to classify various insect species from the provided dataset.

## Features

- Fine-tuned VGG16 model with 5 frozen layers
- Training with GPU support
- Automatic validation during training
- Detailed performance metrics and visualizations
- Modular code structure

## Project Structure

- `model.py`: Defines the VGG-based model architecture
- `data_loader.py`: Handles data loading and preprocessing
- `trainer.py`: Contains the training and evaluation logic
- `main.py`: Main script to run the training and evaluation
- `requirements.txt`: Lists all required dependencies

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To train the model with default settings:

```bash
python main.py
```

### Command-line Arguments

- `--data_dir`: Directory containing the dataset (default: 'dataset')
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of epochs to train for (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--freeze_layers`: Number of layers to freeze in VGG model (default: 5)
- `--results_dir`: Directory to save results (default: 'results')
- `--device`: Device to train on ('cuda' or 'cpu', default: 'cuda')

### Example

```bash
python main.py --epochs 15 --batch_size 64 --freeze_layers 7
```

## Results

After training, the following results will be saved in the results directory:

- Trained model weights
- Training history plot (accuracy and loss)
- Confusion matrix
- Classification report visualization

## Dataset

The dataset should be organized with each class in a separate subdirectory under the main dataset directory:

```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
``` 