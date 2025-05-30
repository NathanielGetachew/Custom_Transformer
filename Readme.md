# Custom Transformer for Text Generation

## Overview
This project implements a custom Transformer model for text generation using the Penn Treebank (PTB) dataset. The model is a decoder-only Transformer, inspired by nanoGPT and the Annotated Transformer, with a custom self-attention implementation.

## Directory Structure
- **src/**: Core implementation.
  - `data/`: Data loading and preprocessing.
  - `models/`: Transformer model and components.
  - `training/`: Training, evaluation, and generation scripts.
  - `utils/`: Utility functions (e.g., plotting).
- **tests/**: Unit tests for model components.
- **data/**: PTB dataset and preprocessed files.
- **checkpoints/**: Model checkpoints and final model.
- **scripts/**: Entry-point scripts for training, evaluation, and generation.

## Setup
1. Install Python 3.11.
2. Create a virtual environment:
3. Install dependencies:
4. (Optional) For GPU support, install PyTorch with CUDA:

## Usage
- **Train the model**:
python scripts/train_model.py
- **Evaluate the model**:
python scripts/evaluate_model.py
- **Generate text**:
python scripts/generate_text.py

## Requirements
See `requirements.txt` for dependencies.
