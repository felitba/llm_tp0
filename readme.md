# Transformers LLM Project

## Project Overview
This project explores the use of transformer-based models for predicting product purchase behavior in a supermarket setting. The goal is to model the Buy Through Rate (BTR) using product metadata, textual descriptions, and structured features from the supermarket catalog.

The work is grounded in a dataset of supermarket products and is designed to support research, experimentation, and model development for recommendation and conversion prediction tasks.

## Problem Statement
We aim to predict whether a product will be bought based on its attributes and contextual information. The target variable is derived from the `bought` column and is used to estimate a purchase likelihood signal for each product listing.

Key project goals include:
- Understanding product-level purchase patterns
- Building a transformer-inspired model architecture
- Preprocessing and encoding product features
- Evaluating model performance with appropriate metrics
- Comparing model variants and ablations
- Developing a personalization strategy for future extensions

## Data
The project uses the `dataset/supermarket_products.csv` dataset, which includes fields such as:
- product title and description
- price and category
- purchase signals (`bought`, `cart`)
- storage type and packaging attributes
- ingredient and nutrition details
- brand, dimensions, and country of origin

The dataset is intended for train/validation/test splitting and downstream feature engineering.

## Project Structure
- `main.py` — training pipeline entry point
- `model.py` — full transformer model implementation
- `embedding.py` — product feature embedding logic
- `tokenizer.py` — text/token preprocessing utilities
- `positional_encoding.py` — positional encoding module
- `encoding_block.py` — transformer encoder building block
- `preprocess_dataset.py` — data preprocessing and feature preparation
- `config/config.py` — configuration loading utilities
- `config/config.json` — hyperparameter and training configuration
- `dataset/supermarket_products.csv` — supermarket product dataset

## Model Approach
The project uses a transformer-style architecture for sequence-based product representation learning. The pipeline is expected to include:
- text tokenization for titles and descriptions
- embedding of categorical and numeric product features
- positional encodings for ordering-sensitive modeling
- encoder blocks for contextual representation learning
- classification output for BTR prediction

## Development Workflow
Planned workflow:
1. Explore the dataset and formulate the prediction task
2. Define the target variable and select relevant features
3. Create preprocessing and encoding pipelines
4. Build and train the transformer model
5. Evaluate using classification metrics
6. Run experiments and ablation studies
7. Document design decisions and results

## Metrics
The project includes evaluation using:
- PR-AUC
- ROC-AUC
- Precision
- Recall
- Validation and test set performance tracking

## Setup
Requirements may include:
- Python 3.10+
- PyTorch or relevant ML framework
- pandas
- numpy
- scikit-learn
- matplotlib / seaborn for analysis and reporting

Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv .venv
.venv\Scripts\activate.bat

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

## Usage
Run the main training script from the project root:

```bash
python main.py
```

Configuration is loaded from:

```text
config/config.json
```

## Collaboration
This project is intended to be developed collaboratively. Please fill in the placeholders below as the team evolves.

### Collaborators
- Project Lead: [Collaborator 1 Name]
- Data Analyst: [Collaborator 2 Name]
- ML Engineer: [Collaborator 3 Name]
- Reviewer / Documentation Lead: [Collaborator 4 Name]

### Contribution Notes
- Update the README as the project evolves
- Keep experiments and model decisions documented
- Record model hyperparameters and results in a reproducible way
- Add notes for any personalization or extension work

## Status
This project is currently in active development. The implementation and experiments are expected to be expanded over time.

## Notes
This repository is intended for coursework, research, and collaborative experimentation around transformers for product conversion prediction.

## Future Work
- Add a richer personalization layer based on user and query context
- Compare transformer models against baseline approaches
- Improve preprocessing and feature engineering
- Produce final presentation and project report
