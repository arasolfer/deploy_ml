## Deploy demo: Model training

Repository for deploying a machine learning model training pipeline. 


## Table of Contents
- [Deploy demo: Model training](#deploy-demo-model-training)
- [Table of Contents](#table-of-contents)
- [Project Overview](#project-overview)
  - [Key Features:](#key-features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)


## Project Overview

This repository contains a machine learning training pipeline designed for data preprocessing, model training, evaluation, and storage. It provides modular functionalities to adapt to different workflows and datasets.

### Key Features:

- Configurable data preprocessing.
- Model training with adjustable hyperparameters.
- Model evaluation using standard metrics.
- Saving models in a deployment-ready format.


## Project Structure

deployment_ml/
├── data/                     # Input data folder 
    ├── external/             # External data 
    ├── interim/              # Interim data
    ├── processed/            # Processed data
    ├── raw/                  # raw data
├── models/                   # Scaler and trained models folder  
├── src/                      # Source code
│   ├── data/                 # Data-related modules
│   │   ├── data_loader.py    # Data loading
│   │   ├── data_splitter.py  # Splits data into train/test sets
│   │   ├── data_processor.py # Data preprocessing and transformation
│   ├── model/                # Model-related modules
│   │   ├── trainer.py        # Model training
│   │   ├── evaluator.py      # Model evaluation
│   │   ├── saver.py          # Saves the trained model
│   ├── ├──main.py            # Orchestrates the training pipeline
├── .gitignore                # Specifies files and folders to ignore by Git
├── pyproject.toml            # Poetry configuration
├── poetry.lock               # poetry dependencies
├── README.md                 # Project documentation (this file)

## Requirements
- Python 3.11 or higher
- Poetry for dependency management

## Dependencies
All dependencies are listed in pyproject.toml. Key libraries include:

- pandas: Data manipulation
- xgboost: Model training
- scikit-learn: Data splitting and evaluation
- joblib: Saving and loading models

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arasolfer/deploy_ml.git
   cd deploy_ml
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Prepare the environment:**
   - Place your dataset in the `data/raw/` folder.
   - Ensure the `models/` folder exists to store trained models.

---

## Usage

To execute the training pipeline, run the following command:

```bash
poetry run python src/model/main.py
```

This command performs the following tasks:
- Loads data from `data/raw/`.
- Preprocesses the data (normalization, imputation, etc.).
- Splits the data into training and testing sets.
- Trains a model using the processed data.
- Evaluates the trained model and displays metrics.
- Saves the trained model in the `models/` folder.

---

## Output

The trained model will be saved in the `models/` folder with a timestamped filename, for example:

```
models/
├── model_2025-01-11_15-30.joblib
```

Evaluation metrics will also be printed to the console for easy analysis.

---

## Contributing

If you wish to contribute to this project:
1. Fork the repository.
2. Create a new branch for your changes:
   ```bash
   git checkout -b feature/new-feature
   ```
3. Push your changes and open a Pull Request.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).



