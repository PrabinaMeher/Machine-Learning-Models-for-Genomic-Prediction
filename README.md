# Machine Learning and Deep Learning Models for Genomic Prediction

This repository contains Python scripts for genomic prediction using different Machine Learning (ML) and Deep Learning (DL) models.

The repository provides scripts for different data partitioning and validation strategies:

* Random Repeated Cross-Validation
* Leave-One-Environment-Out (LOEO) Cross-Validation
* Cross-Environment Validation

The general workflow is:

**Input genomic data → Data splitting → ML/DL model training → Prediction → Performance evaluation**

**Important:** The required data-splitting procedure should be performed **before** running the corresponding ML or DL model scripts.

---

## Requirements

The code was developed and tested using:

| Package      | Version |
| ------------ | ------- |
| Python       | 3.12.2  |
| NumPy        | 1.26.4  |
| Pandas       | 2.3.3   |
| Scikit-learn | 1.7.2   |
| SciPy        | 1.15.3  |
| Optuna       | 4.4.0   |
| XGBoost      | 2.1.1   |
| CatBoost     | 1.2.10  |
| TensorFlow   | 2.16.1  |
| Joblib       | 1.4.2   |

---

## Installation

Create a new Conda environment:

```bash
conda create -n genomic_prediction python=3.12
conda activate genomic_prediction
```

Install the required packages:

```bash
pip install numpy==1.26.4 pandas==2.3.3 scikit-learn==1.7.2 scipy==1.15.3 optuna==4.4.0 xgboost==2.1.1 catboost==1.2.10 tensorflow==2.16.1 joblib==1.4.2
```

---

# Methodology

The scripts implement genomic prediction using both traditional Machine Learning and Deep Learning approaches.

Three validation strategies are provided.

## 1. Random Repeated Cross-Validation

Random repeated cross-validation creates multiple random training and testing partitions from the available dataset.

For each repetition, the models are trained using the training data and evaluated on the corresponding test data.

This approach provides an assessment of model performance across multiple random data partitions.

---

## 2. Leave-One-Environment-Out (LOEO) Cross-Validation

In LOEO cross-validation, one environment is completely left out as the test environment, while the remaining environments are used for training.

The procedure is repeated so that each environment is used as the left-out test environment.

For models using hyperparameter optimization, an inner **5-fold cross-validation** is performed using only the training data to select the best hyperparameters.

The left-out environment is not used during hyperparameter optimization. After selecting the best hyperparameters, the final model is trained using the training data and evaluated on the left-out environment.

---

## 3. Cross-Environment Validation

Cross-environment validation evaluates the ability of a model to predict data from an environment that is separated from the training data.

This approach is used to assess model performance when predicting across different environments.

---

# How to Use the Scripts

The analysis should be performed in **two main steps**:

### Step 1: Generate the required data splits

First, select and run the appropriate data-splitting script according to the validation strategy.

### Step 2: Run the ML or DL model

After the required split files have been generated, run the corresponding Machine Learning or Deep Learning model script.

**Do not run the ML/DL scripts before generating the required data splits.**

---

# Step 1 - Generate Data Splits

## Random Repeated Cross-Validation

Run:

```bash
python Data_splits/Random_repeat_split.py
```

This generates the training and testing splits required for Random Repeated Cross-Validation.

After the split files are generated, the corresponding ML or DL scripts can be run.

---

## Leave-One-Environment-Out Cross-Validation

Run:

```bash
python Data_splits/LOEO_splits.py
```

This generates the required LOEO training and testing splits.

After generating the splits, run the desired ML or DL model.

---

## Cross-Environment Validation

Run:

```bash
python Data_splits/Cross_Env_split.py
```

This generates the training and testing data required for Cross-Environment Validation.

After generating the splits, run the desired ML or DL model.

---

# Step 2 - Run Machine Learning Models

After generating the appropriate data splits, select the ML model corresponding to the validation strategy.

## Random Repeated Cross-Validation

Examples:

```bash
python Random_repeated_cv/ML/SVM.py
```

```bash
python Random_repeated_cv/ML/RandomForest.py
```

```bash
python Random_repeated_cv/ML/XGBoost_model.py
```

Other available ML models include:

* AdaBoost
* Bagging
* CatBoost
* ELM
* Random Forest
* SVM
* XGBoost

---

## LOEO Cross-Validation

After generating LOEO splits:

```bash
python Data_splits/LOEO_splits.py
```

run the desired ML model.

For example:

```bash
python LOEO_cv/ML/SVM.py
```

```bash
python LOEO_cv/ML/CatBoost.py
```

```bash
python LOEO_cv/ML/XGB.py
```

---

## Cross-Environment Validation

First generate the splits:

```bash
python Data_splits/Cross_Env_split.py
```

Then run the desired ML model.

For example:

```bash
python Cross_environment_vaildation/ML/SVM.py
```

or:

```bash
python Cross_environment_vaildation/ML/CAB.py
```

---

# Run Deep Learning Models

After generating the appropriate data splits, the corresponding DL model can be executed.

## Random Repeated Cross-Validation

Examples:

```bash
python Random_repeated_cv/DL/ANN.py
```

```bash
python Random_repeated_cv/DL/MLP.py
```

```bash
python Random_repeated_cv/DL/DNN.py
```

```bash
python Random_repeated_cv/DL/CNN.py
```

Other available DL models include:

* ANN
* MLP
* DNN
* CNN
* RNN
* LSTM
* GRU
* BiLSTM

---

## LOEO Cross-Validation

First generate the LOEO splits:

```bash
python Data_splits/LOEO_splits.py
```

Then run the desired DL model.

Examples:

```bash
python LOEO_cv/DL/ANN.py
```

```bash
python LOEO_cv/DL/MLP.py
```

```bash
python LOEO_cv/DL/DNN.py
```

---

## Cross-Environment Validation

First generate the cross-environment splits:

```bash
python Data_splits/Cross_Env_split.py
```

Then run:

```bash
python Cross_environment_vaildation/DL/MLP.py
```

---

# Hyperparameter Optimization

Hyperparameter optimization is used where applicable to identify suitable model configurations.

For Deep Learning models, **Optuna** is used for hyperparameter optimization.

The optimization may include parameters such as:

* Number of hidden layers
* Number of neurons
* Activation function
* Dropout rate
* Learning rate
* Batch size
* Other model-specific parameters

For LOEO models using hyperparameter optimization, an inner **5-fold cross-validation** is performed only on the training data.
The left-out environment is reserved for final evaluation.

The selected hyperparameter configurations that achieved the highest PCC for a particular Dataset & Model in our analysis are uploaded in supplementary folder. 
---

# Model Training and Evaluation

After the required data splits are generated, the selected model:

1. Loads the training and testing data.
2. Performs required preprocessing and scaling.
3. Performs hyperparameter optimization where applicable.
4. Selects the best model configuration.
5. Trains the final model.
6. Generates predictions for the test data.
7. Calculates performance metrics.
8. Saves prediction and evaluation results.

The exact output files and locations depend on the selected validation strategy and model.

---

# Example Workflows

## Example 1 - LOEO with SVM

### Step 1: Generate LOEO splits

```bash
python Data_splits/LOEO_splits.py
```

### Step 2: Run SVM

```bash
python LOEO_cv/ML/SVM.py
```

The model will train using the available environments and evaluate predictions on each left-out environment.

---

## Example 2 - LOEO with MLP

### Step 1: Generate LOEO splits

```bash
python Data_splits/LOEO_splits.py
```

### Step 2: Run MLP

```bash
python LOEO_cv/DL/MLP.py
```

The MLP model will use the generated LOEO splits for training and evaluation.

---

## Example 3 - Random Repeated Cross-Validation with XGBoost

### Step 1: Generate random repeated splits

```bash
python Data_splits/Random_repeat_split.py
```

### Step 2: Run XGBoost

```bash
python Random_repeated_cv/ML/XGBoost_model.py
```

---

## Example 4 - Cross-Environment Validation with MLP

### Step 1: Generate cross-environment splits

```bash
python Data_splits/Cross_Env_split.py
```

### Step 2: Run MLP

```bash
python Cross_environment_vaildation/DL/MLP.py
```

---

# Reproducibility

For reproducible analyses, use the same:

* Dataset
* Data-splitting procedure
* Python version
* Package versions
* Model script
* Random seed
* Hyperparameter settings

The recommended computational environment is **Python 3.12.2** with the package versions listed in the Requirements section.

---

