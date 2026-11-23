# City Hospital – Breast Cancer Model Dashboard

An interactive machine learning dashboard built with **Python, PyTorch, and Streamlit** that shows how a neural network can classify breast cancer cases as **benign** or **malignant** using the Breast Cancer Wisconsin dataset from Scikit-learn.

Adjust hyperparameters, train a model, watch training and validation loss in real time, inspect individual predictions, and evaluate on a held-out test set.

> ⚠️ **Disclaimer:** This is an educational and experimental project. It is **not** a medically validated diagnostic system and must **not** be used for real clinical decisions.

---

## Table of Contents

- [Screenshots](#screenshots)
- [Features](#features)
- [Objectives](#objectives)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Author](#author)

---

## Screenshots

| Dashboard | Training Results | Final Test Results |
|:---:|:---:|:---:|
| ![Dashboard](screenshots/ss1.JPG) | ![Training Results](screenshots/ss2.JPG) | ![Final Test Results](screenshots/ss3.JPG) |

---

## Features

- **Configurable model and training settings** from the sidebar
- **One-click training** with a live progress bar
- **Live loss chart** showing training vs. validation loss per epoch
- **Training summary** with epochs completed and best validation loss
- **Automatic training diagnosis:** healthy, overfitting, or underfitting
- **Experiment history** table with **CSV export**
- **Single-patient prediction** from the validation set
- **Final test evaluation** with accuracy, loss, and confusion counts

---

## Objectives

- Implement a binary classifier using PyTorch
- Understand neural network training and validation
- Explore how hyperparameters affect model performance
- Monitor training and validation loss across epochs
- Evaluate predictions using confusion matrix metrics
- Build an interactive dashboard for experimenting with model configurations

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Main programming language |
| **PyTorch** | Neural network implementation and training |
| **Scikit-learn** | Dataset loading, data splitting, feature scaling |
| **Pandas** | Results management and tabular display |
| **NumPy** | Numerical operations |
| **Streamlit** | Interactive web dashboard |

---

## Dataset

The project uses the **Breast Cancer Wisconsin** dataset via `sklearn.datasets.load_breast_cancer()`.

| Property | Value |
|---|---|
| Records | 569 |
| Input features | 30 (numerical) |
| Task | Binary classification |
| Malignant label | `1` |
| Benign label | `0` |

### Data Split

Data is split with **stratified sampling** to preserve class proportions.

| Subset | Share | Role |
|---|---:|---|
| Training | 60% | Model fitting |
| Validation | 20% | Monitoring and intermediate evaluation |
| Testing | 20% | Final evaluation only |

### Feature Scaling

Scaling is **optional** and can be toggled in the dashboard. When enabled, `StandardScaler` is fitted on the **training data only** and then applied to the validation and test sets, preventing data leakage.

---

## How It Works

### Model Architecture

A configurable feedforward neural network built with PyTorch:

- Fully connected (`Linear`) layers
- `ReLU` activations
- `Dropout` layers
- A final output layer for binary classification

Adjustable from the dashboard: number of hidden layers, hidden units, and dropout rate.

### Training Setup

| Component | Details |
|---|---|
| Optimizer | Adam |
| Loss function | Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) |
| Learning rate | Configurable |
| Batch size | Configurable |
| Max epochs | Configurable |
| Weight decay | Optional |
| Early stopping | Optional, based on validation loss |

Training and validation losses are recorded after every epoch.

### Evaluation Metrics

- Accuracy
- True positives / False positives
- False negatives / True negatives
- Training and validation loss

The validation set is used during development; the test set is reserved for the final evaluation.

---

## Installation

**Prerequisites:** Python 3.9 or later and `pip`.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/city-hospital-breast-cancer-dashboard.git

# 2. Move into the project folder
cd city-hospital-breast-cancer-dashboard

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The dashboard opens in your default browser. If it doesn't, visit:

```text
http://localhost:8501
```

---

## Usage

1. Launch the Streamlit app.
2. Use the **sidebar** to set model and training parameters (hidden layers, hidden units, learning rate, epochs, batch size, dropout, weight decay, early stopping).
3. Choose whether to enable **feature scaling**.
4. Click **Train**.
5. Watch the **training and validation loss chart** and read the training summary and diagnosis.
6. Review the **experiment history**.
7. Select a **validation patient** to see the model's prediction.
8. Run the **final test evaluation**.
9. Optionally **download the experiment history** as a CSV file.

---

## Results

Results depend on the selected configuration and random seed. The dashboard reports:

- Epochs completed
- Best validation loss
- Malignant cases correctly identified (validation set)
- False alarms (validation set)
- Final test accuracy and loss
- Final test confusion counts

> Actual performance varies with hyperparameters and random seed.

---

## Limitations

- Uses a built-in Scikit-learn dataset, **not real hospital patient records**
- Intended for learning and experimentation, **not clinical diagnosis**
- Performance depends heavily on hyperparameters and training configuration
- Provides no medical explanations or professional diagnostic recommendations
- Predictions must **not** be treated as reliable medical advice

---

## Author

**Zuha Wahab**
GitHub: [@Zuhaverse](https://github.com/Zuhaverse)

---

<sub>**Disclaimer:** This application is an academic machine learning project and is not intended for clinical use.</sub>
