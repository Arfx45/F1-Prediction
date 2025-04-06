# F1 Race Prediction Project

Machine learning models to predict Formula 1 race outcomes using historical data and driver statistics.

## Project Overview

This project implements multiple machine learning algorithms to predict Formula 1 race winners using data from 1950-2020. Models include Random Forest, SVM, RNN, XGBoost, and Logistic Regression.

## Dataset

- Source: [Formula 1 World Championship (1950-2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- Features include:
  - Driver Elo ratings
  - Grid positions
  - Circuit characteristics
  - Historical performance

## Models Implemented

- Random Forest (97.40% accuracy)
- Support Vector Machine (96.80% accuracy)
- Recurrent Neural Network (96.19% accuracy)
- XGBoost (96.08% accuracy)
- Logistic Regression (95.80% accuracy)

## Project Structure

```
F1-Prediction/
├── data/
│   ├── circuits.csv
│   ├── drivers.csv
│   ├── races.csv
│   ├── results.csv
│   └── driver_elo.csv
├── models/
│   ├── rnn.py
│   ├── xgboost_model.py
│   ├── svm_model.py
│   ├── random_forest.py
│   └── logistic_regression.py
└── simulate/
    └── simulateRNN2024.py
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/F1-Prediction.git
cd F1-Prediction
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Train models:
```bash
python rnn.py
python xgboost_model.py
python svm_model.py
python random_forest.py
python logistic_regression.py
```

4. Run predictions:
```bash
python simulateRNN2024.py
```

## Results

Model predictions are saved in:
- `data/predicted_models/predicted_results2024RNN.csv`
- `data/predicted_models/predicted_results2024XGB.csv`
- `data/predicted_models/predicted_results2024SVM.csv`
- `data/predicted_models/predicted_results2024RF.csv`
- `data/predicted_models/predicted_results2024LR.csv`

## Contributors

- Ryan Shafi (501167088)
- Arnab Nath (501165959)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
