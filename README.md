
# EEG Eye State Classification using Random Forest, XGBoost, and LSTM

## Project Overview

This project aims to classify eye state (open or closed) based on EEG signal readings using three supervised learning models: Random Forest, XGBoost, and a Long Short-Term Memory (LSTM) neural network. The dataset used is the EEG Eye State Dataset from the UCI Machine Learning Repository, which contains 14 EEG features collected from a single subject. The goal is to detect fatigue or drowsiness by analyzing these signals and accurately predicting the binary eye state.

## Dataset Description

The EEG Eye State Dataset includes EEG signal recordings from 14 sensors with a corresponding target column indicating the eye state (0 for open, 1 for closed). The data has no missing values and is suitable for binary classification tasks. For this project, we used the `ucimlrepo` Python package to fetch and load the dataset directly into a Pandas DataFrame.

## Approach and Methodology

The dataset was preprocessed by applying standard scaling to normalize the features, followed by feature selection using the SelectKBest method with ANOVA F-values to retain the top eight most informative features. These selected features were used consistently across all models to ensure fair comparison.

Three models were implemented and evaluated. Random Forest and XGBoost were trained using scikit-learn and XGBoost's Python API, respectively. Both models were evaluated based on accuracy, precision, and recall. The LSTM model was constructed using TensorFlow/Keras. Since LSTMs require 3D input, the feature matrix was reshaped accordingly, and labels were converted into categorical format for training. The LSTM architecture was kept simple with a single LSTM layer followed by a dense output layer using softmax activation.

## Results and Evaluation

All three models were evaluated on the same test split. Random Forest and XGBoost performed similarly, with high accuracy and balanced precision-recall metrics, indicating effective classification of both eye-open and eye-closed states. The LSTM model, despite the limited temporal depth of the data, achieved comparable results due to the rich signal-based feature set. Performance was measured using accuracy, precision, and recall, and results were consolidated in a comparison table displayed at the end of the notebook.

## Files

The project includes a single Jupyter Notebook containing data loading, preprocessing, model implementation, evaluation, and result comparison. This notebook serves as a complete and reproducible pipeline for EEG-based eye state classification.

## Integration with Digital Twin (Optional)

This model can be integrated into a Digital Twin system representing a real-time avatar of a human operator, driver, or pilot. EEG signals from wearable sensors can be streamed into the twin, which processes the signals using the trained classifier. In the event of detected eye closure or signs of fatigue, the Digital Twin can trigger alerts, activate safety protocols, or log the incident for further review, thereby enhancing situational awareness and safety in high-risk environments.

Dataset :- "https://www.google.com/url?q=https%3A%2F%2Farchive.ics.uci.edu%2Fdataset%2F264%2Feeg%2Beye%2Bstate"
