"""model_training.py
Train multiple models and save the best performing one (by F1-score).
Produces a simple evaluation printout.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from data_preprocessing import load_data, clean_data, feature_target_split

def train_and_evaluate(path='diabetes.csv', random_state=42):
    df = load_data(path)
    df = clean_data(df)
    X, y = feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Models to try
    models = {
        'logreg': LogisticRegression(max_iter=1000, random_state=random_state),
        'rf': RandomForestClassifier(n_estimators=100, random_state=random_state)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)
        results[name] = {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_score(y_test, preds, zero_division=0),
            'f1': f1_score(y_test, preds, zero_division=0),
            'model': model
        }

    # Pick best by F1
    best_name = max(results, key=lambda n: results[n]['f1'])
    best = results[best_name]
    print('Evaluation results:')
    for k, v in results.items():
        print(f"Model: {k} -> Accuracy: {v['accuracy']:.4f}, Precision: {v['precision']:.4f}, Recall: {v['recall']:.4f}, F1: {v['f1']:.4f}")

    # Save scaler + model using joblib
    pipeline = {'scaler': scaler, 'model': best['model']}
    joblib.dump(pipeline, 'diabetes_model.pkl')
    print(f"Best model: {best_name} saved to diabetes_model.pkl")

    # Detailed report for best model
    y_pred = best['model'].predict(X_test_sc)
    print('\nClassification report for best model:')
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    train_and_evaluate()
