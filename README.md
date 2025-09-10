# Diabetes Prediction - AI for Healthcare (Beginner Project)

## Overview
This repository contains code to build a Diabetes Prediction model using the Pima Indians Diabetes Dataset.
The project includes data preprocessing, model training, evaluation, and a simple Streamlit app to demo predictions.

**Note:** This bundle does NOT include the dataset (`diabetes.csv`). Download the **Pima Indians Diabetes Dataset** (CSV) from Kaggle:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
Place the file in the project root and name it `diabetes.csv`.

## Files
- data_preprocessing.py  -> Load & preprocess the dataset
- model_training.py      -> Train models and save best model (Pickle)
- app_streamlit.py       -> Streamlit app to interact with the model
- requirements.txt       -> Python libraries required
- full_code.pdf          -> PDF containing all code (for easy reading/printing)
- diabetes_model.pkl     -> (created after running model_training.py)

## Quick steps
1. Create a virtualenv and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Put `diabetes.csv` in the project root.
3. Train and save model:
   ```bash
   python model_training.py
   ```
   This will create `diabetes_model.pkl`.
4. Run the app:
   ```bash
   streamlit run app_streamlit.py
   ```

## License
MIT
