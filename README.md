# 🩺 AI Healthcare - Diabetes Prediction

An AI-powered **healthcare dashboard** built with **Python, Streamlit, and Machine Learning** to predict whether a patient is likely to have diabetes based on medical parameters.

---

## 🚀 Features
- ✅ **Single Patient Prediction** – Enter details manually and get instant results.  
- ✅ **Batch Prediction (CSV Upload)** – Upload a CSV file with multiple patients' data and get predictions for all.  
- ✅ **Downloadable Results** – Export predictions as a CSV file for further analysis.  
- ✅ **Probability Scores** – Get the likelihood of diabetes for each patient.  
- ✅ **Feature Importance Visualization** – Understand which features influence predictions most.  
- ✅ **Modern UI** – Gradient backgrounds, custom header, and professional healthcare theme.  

---

## 📊 Dataset
This project uses the **PIMA Indians Diabetes Dataset** (UCI Repository).  
It contains **8 medical features**:  
- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  

Target Variable: `Outcome` (0 = No Diabetes, 1 = Diabetes).  

---

## ⚙️ Tech Stack
- **Python 3.9+**
- **Streamlit** – Interactive dashboard  
- **Scikit-learn** – Machine learning  
- **Pandas & NumPy** – Data handling  
- **Matplotlib/Seaborn** – Visualization  

---

## 📂 Project Structure
AI-Healthcare-Diabetes-Predictor/
│── app_streamlit.py # Streamlit app (dashboard)
│── model_training.py # Model training script
│── diabetes_model.pkl # Saved ML model (generated after training)
│── patients.csv # Sample input file for batch predictions
│── requirements.txt # Python dependencies
│── README.md # Project documentation

yaml
Copy code

---

## 🛠️ Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/AI-Healthcare-Diabetes-Predictor.git
   cd AI-Healthcare-Diabetes-Predictor
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Train the model (creates diabetes_model.pkl):

bash
Copy code
python model_training.py
Run the Streamlit app:

bash
Copy code
streamlit run app_streamlit.py
