# Rice Blast Disease Prediction & Monitoring Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-86.5%25-brightgreen.svg)

##Live Demo: [Click here to view the live Streamlit application!](https://rice-blast-disease-prediction-cmv5wccba4xdwzlspcterb.streamlit.app/)

##  Project Overview
This project provides an end-to-end Machine Learning pipeline and a premium interactive web dashboard to detect **Rice Blast disease** at early stages. By analyzing environmental conditions (Temperature, Humidity, Leaf Wetness) and physical plant features (Lesion Size, Nitrogen Levels), the system predicts the likelihood of disease occurrence, enabling rapid intervention for crop protection.

##  Key Features & Techniques
- **Interactive Web Dashboard:** A premium, modern UI built with **Streamlit** to predict crop health dynamically using sliders, visual metric cards, and performance dashboards.
- **Algorithm:** Multilayer Perceptron (MLP) Neural Network (`MLPClassifier`).
- **Realistic Performance:** Achieves a highly realistic **86.5% Accuracy**, along with precision, recall, and F1-score evaluation. 
- **Class Imbalance Handling:** Used **SMOTE** (Synthetic Minority Over-sampling Technique) to oversample the minority disease class, significantly improving recall and overall model robustness in real-world skewed agricultural datasets.
- **Data Preprocessing:** Standardized features using `StandardScaler` to ensure the neural network converges quickly and effectively.

##  File Structure
- `app.py`: The main Streamlit web application featuring a beautiful frontend for making predictions and visualizing model metrics.
- `train_model.py`: Handles class imbalance via SMOTE, preprocesses data, trains the MLP model, generates evaluation plots, and saves the final model/metrics.
- `predict.py`: A terminal-based script demonstrating how to load the saved `.pkl` files and make terminal-based predictions.
- `data_generation.py`: Generates the synthetic agricultural dataset (`rice_blast_data.csv`) mimicking real-world conditions.
- `metrics.json` & `plots/`: Automatically generated files containing the model's performance metrics and visual charts (Confusion Matrix, Feature Correlation, Class Distribution).
- `requirements.txt`: Python package dependencies required to run the project.

##  How to Run Locally

### 1. Clone & Install Dependencies
First, clone the repository and install the required Python packages:
```bash
# git clone https://github.com/yourusername/RiceBlastPrediction.git
# cd RiceBlastPrediction
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard (Recommended)
To open the interactive web application, run:
```bash
streamlit run app.py
```
This will automatically open a browser window at `http://localhost:8501`.

### 3. Terminal-Based Predictions
If you prefer to run a quick terminal test without the UI:
```bash
python predict.py
```

### 4. Re-train the Model (Optional)
If you wish to re-generate the dataset and train the model from scratch:
```bash
python data_generation.py
python train_model.py
```

