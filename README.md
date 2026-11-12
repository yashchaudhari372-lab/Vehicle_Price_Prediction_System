<div align="center">

# 🚗 Vehicle Price Prediction System

### Machine Learning | Random Forest Regression | Python | Flask

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)](https://flask.palletsprojects.com/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?logo=vercel)](https://vehicle-price-prediction-random-for.vercel.app/)

**Estimate vehicle prices using a trained machine learning regression model through an interactive web application.**

[🌐 Live Demo](https://vehicle-price-prediction-random-for.vercel.app/) · [💻 Source Code](https://github.com/yashchaudhari372-lab/Vehicle_Price_Prediction_System)

</div>

---

## 📌 Project Overview

The **Vehicle Price Prediction System** is an end-to-end machine learning project designed to estimate vehicle prices based on information supplied by a user. It demonstrates how predictive modeling can support data-informed pricing decisions in the automotive market.

The application loads a pre-trained **Random Forest Regressor** and presents predictions through a Python-based web interface.

## ✨ Key Features

- 🚘 Vehicle price estimation using machine learning
- 🌳 Random Forest regression model
- 🖥️ Interactive web interface for entering vehicle details
- ⚡ Predictions from a saved model without retraining on every request
- ☁️ Publicly accessible deployed application
- 🧩 End-to-end demonstration of model integration with a web app

## 📊 Model Performance

| Metric | Reported result |
|---|---:|
| Algorithm | Random Forest Regressor |
| R² Score | **95.67%** |

> **Evaluation note:** The R² score is the result reported for this project by its author. The training dataset, evaluation script, test split, and reproducible metrics are not currently included in the public repository. R² describes variance explained on the evaluation data; it is not the percentage of individual price predictions that are correct.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | Random Forest Regressor, scikit-learn |
| Backend / Web Application | Flask |
| Model Storage | Pickle (`.pkl`) |
| Deployment | Vercel |
| Version Control | Git, GitHub |

## 🔄 How It Works

```text
Vehicle details entered by user
              ↓
       Web application
              ↓
      Input processing
              ↓
  Pre-trained Random Forest model
              ↓
     Predicted vehicle price
              ↓
       Display to user
```

### Machine Learning Workflow

The development workflow for a vehicle-price regression system typically includes:

1. **Data collection:** Gather historical vehicle listings and sale-price data.
2. **Data preprocessing:** Handle missing values, outliers, and data types.
3. **Feature engineering:** Prepare relevant vehicle attributes for modeling.
4. **Model training:** Train a Random Forest Regressor on labeled price data.
5. **Evaluation:** Evaluate predictions using regression metrics such as R², MAE, and RMSE.
6. **Model serialization:** Save the trained model as a `.pkl` file.
7. **Application integration:** Load the model in the Flask application and serve predictions.
8. **Deployment:** Make the prediction interface available online.

> These steps describe the general workflow. The repository currently publishes the application and serialized model, rather than the full training notebook or dataset.

## 📂 Repository Structure

```text
Vehicle_Price_Prediction_System/
├── app.py                 # Flask application and prediction interface
├── RFA_model.pkl          # Serialized trained regression model
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## ⚙️ Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/yashchaudhari372-lab/Vehicle_Price_Prediction_System.git
cd Vehicle_Price_Prediction_System
```

### 2. Create and activate a virtual environment (recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
python app.py
```

Open the local address displayed in your terminal (commonly `http://127.0.0.1:5000/`). Ensure the model file remains in the same location expected by `app.py`.

## 🌐 Live Demo

Try the deployed application:

**[🚀 Vehicle Price Prediction — Live Web App](https://vehicle-price-prediction-random-for.vercel.app/)**

Enter the requested vehicle information and submit the form to obtain an estimated price. Actual resale prices may differ depending on local market conditions, vehicle condition, and other factors.

## 🎯 Practical Applications

- **Used-vehicle buyers:** Compare a listing price with a model-generated estimate.
- **Sellers:** Explore a data-informed starting point for pricing.
- **Dealerships:** Support initial vehicle valuation workflows.
- **Data science learning:** Demonstrate a complete ML-to-web deployment pipeline.

## 🔮 Potential Future Improvements

- Add reproducible model-training and evaluation notebooks.
- Publish the dataset description and feature definitions.
- Report MAE and RMSE alongside R² on a held-out test set.
- Add model explanations and feature-importance visualizations.
- Validate user inputs and provide a prediction range or uncertainty estimate.
- Add screenshots of the application and example predictions.

## 👨‍💻 Author

**Yash Chaudhari**  
Aspiring Data Scientist & Data Analyst

- **GitHub:** [yashchaudhari372-lab](https://github.com/yashchaudhari372-lab)
- **LinkedIn:** [yashchaudhari08](https://www.linkedin.com/in/yashchaudhari08/)

---

<div align="center">

⭐ **If you find this project useful, consider starring the repository!**

</div>
