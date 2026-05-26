# 🏠 House Price Prediction - Complete Data Science Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-11557c?logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-4c72b0?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Hosted-181717?logo=github&logoColor=white)

---

## 🌐 Live Demo
👉 **[View Live App on Streamlit Cloud](YOUR_STREAMLIT_URL_HERE)**
📓 **[View Jupyter Notebook](YOUR_NOTEBOOK_URL_HERE)**

---

## 📌 Project Overview

> Predict house sale prices using Machine Learning on the **Ames Housing Dataset**.
> Full end-to-end Data Science project — from raw data to deployed web app.

| Item | Details |
|------|---------|
| 🎯 Goal | Predict house sale prices |
| 📊 Dataset | Ames Housing Dataset (Kaggle) |
| 🏠 Records | 1,460 houses |
| 🔢 Features | 79 original + 8 engineered |
| 🤖 Best Model | Gradient Boosting Regressor |
| 📈 Accuracy | ~90% R² Score |
| 🌐 Deployment | Streamlit Cloud |

---

## 🛠️ Tools & Technologies

### 💻 Languages & Environment
| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Main programming language |
| **Jupyter Notebook** | Latest | Interactive EDA & analysis |
| **Git & GitHub** | Latest | Version control & hosting |

### 📊 Data Analysis & Manipulation
| Tool | Version | Purpose |
|------|---------|---------|
| **Pandas** | 2.0.3 | Data loading, cleaning, manipulation |
| **NumPy** | 1.24.3 | Numerical operations, log transform |

### 📈 Data Visualization
| Tool | Version | Purpose |
|------|---------|---------|
| **Matplotlib** | 3.7.2 | Bar charts, histograms, scatter plots |
| **Seaborn** | 0.12.2 | Heatmaps, boxplots, styled charts |

### 🤖 Machine Learning
| Tool | Version | Purpose |
|------|---------|---------|
| **Scikit-learn** | 1.3.0 | ML models, preprocessing, evaluation |

### 🌐 Deployment
| Tool | Version | Purpose |
|------|---------|---------|
| **Streamlit** | 1.28.0 | Interactive web dashboard |

---

## 🤖 Machine Learning Models

| Model | R² Score | RMSE | Notes |
|-------|----------|------|-------|
| Linear Regression | ~0.83 | ~0.17 | Baseline model |
| Ridge Regression | ~0.85 | ~0.16 | Handles overfitting |
| Lasso Regression | ~0.84 | ~0.17 | Feature selection |
| Random Forest | ~0.88 | ~0.14 | Feature importance |
| **Gradient Boosting** | **~0.90** | **~0.12** | **🏆 Best Model** |

---

## 📁 Project Structure

```
HousePricePrediction/
│
├── 📓 house_price_prediction.ipynb   # Full Jupyter Notebook (EDA + ML)
├── 🌐 app.py                          # Streamlit Web App
├── 📋 requirements.txt                # Python dependencies
├── 📄 README.md                       # Project documentation
├── 🗂️ train.csv                       # Training dataset (1460 rows)
└── 🗂️ test.csv                        # Test dataset
```

---

## 📊 Project Workflow

```
📂 Load Data
    ↓
🔍 Explore Data (EDA)
    ↓
🧹 Clean Data (Handle Missing Values)
    ↓
⚙️  Feature Engineering (8 New Features)
    ↓
📊 Visualize (7+ Charts)
    ↓
🤖 Build & Train 5 ML Models
    ↓
📈 Evaluate (R², RMSE, MAE)
    ↓
🔮 Deploy (Streamlit Web App)
```

---

## ⚙️ Feature Engineering

New features created from existing data:

| Feature | Formula | Meaning |
|---------|---------|---------|
| `HouseAge` | 2025 - YearBuilt | Age of the house |
| `RemodAge` | 2025 - YearRemodAdd | Years since remodel |
| `TotalSF` | Basement + 1st + 2nd Floor | Total square footage |
| `TotalBathrooms` | Full + Half + Basement baths | Combined bathrooms |
| `TotalPorchSF` | All porch areas combined | Total porch area |
| `HasGarage` | GarageArea > 0 | Binary: has garage? |
| `HasPool` | PoolArea > 0 | Binary: has pool? |
| `HasFireplace` | Fireplaces > 0 | Binary: has fireplace? |

---

## 📈 Key Findings

1. **OverallQual** is the #1 predictor of house price
2. **GrLivArea** (Above ground living area) has strongest positive correlation
3. **TotalSF** (Total square footage) significantly impacts pricing
4. **Gradient Boosting** outperforms all other models
5. Newer houses sell at significantly higher prices
6. Houses with garage, fireplace & pool command premium prices

---

## 🚀 Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/HousePricePrediction.git
cd HousePricePrediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebook
```bash
jupyter notebook house_price_prediction.ipynb
```

### 4️⃣ Run Streamlit App
```bash
streamlit run app.py
```

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push project to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New App"**
4. Select your repo & set `app.py` as main file
5. Click **Deploy** 🎉

---

## 📸 Project Screenshots

### 📊 Data Overview Tab
> Dataset summary, sample data, missing values analysis

### 📈 Visualization Tab
> Price distribution, correlation charts, scatter plots

### 🤖 Model Results Tab
> Model comparison, actual vs predicted, feature importance

### 🔮 Price Predictor Tab
> Interactive form to predict any house price instantly

---

## 👨‍💻 Author

**Your Name**
- 🐙 GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- 💼 LinkedIn: [Your LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
- 📧 Email: your.email@gmail.com

---

## 📄 License

This project is licensed under the **MIT License** — free to use and modify.

---

## 🙏 Acknowledgements

- Dataset: [Kaggle - Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Inspiration: [DataWithBaraa](https://github.com/DataWithBaraa)

---

⭐ **If this project helped you, please give it a star on GitHub!**

> *"Data is the new oil — and this project shows how to refine it!"*
