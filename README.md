
# 🏡 House Price Prediction Using Linear Regression Models

## 📌 Project Overview
This project predicts house prices using machine learning regression algorithms. It includes data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## 🎯 Objectives
- Analyze property data to identify important price factors.
- Perform data preprocessing and feature engineering.
- Train and compare multiple regression models.
- Evaluate model performance using regression metrics.

## 🛠️ Technologies Used
- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Jupyter Notebook

## 🤖 Machine Learning Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Gradient Boosting Regressor

## ⚙️ Project Workflow
1. Data collection and loading
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature engineering and selection
5. Model training and comparison
6. Hyperparameter tuning using GridSearchCV
7. Model evaluation using RMSE, MAE, and R²

## 📊 Evaluation Metrics
- **RMSE:** Measures the typical prediction error, with greater penalties for large errors.
- **MAE:** Measures the average absolute prediction error.
- **R² Score:** Measures how much variation in house prices the model explains.

## 📂 Project Structure
```text
House_Price_Prediction/
│
├── dataset/
│   └── house_prices.csv
│
├── notebooks/
│   └── house_price_analysis.ipynb
│
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── models/
│   ├── linear_regression.pkl
│   ├── ridge_regression.pkl
│   └── lasso_regression.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 Installation and Usage

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/house-price-prediction-ml.git
cd house-price-prediction-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook
```

Open `house_price_prediction.ipynb` and run the cells in order.

## 📈 Results
The regression models are compared using RMSE, MAE, and R² Score. The best-performing model is selected based on evaluation results.

*Add your actual model metrics and visualizations here after running the project.*

## 👨‍💻 Author
**Your Name**

- GitHub: https://github.com/Shankhadeep04

## 📄 License
This project is intended for educational and portfolio purposes.
