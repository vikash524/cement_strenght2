# Concrete Compressive Strength Prediction

## Overview

This project aims to predict the compressive strength of concrete based on the input features such as the quantities of cement, water, superplasticizer, and other components in the mixture. The dataset includes data on different physical properties of concrete, and the target variable is the compressive strength of the concrete. Machine learning models are trained to predict this strength.

The models used in this project include linear regression (SGDRegressor), XGBoost, and hyperparameter optimization techniques such as **GridSearchCV** and **RandomizedSearchCV** to improve model performance.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Data Overview](#data-overview)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Author Information](#author-information)
- [License](#license)

## Project Description

The goal of this project is to predict the compressive strength of concrete using a dataset containing various concrete component quantities. The project involves:
- **Data Preprocessing**: Handling missing values, visualizing relationships between features, and scaling data.
- **Feature Selection**: Using Variance Inflation Factor (VIF) to remove highly correlated features.
- **Model Training**: Implementing multiple models (SGDRegressor, XGBoost) and tuning them to achieve the best performance.
- **Model Evaluation**: Evaluating model performance using metrics such as **RMSE** (Root Mean Squared Error) and **R² score**.
- **Hyperparameter Tuning**: Using **GridSearchCV** and **RandomizedSearchCV** for model optimization.
- **Deployment**: The model can be deployed as an interactive web application using **Streamlit**.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:


2. **Install required libraries**:
   Create a virtual environment and install the required dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate  # For Windows
   
   pip install -r requirements.txt
   ```

## Dependencies

The following Python libraries are used in this project:

- `numpy`: For numerical operations and data manipulation.
- `pandas`: For data analysis and manipulation.
- `scikit-learn`: For machine learning models and metrics.
- `xgboost`: For XGBoost implementation.
- `streamlit`: For creating the web application interface.
- `seaborn`: For creating visualizations like correlation matrices and plots.
- `matplotlib`: For creating various visualizations.
- `statsmodels`: For statistical tests and model diagnostics.

## Data Overview

The dataset used in this project contains the following features:

1. **Cement**: Cement (kg in a m^3 mixture)
2. **Blast Furnace Slag**: Blast Furnace Slag (kg in a m^3 mixture)
3. **Fly Ash**: Fly Ash (kg in a m^3 mixture)
4. **Water**: Water (kg in a m^3 mixture)
5. **Superplasticizer**: Superplasticizer (kg in a m^3 mixture)
6. **Coarse Aggregate**: Coarse Aggregate (kg in a m^3 mixture)
7. **Fine Aggregate**: Fine Aggregate (kg in a m^3 mixture)
8. **Age**: Age of the concrete (in days)
9. **Concrete Compressive Strength**: Target variable, the compressive strength (in MPa, Megapascals)

### Data Source

This dataset is available publicly and can be accessed at the following link:
[Concrete Compressive Strength Prediction Dataset](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)

## Models Used

1. **SGDRegressor**: A linear regression model trained using stochastic gradient descent. This model provides a baseline prediction for the compressive strength.
2. **XGBoost**: A gradient boosting model that is known for its high performance in predictive tasks. It is trained and tuned using techniques such as **RandomizedSearchCV** and **GridSearchCV**.
3. **Hyperparameter Optimization**: Techniques like **GridSearchCV** and **RandomizedSearchCV** are used to tune the hyperparameters of the models for better accuracy and performance.

### Evaluation Metrics

The following metrics were used to evaluate the models:

- **R² Score**: Measures the proportion of variance explained by the model.
- **Root Mean Squared Error (RMSE)**: Measures the prediction error of the model. The lower the RMSE, the better the model performance.
- **Durbin-Watson Test**: Used to check for autocorrelation in the residuals.
- **Q-Q Plot**: Used to check if the residuals are normally distributed.

## Author Information

- **Name**: Vikash Chauhan
- **Email**: vikashchauhanvv26@gmail.com


### About the Author

Vikash Chauhan is a Data Science student at apj abdul kalam college with a strong passion for machine learning, statistics, and data analysis. This project represents one of his efforts to apply machine learning techniques to real-world problems and gain hands-on experience.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to edit and expand the README as per your requirements.
