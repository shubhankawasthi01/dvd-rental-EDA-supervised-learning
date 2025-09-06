# dvd-rental-EDA-supervised-learning
DVD Rental Analysis and Prediction

Overview

This project analyzes a DVD rental dataset to predict rental duration using machine learning. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, and model training with hyperparameter tuning.

Dataset

The dataset (rental_info.csv) contains DVD rental details, including rental/return dates, payment amounts, movie attributes (e.g., length, rating, special features), and more.

Key Steps





Data Preprocessing: Converted date columns to datetime, calculated rental duration, and created dummy variables for special features.



EDA: Visualized distributions, correlations, and relationships between rental duration, movie ratings, and numerical features using histograms, boxplots, and heatmaps.



Feature Selection: Used Lasso regression to select relevant features for modeling.



Model Training: Evaluated Ridge, Lasso, Random Forest, Linear Regression, and KNN models with and without hyperparameter tuning via RandomizedSearchCV.



Results: Compared model performance using Mean Squared Error (MSE) and R² scores.

Dependencies





Python 3.x



Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

Usage





Clone the repository.



Install dependencies: pip install -r requirements.txt



Place rental_info.csv in the project directory.



Run the script: python rental_analysis.py

Results





Identified key predictors of rental duration using Lasso feature selection.



Random Forest with hyperparameter tuning achieved the best performance (lowest MSE, highest R²).



Visualizations revealed insights into rental patterns by movie rating and special features.

Future Work





Incorporate additional features (e.g., customer demographics).



Experiment with advanced models like Gradient Boosting.



Optimize hyperparameter search for better performance.
