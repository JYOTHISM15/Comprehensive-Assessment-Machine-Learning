# Car Price Prediction using Machine Learning
## Objective
The project aims to predict the price of cars in the American market by analyzing various factors (independent variables) affecting the price. The goal is to understand how the prices vary with respect to features like car specifications, brand, year, etc. The company will use the model to understand the pricing dynamics in a new market, which will assist in their decision-making process for producing cars locally.

## Dataset
The dataset contains information about various car models and their attributes. It includes independent variables such as car specifications, brand, engine type, and other features that may affect the price of a car.

## Dataset link: Car Price Data 
https://drive.google.com/file/d/1FHmYNLs9v0Enc-UExEMpitOFGsWvB2dP/view?usp=drive_link

### Steps Performed
### 1. Data Preprocessing
Loaded the dataset and checked for missing values.
Cleaned the dataset by handling missing values and encoding categorical variables.
Performed feature scaling to standardize numerical values where necessary.
Split the data into training and testing sets.
### 2. Model Implementation
Five regression models were implemented to predict the price of cars:

#### Linear Regression: A simple linear model to predict the price based on the relationship between features and price.
Decision Tree Regressor: A tree-based model that captures non-linear relationships and interactions between features.
Random Forest Regressor: An ensemble method based on multiple decision trees to improve performance and reduce overfitting.
Gradient Boosting Regressor: A boosting technique that builds trees sequentially to reduce errors and improve prediction accuracy.
Support Vector Regressor (SVR): A regression model based on support vector machines, effective for high-dimensional feature spaces.
### 3. Model Evaluation
The models were evaluated using the following metrics:

#### R-squared (R²): To measure how well the model explains the variance in the target variable (car price).
Mean Squared Error (MSE): To quantify the difference between predicted and actual values.
Mean Absolute Error (MAE): To measure the average absolute difference between predicted and actual prices.
The evaluation results were used to compare the models and identify the best-performing one.

### 4. Feature Importance Analysis
The feature importance of different variables affecting car prices was analyzed using Random Forest and Gradient Boosting models. This helped in identifying which variables had the greatest impact on the price prediction. Visualizations of feature importances were provided.

### 5. Hyperparameter Tuning
Hyperparameter tuning was performed using GridSearchCV to improve the performance of the models. The models that performed best in the initial evaluation were further optimized by selecting the optimal hyperparameters.

## Results
Best Performing Model: The Random Forest Regressor outperformed other models based on R² and MSE. It effectively captured the relationships between features and the car price.
Key Features: Key variables such as engine type, car brand, and manufacturing year were found to have the greatest influence on the car prices.

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

## Conclusion
The model provides insights into the factors affecting car pricing in the US market, with a high-performing model ready for predicting future prices. By tuning the models and analyzing feature importance, the company can adjust their business strategies and design decisions to meet specific price targets in the US market.
