import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

st.title("Interactive Linear Regression and MSE Exploration")

# Introduction and Mathematical Background
st.markdown("""
## Introduction to Linear Regression
Linear Regression is a fundamental statistical method used for modeling the relationship between a scalar dependent variable \( y \) and one or more explanatory variables (or independent variables) denoted \( X \). The key goal is to find a linear relationship between these variables which can be used for prediction.
""")

# Mathematical Formula of Linear Regression
st.subheader("Mathematical Formulation")
st.markdown("""
The equation of a linear regression model is:
""")
st.latex(r'''
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon
''')
st.markdown("""
Where:
- \( \beta_0, \beta_1, ..., \beta_n \) are the coefficients of the model.
- \( x_1, x_2, ..., x_n \) are the predictor variables.
- \( \epsilon \) represents the error term, accounting for the variability in \( y \) not explained by the predictors.
""")

# Mean Squared Error (MSE) Explanation
st.subheader("Mean Squared Error (MSE)")
st.markdown("""
MSE is a measure used to evaluate the performance of a regression model by calculating the average squared difference between the actual observed responses and the responses predicted by the model. The formula for MSE is:
""")
st.latex(r'''
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
''')
st.markdown("""
Where \( \hat{y}_i \) are the predicted values by the model.
""")

st.subheader("Linear Regression & MSE")
# Sidebar for Configuration
st.sidebar.header('Configure Parameters')
num_features = st.sidebar.slider('Feature count', 1, 10, 1)
num_samples = st.sidebar.slider('Sample count', 1, 100, 10)

# Data Generation
st.subheader("Data Preparation")
st.markdown("""
Synthetic data is generated here based on the specified number of features and samples to demonstrate the model's behavior under controlled conditions.
""")
np.random.seed(42)
X = np.random.random_sample((num_samples, num_features))
y = np.random.random_sample(num_samples)

linreg = LinearRegression().fit(X, y)
intercept = linreg.intercept_
coef = linreg.coef_
y_pred = linreg.predict(X)
mse = mean_squared_error(y, y_pred)
# MSE Calculation and Display
st.subheader("Performance Metrics")
steps = np.array(list(range(-100, 110)))
intercept_new = st.sidebar.select_slider("omega-0", steps / 10 * intercept, intercept, format_func=lambda x: "%.4f" %(x))
coef_new = np.zeros(num_features)
for i in range(num_features):
    coef_new[i] = st.sidebar.select_slider("omega-" + str(i+1), steps / 10 * coef[i], coef[i], format_func=lambda x: "%.4f" %(x))

y_pred_new = np.matmul(X, coef_new.T) + intercept_new
mse_new = mean_squared_error(y, y_pred_new)

if num_features == 1:
    fig = plt.figure()
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.plot([-0.1, 1.1], [-0.1*coef_new[0]+intercept_new, 1.1*coef_new[0]+intercept_new], "green")
    plt.plot([-0.1, 1.1], [-0.1*coef[0]+intercept, 1.1*coef[0]+intercept], c="violet")
    plt.scatter(X[:,0], y)
    st.pyplot(fig)

for i in range(num_features):
    fig2 = plt.figure()
    plt.title("omega-%i" %(i))
    plt.xlabel("")
    plt.ylabel("MSE - MSE_best")
    plt.xlabel("omega - omega_best")
    xx = np.linspace(-3 * coef[i], 3 * coef[i], 100)
    yy = np.array([mean_squared_error(y, np.matmul(X, np.hstack([coef[:i], [a+coef[i]],  coef[i+1:]]).T) + intercept) for a in xx]) - mse
    plt.plot(xx, yy)
    st.pyplot(fig2)

st.write("MSE Violet: ", mse)
st.write("MSE Green: ", mse_new)