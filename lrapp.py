import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


st.subheader("Linear Regression & MSE")

num_features = st.sidebar.slider('Feature count', 1, 10, 1)
num_samples = st.sidebar.slider('Sample count', 1, 100, 10)

np.random.seed(42)
X = np.random.random_sample((num_samples, num_features))
y = np.random.random_sample(num_samples)

linreg = LinearRegression().fit(X, y)
intercept = linreg.intercept_
coef = linreg.coef_
y_pred = linreg.predict(X)
mse = mean_squared_error(y, y_pred)

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