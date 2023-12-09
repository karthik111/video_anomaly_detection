import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 3D polynomial regression
x = 7 * np.random.rand(100, 1) - 2.8
y = 7 * np.random.rand(100, 1) - 2.8
#y = 3*x**2+2*x-4

z = x**2 + y**2 + 0.2*x + 0.2*y + 0.1*x*y +2 + np.random.randn(100, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(r2_score(y_test, y_pred))

plt.plot(x_train, lr.predict(x_train), color="r")
plt.plot(x, y, "b.")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#applying polynomial regression degree 2
poly = PolynomialFeatures(degree=2, include_bias=True)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)
#include bias parameter
lr = LinearRegression()
lr.fit(x_train_trans, y_train)
y_pred = lr.predict(x_test_trans)
print(r2_score(y_test, y_pred))

plt.plot(y_test, lr.predict(x_test_trans), color="r")
plt.plot(x, y, "b.")
plt.xlabel("X")
plt.ylabel("Y")

plt.plot(x, y, "b.")
plt.xlabel("X")
plt.ylabel("Y")''
plt.show()

def polynomial_regression(degree):
    X_new=np.linspace(-3, 3, 100).reshape(100, 1)
    X_new_poly = poly.transform(X_new)
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ])
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    #plotting prediction line
    plt.plot(X_new, y_newbig,'r', label="Degree " + str(degree), linewidth=2)
    plt.plot(x_train, y_train, "b.", linewidth=3)
    plt.plot(x_test, y_test, "g.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.axis([-3, 3, 0, 10])
    plt.show()

polynomial_regression(25)



x = np.array([50,25,35,70,45,65,80])
x = np.array([50,25,35,70,45,80])
x = x[:, np.newaxis]

y = np.array([356,213,295,366,280,334,412])
y = np.array([356,213,295,366,280,412])
y = y[:, np.newaxis]

z = np.array([1.51,1.33,1.42,2.23,2.1,3.02,2.35])
z = np.array([1.51,1.33,1.42,2.23,2.1,2.35])
z = z[:, np.newaxis]
#
#
# x = np.array([50,25,35, 70,25.534,35.5757, 70.34])
# x = x[:, np.newaxis]
#
# y = np.array([356,213,295, 366,213.44,296, 367])
# y = y[:, np.newaxis]
#
# z = np.array([1.51,1.33,1.42, 2.23,1.43,1.52, 2.43])
# z = z[:, np.newaxis]

# import plotly.express as px
# df = px.data.iris()
# fig = px.scatter_3d(df, x=x.ravel(), y=y.ravel(), z=z.ravel())
# fig.show()


# 3D polynomial regression
# x = 7 * np.random.rand(10, 1) - 2.8
# y = 7 * np.random.rand(10, 1) - 2.8
# z = x**2 + y**2 + 0.2*x + 0.2*y + 0.1*x*y +2 + np.random.randn(10, 1)
#
# # 3D polynomial regression
# x = 7 * np.random.rand(100, 1) - 2.8
# y = 7 * np.random.rand(100, 1) - 2.8
# z = x**2 + y**2 + 0.2*x + 0.2*y + 0.1*x*y +2 + np.random.randn(100, 1)

x = np.concatenate((x,y),axis=1)
y = z

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(x=x[:,0].tolist(), y=x[:,1].tolist(), z=np.concatenate(y).tolist(), mode='markers')])
# Update layout for better visibility
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
# Show the plot
fig.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(f"R2 score with Linear Regression: {r2_score(y_test, y_pred)}")

#plt.plot(x_train, lr.predict(x_train), color="r")
plt.plot(x, y, "b.")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#applying polynomial regression degree 2
poly = PolynomialFeatures(degree=7, include_bias=True)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)
#include bias parameter
lr = LinearRegression()
lr.fit(x_train_trans, y_train)
y_pred = lr.predict(x_test_trans)
print(f"R2 score with Polynomial Regression: {r2_score(y_test, y_pred)}")

plt.plot(y_test, lr.predict(x_test_trans), color="r")
plt.plot(x, y, "b.")
plt.xlabel("X")
plt.ylabel("Y")

#z = x**2 + y**2 + 0.2*x + 0.2*y + 0.1*x*y +2 + np.random.randn(100, 1)

# t (minutes) | x (grams) | y (cm^2)
# 1.51       | 50        | 356
# 1.33       | 25        | 213
# 1.42       | 35        | 295
# 2.23       | 70        | 366
# 2.10       | 45        | 280
# 3.02       | 65        | 334
# 2.35       | 80        | 412


import numpy as np
from scipy.optimize import minimize

# Define the negative of the objective function (since we want to maximize)
def negative_objective(params):
    x, y = params
    return  -1.33928396 * 10**(-6) + 1.31016922 * 10**(-6) - 7.51382531 * 10**(-4) * x**2 - 9.92306438 * 10**(-4) * x * y + 1.86741515 * 10**(-4) * y**2 - 1.89103553 * 10**(-3) * x**3 - 1.20576315 * 10**(-4) * y**2 * x + 8.48721351 * 10**(-4) * y * x**2 + 5.37412823 * 10**(-6) * y**3

# Initial guess
initial_guess = [25, 212]
# Try different optimization methods
methods = ['Powell', 'L-BFGS-B', 'COBYLA']

bounds = [(10, 100), (40, 250)]

for method in methods:
    print(f"\nOptimizing with method: {method}")

    # Perform the optimization
    result = minimize(negative_objective, initial_guess, method=method, bounds=bounds)

    # Extract optimal values
    optimal_x, optimal_y = result.x
    optimal_z = result.fun  # Convert back to the original function value

    # Print the optimal values
    print(f"Optimal x: {optimal_x}")
    print(f"Optimal y: {optimal_y}")
    print(f"Optimal z: {optimal_z}")



# Define the function t
def t(x, y):
    return (-1.33928396e-6 + 1.31016922e-6 - 7.51382531e-4 * x**2 -
            9.92306438e-4 * x * y + 1.86741515e-4 * y**2 -
            1.89103553e-3 * x**3 - 1.20576315e-4 * y**2 * x +
            8.48721351e-4 * y * x**2 + 5.37412823e-6 * y**3)

# Define the search ranges
x_lb, x_ub = 25, 80
y_lb, y_ub = 213, 412
t_lb, t_ub = 0, 2.35
step = 0.01

# Initialize variables
xmin, ymin, tmin = x_lb, y_lb, t(x_lb, y_lb)

# Brute-force search
for x in range(int(x_lb / step), int(x_ub / step) + 1):
    for y in range(int(y_lb / step), int(y_ub / step) + 1):
        # Map indices back to actual values
        x_val, y_val = x * step, y * step

        # Evaluate the function
        tval = t(x_val, y_val)

        # Check constraints
        if t_lb <= tval <= t_ub:
            # Update minimum if the newly evaluated t is within the specified range
            if tval < tmin:
                tmin = tval
                # Save current x and y where t is minimum
                xmin, ymin = x_val, y_val

# Display the result
print(f'Minimum value of t: {tmin:.5f}')
print(f'Optimal values - x: {xmin:.5f}, y: {ymin:.5f}')
