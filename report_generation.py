import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from matplotlib.backends.backend_pdf import PdfPages


# Loading the datset data here
import pandas as pd
df = pd.read_csv("src\Supermart_cleaned_dataset.csv")


# Splitting the dataset
X = df.drop(['Sales', 'Profit','Discount_level'], axis=1)
y = df[['Sales', 'Profit']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# creating a mutioutput regressor model
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42))
]

stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression()
)

multi_stacked = MultiOutputRegressor(stacked_model)
multi_stacked.fit(X_train, y_train)

ensemble_pred = multi_stacked.predict(X_test)

# Creating residuals
residuals_sales = y_test['Sales'].values - ensemble_pred[:, 0]
residuals_profit = y_test['Profit'].values - ensemble_pred[:, 1]

# Creating PDF
with PdfPages("model_evaluation_report.pdf") as pdf:

    # 1. Actual vs Predicted Sales
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test['Sales'], y=ensemble_pred[:, 0], color='skyblue')
    plt.plot([y_test['Sales'].min(), y_test['Sales'].max()],
             [y_test['Sales'].min(), y_test['Sales'].max()],
             'r--', label='Perfect Prediction Line')
    plt.title("Actual vs Predicted Sales")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 2. Actual vs Predicted Profit
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test['Profit'], y=ensemble_pred[:, 1], color='lightgreen')
    plt.plot([y_test['Profit'].min(), y_test['Profit'].max()],
             [y_test['Profit'].min(), y_test['Profit'].max()],
             'r--', label='Perfect Prediction Line')
    plt.title("Actual vs Predicted Profit")
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 3. Line Plot (First 100 Samples)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y_test['Sales'].values[:100], label='Actual Sales', marker='o')
    plt.plot(ensemble_pred[:, 0][:100], label='Predicted Sales', marker='x')
    plt.title("Line Plot: Actual vs Predicted Sales")
    plt.xlabel("Sample Index")
    plt.ylabel("Sales")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_test['Profit'].values[:100], label='Actual Profit', marker='o')
    plt.plot(ensemble_pred[:, 1][:100], label='Predicted Profit', marker='x')
    plt.title("Line Plot: Actual vs Predicted Profit")
    plt.xlabel("Sample Index")
    plt.ylabel("Profit")
    plt.legend()

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 4. Residual Distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals_sales, kde=True, color='skyblue')
    plt.title("Residual Distribution: Sales")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.histplot(residuals_profit, kde=True, color='lightgreen')
    plt.title("Residual Distribution: Profit")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("✅ PDF report saved as: model_evaluation_report.pdf")