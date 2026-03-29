import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from feature_extraction import create_advanced_features
from models import RidgeRegression, LassoRegression

# Load data
df = pd.read_csv("data/processed/cleaned_traffic.csv")

# Rename column properly
if 'Vehicles' in df.columns:
    df.rename(columns={'Vehicles': 'traffic'}, inplace=True)

# After feature engineering
df = create_advanced_features(df)

df = df.dropna()  

# Drop non-numeric columns
for col in ['DateTime', 'Junction', 'ID']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Prepare data
X = df.drop(columns=['traffic'])
y = df['traffic']

X = X.select_dtypes(include=['number'])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize models
ridge = RidgeRegression()
lasso = LassoRegression()
# Models (NO .values)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

# Evaluation
print("Ridge MSE:", mean_squared_error(y_test, ridge_pred))
print("Lasso MSE:", mean_squared_error(y_test, lasso_pred))
print("Ridge Weights:", ridge.W)
print("Lasso Weights:", lasso.W)
print("\nDifference (Ridge - Lasso):",
      mean_squared_error(y_test, ridge_pred) - mean_squared_error(y_test, lasso_pred))
# Feature importance
def plot_weights(model, feature_names, title):
    plt.bar(feature_names, model.W)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

plot_weights(ridge, X.columns, "Ridge Importance")
plot_weights(lasso, X.columns, "Lasso Importance")
if mean_squared_error(y_test, lasso_pred) < mean_squared_error(y_test, ridge_pred):
    print("\nLasso performed better → indicates irrelevant features were removed.")
