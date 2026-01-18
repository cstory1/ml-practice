import pandas as pd
from sklearn.linear_model import LinearRegression
# import kagglehub

# Download latest version
# path = kagglehub.dataset_download("neurocipher/student-performance")

# print("Path to dataset files:", path)
linear_regression = LinearRegression()

# Download latest version
path = 'StudentPerformance.csv'
df = pd.read_csv(path)

X = df[["Hours Studied", "Previous Scores", "Sleep Hours",
        "Sample Question Papers Practiced"]]
y = df["Performance Index"]


linear_regression.fit(X, y)

print(linear_regression.coef_)
X_pred = pd.DataFrame(
    [[6, 96, 9, 0]],
    columns=X.columns
)
pred = linear_regression.predict(X_pred)
print(pred)
print("R² score:", linear_regression.score(X, y))

for feature, coef in zip(X.columns, linear_regression.coef_):
    print(f"{feature}: {coef}")
