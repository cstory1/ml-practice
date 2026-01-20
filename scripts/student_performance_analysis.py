import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump

# import kagglehub

# Download latest version
# path = kagglehub.dataset_download("neurocipher/student-performance")

# print("Path to dataset files:", path)
linear_regression = LinearRegression()

# Download latest version
path = '../data/StudentPerformance.csv'
df = pd.read_csv(path)

X = df[["Hours Studied", "Previous Scores", "Sleep Hours",
        "Sample Question Papers Practiced"]]
y = df["Performance Index"]

model = linear_regression.fit(X, y)
dump(model, '../pipelines/StudentPerformance.joblib')

print(linear_regression.coef_)


class InputData:
    hours_studied: float
    previous_scores: float
    sleep_hours: float
    sample_questions: float


def preprocessor(input_data: InputData):
    data = pd.DataFrame(
        [[input_data.hours_studied, input_data.previous_scores, input_data.sleep_hours, input_data.sample_questions]],
        columns=["Hours Studied", "Previous Scores", "Sleep Hours",
                 "Sample Question Papers Practiced"])
    return data




new_data = InputData()
new_data.hours_studied = 6
new_data.previous_scores = 96
new_data.sleep_hours = 9
new_data.sample_questions = 0

X_pred = preprocessor(new_data)

pred = linear_regression.predict(X_pred)
print(pred)
print("R² score:", linear_regression.score(X, y))

for feature, coef in zip(X.columns, linear_regression.coef_):
    print(f"{feature}: {coef}")
