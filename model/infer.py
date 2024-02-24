import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


def model_infer(model, test_df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ["Sex", "Embarked"]
    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    target_col = "Survived"

    X_test = test_df.loc[:, cat_cols + num_cols].fillna({"Age": 0})
    X_test_h = pd.get_dummies(X_test, columns=cat_cols)
    y_test = test_df[target_col]

    preds = model.predict(X_test_h)

    print("Accuracy = ", accuracy_score(y_test, preds))
    return preds


if __name__ == "__main__":
    current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(current_file_dir, "data", "test.csv")
    model_path = os.path.join(current_file_dir, "data", "trained_model.sav")

    preds_path = os.path.join(current_file_dir, "data", "predictions.csv")

    test_df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    predictions = pd.DataFrame(model_infer(model, test_df)).to_csv(preds_path)
