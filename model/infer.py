import os

import hydra
import joblib
import pandas as pd
from dvc.api import DVCFileSystem
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, roc_auc_score


def model_infer(model, test_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    cat_cols = ["Sex", "Embarked"]
    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    target_col = "Survived"

    X_test = test_df.loc[:, cat_cols + num_cols].fillna({"Age": 0})
    X_test_h = pd.get_dummies(X_test, columns=cat_cols)
    y_test = test_df[target_col]

    preds = model.predict(X_test_h)

    print("Accuracy = ", accuracy_score(y_test, preds))
    print("Roc-Auc Score = ", roc_auc_score(y_test, preds))
    return preds


@hydra.main(config_path="./config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(current_file_dir, cfg["data"]["path"], "test.csv")
    model_path = os.path.join(current_file_dir, "model", "trained_model.sav")
    preds_path = os.path.join(current_file_dir, "data", "predictions.csv")

    fs = DVCFileSystem(os.path.join(current_file_dir, "data"))
    fs.get_file("/data/test.csv", data_path)

    test_df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    pd.DataFrame(model_infer(model, test_df, cfg)).to_csv(preds_path)


if __name__ == "__main__":
    main()
