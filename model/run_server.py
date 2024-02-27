import os

import hydra
import pandas as pd
import requests
from dvc.api import DVCFileSystem
from omegaconf import DictConfig


@hydra.main(config_path="./config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(current_file_dir, cfg["data"]["path"], "test.csv")

    if os.path.isfile(data_path):
        os.remove(data_path)

    fs = DVCFileSystem(os.path.join(current_file_dir, "data"))
    fs.get_file("/data/test.csv", data_path)
    test_df = pd.read_csv(data_path)

    cat_cols = ["Sex", "Embarked"]
    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

    X_test = test_df.loc[:, cat_cols + num_cols].fillna({"Age": 0})
    X_test_h = pd.get_dummies(X_test, columns=cat_cols, dtype=float)
    url = f"{cfg['infer']['mlflow_server']}/invocations"

    json_data = {"dataframe_split": X_test_h[:6].to_dict(orient="split")}
    response = requests.post(url, json=json_data)

    print("Predictions: ", response.json())


if __name__ == "__main__":
    main()
