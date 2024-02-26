import os

import hydra
import joblib
import pandas as pd
from dvc.api import DVCFileSystem
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier


def model_train(train_df: pd.DataFrame, cfg: DictConfig) -> RandomForestClassifier:
    cat_cols = ["Sex", "Embarked"]
    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    target_col = "Survived"

    X_train = train_df.loc[:, cat_cols + num_cols].fillna({"Age": 0})
    X_train_h = pd.get_dummies(X_train, columns=cat_cols)
    y_train = train_df[target_col]

    model = RandomForestClassifier(
        n_estimators=cfg["model"]["n_estimators"],
        criterion=cfg["model"]["criterion"],
        min_samples_split=cfg["model"]["min_samples_split"],
        min_samples_leaf=cfg["model"]["min_samples_leaf"],
        min_weight_fraction_leaf=cfg["model"]["min_weight_fraction_leaf"],
        max_features=cfg["model"]["max_features"],
        min_impurity_decrease=cfg["model"]["min_impurity_decrease"],
        bootstrap=cfg["model"]["bootstrap"],
        oob_score=cfg["model"]["oob_score"],
        verbose=cfg["model"]["verbose"],
        warm_start=cfg["model"]["warm_start"],
        ccp_alpha=cfg["model"]["ccp_alpha"],
    )
    model.fit(X_train_h, y_train)
    return model


@hydra.main(config_path="./config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(current_file_dir, cfg["data"]["path"], "train.csv")
    model_path = os.path.join(current_file_dir, "model", "trained_model.sav")

    fs = DVCFileSystem(os.path.join(current_file_dir, "data"))
    fs.get_file("/data/train.csv", data_path)

    train_df = pd.read_csv(data_path)
    model = model_train(train_df, cfg)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    main()
