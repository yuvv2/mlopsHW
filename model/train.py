import os

import hydra
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import onnx
import onnxruntime as rt
import pandas as pd
import seaborn as sns
from dvc.api import DVCFileSystem
from mlflow.models import infer_signature
from omegaconf import DictConfig
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import learning_curve


def model_train(train_df: pd.DataFrame, cfg: DictConfig) -> RandomForestClassifier:
    cat_cols = ["Sex", "Embarked"]
    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    target_col = "Survived"

    X_train = train_df.loc[:, cat_cols + num_cols].fillna({"Age": 0})
    X_train_h = pd.get_dummies(X_train, columns=cat_cols, dtype=float)
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
    y_val = model.predict(X_train_h)

    # График 1
    train_size, train_scores, _ = learning_curve(
        model, X_train_h, y_train, train_sizes=[0.3, 0.6, 0.9]
    )

    fig1, _ = plt.subplots()
    plt.plot(train_size, np.mean(train_scores, axis=1))
    fig1 = plt.gcf()

    # График 2
    fig2, _ = plt.subplots()
    sns.heatmap(
        X_train_h.corr(),
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        vmin=-1.0,
        vmax=1.0,
        square=True,
    )
    fig2 = plt.gcf()

    # График 3
    fig3, _ = plt.subplots()
    RocCurveDisplay.from_predictions(y_train, y_val)
    fig3 = plt.gcf()

    # Делаем метрики
    accuracy = accuracy_score(y_train, y_val)
    precision = precision_score(y_train, y_val)
    recall = recall_score(y_train, y_val)
    f1 = f1_score(y_train, y_val)

    mlflow.set_tracking_uri(cfg["train"]["mlflow_server"])
    mlflow.set_experiment(cfg["train"]["experiment_name"])
    with mlflow.start_run():
        # Log the loss metric
        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.log_metric("val_precision", precision)
        mlflow.log_metric("val_recall", recall)
        mlflow.log_metric("val_f1", f1)

        mlflow.log_figure(fig1, "learning_curve.png")
        mlflow.log_figure(fig2, "heatmap.png")
        mlflow.log_figure(fig3, "roc_auc_curve.png")

        mlflow.set_tag("Training process", "Random Forest Classifier for Titanic Dataset")

        # Infer the model signature
        initial_type = [("float_input", FloatTensorType([None, len(X_train_h.columns)]))]
        onx_model = convert_sklearn(model, initial_types=initial_type)

        with open("./model.onnx", "wb") as f:
            f.write(onx_model.SerializeToString())

        onnx_model = onnx.load_model("./model.onnx")
        sess = rt.InferenceSession(
            "./model.onnx",
            providers=rt.get_available_providers(),
        )

        input_name = sess.get_inputs()[0].name
        pred_onx = sess.run(None, {input_name: np.array(X_train_h).astype(np.float32)})[0]

        signature = infer_signature(X_train_h, pred_onx)
        mlflow.onnx.save_model(onnx_model=onnx_model, path="./model", signature=signature)
        mlflow.onnx.log_model(onnx_model, "model", signature=signature)

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
