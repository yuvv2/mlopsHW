import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import learning_curve


def get_lr(model: RandomForestClassifier, X_train_h, y_train: pd.DataFrame):
    train_size, train_scores, _ = learning_curve(
        model, X_train_h, y_train, train_sizes=[0.3, 0.6, 0.9]
    )
    fig, _ = plt.subplots()
    plt.plot(train_size, np.mean(train_scores, axis=1))
    return fig


def get_hm(X_train_h: pd.DataFrame):
    fig, _ = plt.subplots()
    sns.heatmap(
        X_train_h.corr(),
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        vmin=-1.0,
        vmax=1.0,
        square=True,
    )
    return fig


def get_roc(y_train, y_val: pd.DataFrame):
    RocCurveDisplay.from_predictions(y_train, y_val)
    return plt.gcf()
