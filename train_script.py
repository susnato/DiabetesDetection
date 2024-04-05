import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost
import datetime

SAVE_PATH = "./models/"
MODEL_PARAMS = {
    'n_estimators': 1000,
    'objective': "binary:logistic",
    'max_depth': 4,
    'learning_rate': 0.004,
    'subsample': 0.5,
    'device': "cuda",
    'eval_metric': "auc",
}

def train(train_data, eval_data, feature_cols, label_col):
    time_now = datetime.datetime.now()
    model_name = f"{time_now.year}-{time_now.month}-{time_now.day}-{time_now.hour}-{time_now.minute}-{time_now.second}"
    model_path = os.path.join(SAVE_PATH, f"{model_name}.json")

    model = xgboost.XGBClassifier(**MODEL_PARAMS)

    # train
    model.fit(train_data[feature_cols], train_data[label_col],
              eval_set = [(train_data[feature_cols], train_data[label_col]),
                          (eval_data[feature_cols], eval_data[label_col])],
              verbose=True,
              )

    # save
    model.save_model(model_path)
    print(f"Model saved at: {model_path}")

    return model_path

def eval(model_path, data, feature_cols, label_col):
    print("Running final evaluation:")

    y_true = data[label_col]
    eval_data = data[feature_cols]

    model = xgboost.XGBClassifier()
    model.load_model(model_path)
    print("Evaluation model loaded!")

    eval_results = model.predict_proba(eval_data)[:, 1]
    # print([(v1, v2) for v1, v2 in zip(y_true, eval_results)])

    print(f"Eval AUC-ROC score: {roc_auc_score(y_true=y_true, y_score=eval_results)}")

def main():
    data = pd.read_csv("./data/diabetes.csv")
    train_data, eval_data = train_test_split(data, train_size=0.8, stratify=data["Outcome"], random_state=42)
    feature_cols = list(set(train_data.columns) - set(["Outcome"]))
    label_col = "Outcome"

    print(f"Train data size: {train_data.shape[0]}")
    print(f"Test data size: {eval_data.shape[0]}")
    print(f"Feature columns: {feature_cols}")

    model_path = train(train_data, eval_data, feature_cols, label_col)
    eval_results = eval(model_path, eval_data, feature_cols, label_col)

if __name__ == "__main__":
    main()

