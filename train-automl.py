#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import streamlit as st
import shutil
import pandas as pd
import mercury as mr
from supervised.automl import AutoML 
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from IPython.display import Image, display
import torch


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


app = mr.App(title="Train AutoML 🧑‍💻", description="Train ML pipeline with MLJAR AutoML")


# # Train Machine Learning Pipeline with MLJAR AutoML
# Please follow the steps:
# 1. Upload CSV file with data. Data should heave column names in the first line.
# 2. Select input features and target column.
# 3. Select AutoML training mode, algorithms and training time limit.
# 4. Directory with all ML models will be zipped and available to download.

# In[4]:


data_file = mr.File(label="Upload CSV with training data", max_file_size="1MB")


# In[ ]:


data_file.filename


# In[ ]:


if data_file.filename is None:
    st.error("Stopping app due to issue")
    st.stop()

# Load Data
if data_file.filename.endswith(".csv"):
    df = pd.read_csv(data_file.filepath)
elif data_file.filename.endswith((".xls", ".xlsx")):
    df = pd.read_excel(data_file.filepath, engine='openpyxl')
else:
    raise ValueError("Unsupported file format. Please upload .csv or .xlsx")

mr.Markdown("### Uploaded Data")
df.head()


# In[ ]:


def detect_data_type(df):
    # Prefer "text" column explicitly if present
    if 'text' in df.columns:
        return "nlp", "text"
    # Otherwise check if any object column has avg length > 20
    for col in df.columns:
        if df[col].dtype == object:
            avg_len = df[col].dropna().map(lambda x: len(str(x))).mean()
            if avg_len and avg_len > 20:
                return "nlp", col
    return "tabular", None

data_type, text_column = detect_data_type(df)


# In[ ]:


if data_type == "nlp":
    label_column = [col for col in df.columns if col != text_column][0]
    df = df[[text_column, label_column]].dropna()

    # Encode labels
    le = LabelEncoder()
    df[label_column] = le.fit_transform(df[label_column])

    # Tokenize and prepare dataset
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Split train/test
    train_test = tokenized_dataset.train_test_split(test_size=0.2)
    train_ds = train_test["train"]
    test_ds = train_test["test"]

    # Model setup
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(set(df[label_column]))
    )

    training_args = TrainingArguments(
    output_dir="./nlp_model",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    trainer.train()

    # Evaluation
    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    acc = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, output_dict=True)
    class_names = le.classes_

    mr.Markdown("### NLP Classification Report")
    mr.Markdown(f"- Accuracy: **{acc:.4f}**")
    mr.Markdown(f"- Labels: {list(class_names)}")

    fig, ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_nlp.png")
    mr.Markdown("![Confusion Matrix](confusion_matrix_nlp.png)")

    # Save model + label encoder
    model_path = "best_model_nlp"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    joblib.dump(le, os.path.join(model_path, "label_encoder.pkl"))

    shutil.make_archive(model_path, "zip", model_path)
    # mr.File(label="Download Best NLP Model (.zip)", path=model_path + ".zip")
# ─────────────────────────────────────────────────────────────────────────────
# TABULAR CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

elif data_type == "tabular":
    x_columns = mr.MultiSelect(label="Select input features", 
                               value=list(df.columns)[:-1], choices=list(df.columns))
    y_column = mr.Select(label="Select target column", 
                         value=list(df.columns)[-1], choices=list(df.columns))

    if x_columns.value is None or len(x_columns.value) == 0 or y_column.value is None:
        mr.Markdown("Please select input features and target column.")
        st.error("Stopping app due to issue")
        st.stop()

    # Mode and Algo Selection
    mode = mr.Select(label="AutoML Mode", value="Explain", choices=["Explain", "Perform", "Compete"])
    algos = {
        "Explain": ["Baseline", "Linear", "Decision Tree", "Random Forest", "Xgboost", "Neural Network"],
        "Perform": ["Linear", "Random Forest", "LightGBM", "Xgboost", "CatBoost", "Neural Network"],
        "Compete": ["Decision Tree", "Random Forest", "Extra Trees", "LightGBM", 
                    "Xgboost", "CatBoost", "Neural Network", "Nearest Neighbors"]
    }
    algorithms = mr.MultiSelect(label="Algorithms", value=algos[mode.value], choices=algos[mode.value])
    time_limit = mr.Select(label="Time limit (seconds)", value="60", choices=["60", "120", "240", "300"])
    start_training = mr.Button(label="Start training", style="success")
    output_dir = mr.OutputDir()

    automl = AutoML(mode=mode.value, algorithms=algorithms.value,
                    total_time_limit=int(time_limit.value))

    if start_training.clicked:
        mr.Markdown("### AutoML Training Logs")
        automl.fit(df[x_columns.value], df[y_column.value])

        # Save and zip model
        output_filename = os.path.join(output_dir.path, automl._results_path)
        shutil.make_archive(output_filename, 'zip', automl._results_path)
        # mr.File(label="Download Best Tabular Model (.zip)", file=output_filename + ".zip")


        # SHAP Explanation (only for tree-based models)
        try:
            X_sample = df[x_columns.value].sample(100)
            explainer = shap.Explainer(automl._best_model.model, X_sample)
            shap_values = explainer(X_sample)
            fig = shap.plots.beeswarm(shap_values, show=False)
            plt.tight_layout()
            plt.savefig("shap_plot.png")
            mr.Image("shap_plot.png")
        except Exception as e:
            mr.Markdown(f"SHAP Explanation skipped: {e}")

        # Metrics
        X_train, X_test, y_train, y_test = train_test_split(
            df[x_columns.value], df[y_column.value], test_size=0.2, random_state=42)
        y_pred = automl.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mr.Markdown(f"### Accuracy of best model: **{acc:.4f}**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6,6))
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", values_format='d')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix_tabular.png")
        display(Image(filename="confusion_matrix_tabular.png"))

        automl.report()

