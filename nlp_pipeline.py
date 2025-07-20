from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

def run_nlp_pipeline(df, text_column):
    # Determine which column holds the labels.
    # For simplicity, assume the dataset has two columns: one text column and one label column.
    label_col = [col for col in df.columns if col != text_column][0]
    
    # Encode target labels
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    num_labels = len(np.unique(df[label_col]))
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    
    # Prepare the dataset in HuggingFace format
    dataset = Dataset.from_pandas(df[[text_column, label_col]])
    
    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column(label_col, "labels")
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Configure training arguments
   training_args = TrainingArguments(
    output_dir="./nlp_model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    disable_tqdm=False,
)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset  # For demo, using train data as eval; use separate validation set when possible.
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate model (dummy evaluation for demo)
    eval_result = trainer.evaluate()
    
    # Display a simple plain-language explanation
    explanation = (f"Trained DistilBERT model for text classification on column '{text_column}'. "
                   f"Number of classes: {num_labels}. Evaluation metrics: {eval_result}")
    
    # Save the model locally for download
    model_save_path = "best_nlp_model"
    model.save_pretrained(model_save_path)
    
    return model_save_path, explanation, le  # Return label encoder in case needed for inference
