import pandas as pd
from transformers import XLNetTokenizer, XLNetForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
train_df = pd.read_csv('posts_train.csv')

# Explore the first few records
print(train_df.head())

# Explore the distribution of classes
print(train_df['class_id'].value_counts())

# Load the datasets
train_df = pd.read_csv('posts_train.csv')
val_df = pd.read_csv('posts_val.csv')
test_df = pd.read_csv('posts_test.csv')

# Load the tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# Tokenize your data
train_encodings = tokenizer(train_df['post'].tolist(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_df['post'].tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_df['post'].tolist(), truncation=True, padding=True, max_length=512)

#Labels
train_labels = train_df['class_id'].values
val_labels = val_df['class_id'].values
test_labels = test_df['class_id'].values

class RedditDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the dataset objects
train_dataset = RedditDataset(train_encodings, train_labels)
val_dataset = RedditDataset(val_encodings, val_labels)
test_dataset = RedditDataset(test_encodings, test_labels)

# Load pre-trained XLNet model with a classification head
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=6)  # We have 6 classes

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_steps=10_000,
    save_total_limit=2,
    remove_unused_columns=False,
    output_dir='./results',
    logging_dir='./logs',
    logging_steps=200,
    gradient_accumulation_steps=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

#evaluate
eval_results = trainer.evaluate()
test_results = trainer.evaluate(test_dataset)
print(test_results)

predictions = trainer.predict(test_dataset)
y_true = test_dataset.labels  
y_pred = np.argmax(predictions.predictions, axis=1)

# Get overall precision, recall, F1 score, and accuracy
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Accuracy: {accuracy:.2f}')


# Get per-class precision, recall, and F1 score
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

# Compute per-class accuracy
unique_labels = np.unique(y_true)
per_class_accuracy = []

for label in unique_labels:
    mask = (y_true == label)
    class_true = y_true[mask]
    class_pred = y_pred[mask]
    class_accuracy = accuracy_score(class_true, class_pred)
    per_class_accuracy.append(class_accuracy)

# Print or otherwise display the per-class metrics and accuracy
for i, (p, r, f, s, a) in enumerate(zip(precision, recall, f1, support, per_class_accuracy)):
    print(f'Class {i}:')
    print(f'  Precision: {p:.2f}')
    print(f'  Recall: {r:.2f}')
    print(f'  F1 Score: {f:.2f}')
    print(f'  Support: {s}')
    print(f'  Accuracy: {a:.2f}')


test_df['predicted_labels'] = y_pred

#correctl classified
correctly_classified = test_df[test_df['class_id'] == test_df['predicted_labels']]
print(correctly_classified.sample(5))

#not correctly classified
incorrectly_classified = test_df[test_df['class_id'] != test_df['predicted_labels']]
print(incorrectly_classified.sample(5))

# For training dataset:
print(train_df['class_id'].value_counts())

# For validation dataset:
print(val_df['class_id'].value_counts())

# For testing dataset:
print(test_df['class_id'].value_counts())

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
labels = ['adhd', 'anxiety', 'bipolar', 'depression', 'ptsd', 'none']

# Plot confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()