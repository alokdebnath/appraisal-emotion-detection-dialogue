import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn import Linear, MSELoss, DataParallel
from transformers.optimization import AdamW
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from tqdm import tqdm
from time import sleep
import argparse
import random

# Define a custom dataset for regression
class RegressionDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {'sentence': self.sentences[idx], 'label': torch.tensor(self.labels[idx], dtype=torch.float)}

# Function to train the regression model
def train_regression_model(model, tokenizer, dim, train_dataloader,
                           val_dataloader, num_epochs=10, learning_rate=3e-5,
                           device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    # Define loss function and optimizer
    criterion = MSELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        print(f"{'='*80} \n Epoch {epoch + 1}/{num_epochs} \n {'='*80} ")
        train_batch_ix = 0
        val_batch_ix = 0

        # for batch in train_dataloader:
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                train_batch_ix += 1
                model.train()

                # Move input and target to the GPU if available
                inputs = tokenizer(batch['sentence'], return_tensors='pt', truncation=True, padding=True)
                inputs.to(device)
                targets = batch['label'].unsqueeze(1).to(inputs['input_ids'].device)
                optimizer.zero_grad()

                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    # Compute the loss
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Print the loss for each batch
                tepoch.set_postfix(dim=dim, train_loss=loss.item())
        print('-'*80)

        with tqdm(val_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                val_batch_ix += 1
                model.eval()
                with torch.no_grad():
                    inputs = tokenizer(batch['sentence'], return_tensors='pt', truncation=True, padding=True)
                    inputs.to(device)
                    targets = batch['label'].unsqueeze(1).to(inputs['input_ids'].device)

                    outputs = model(**inputs)
                    logits = outputs.logits

                    # Compute the loss
                    loss = criterion(logits, targets)
                    logits = logits.detach().cpu().tolist()
                    targets = targets.detach().cpu().tolist()

                    mse = mean_squared_error(targets, logits)
                    mae = mean_absolute_error(targets, logits)
                    evs = explained_variance_score(targets, logits)
                    r2 = r2_score(targets, logits)

                tepoch.set_postfix(dim=dim, val_loss=loss.item(), mse=mse, mae=mae, evs=evs, r2=r2)

        # Clear CUDA Cache
        del mse, mae, evs, r2
        torch.cuda.empty_cache()

# # Example usage for training

# # Assume you have a list of sentences and corresponding regression labels
# train_sentences = ["This is a sample sentence.", "Another sentence for training."]
# train_labels = [0.5, 1.2]

def dataset_creator(filepath):
    df = pd.read_csv(filepath, sep='\t', header=0)
    columns = df.columns.tolist()
    sentences = df['generated_text'].tolist()
    emotion_categorical_labels = df['emotion'].tolist()
    emotion_dimensional_labels = [df[col].tolist() for col in columns[6:18]]
    event_metadata = [df[col].tolist() for col in columns[19:23]]
    appraisal_dimension_labels = [df[col].tolist() for col in columns[24:45]]
    return columns, sentences, emotion_categorical_labels, emotion_dimensional_labels, event_metadata, appraisal_dimension_labels


if __name__ == '__main__':
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    print("Using: " + str(device))

    seed = 5186312
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='roberta-large')
    parser.add_argument('-t', '--train_path', default='./data/train.csv')
    parser.add_argument('-e', '--eval_path', default='./data/test.csv')
    parser.add_argument('-v', '--val_path', default='./data/val.csv')
    parser.add_argument('-s', '--save_path', default='./models/large_')
    args = parser.parse_args()

    trainpath = args.train_path
    columns, train_sentences, train_emotion_categorical_labels, train_emotion_dimensional_labels, train_event_metadata, train_appraisal_dimension_labels = dataset_creator(trainpath)

    # testpath = args.eval_path
    # _, test_sentences, test_emotion_categorical_labels, test_emotion_dimensional_labels, test_event_metadata, test_appraisal_dimension_labels = dataset_creator(testpath)

    valpath = args.val_path
    _, val_sentences, val_emotion_categorical_labels, val_emotion_dimensional_labels, val_event_metadata, val_appraisal_dimension_labels = dataset_creator(valpath)

    # Create a RegressionDataset and DataLoader

    for i in range(21):
        dim = columns[24 + i]
        print(f"{'-'*20}> Training: {dim} <{'-'*20}")
        train_dataset = RegressionDataset(train_sentences, train_appraisal_dimension_labels[i])
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        # test_dataset = RegressionDataset(test_sentences, test_appraisal_dimension_labels[i])
        # test_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        val_dataset = RegressionDataset(val_sentences, val_appraisal_dimension_labels[i])
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

        # Load pre-trained RoBERTa model and tokenizer
        model_name = args.model_name
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=1)
        model.to(device)

        # print(model)
        if torch.cuda.device_count()  >  1:
            model = DataParallel(model)

        # Train the regression model
        train_regression_model(model, tokenizer, dim, train_dataloader, val_dataloader)

        # Save the trained model
        print(f"{'-'*20}> Saving: {dim} <{'-'*20}")
        if torch.cuda.device_count() > 1:
            model.module.save_pretrained(str(args.save_path) + dim + '/')
        else:
            model.save_pretrained(str(args.save_path) + dim + '/')
        print(f"{'+'* 80}")
