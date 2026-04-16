import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
import os

CONFIG = {
    'model_path': 'your_model_path_here',
    'train_file': './processed_data/dataset_5class_train_balanced.csv',
    'test_file': './processed_data/dataset_5class_test_balanced.csv',
    'batch_size': 8,
    'max_length': 128,
    'epochs': 100,
    'lr': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 4,
    'valid_classes': ["6000019", "1000005", "1000004", "0"],
    'features': ['speed', 'pressure', 'x_acceleration', 'y_acceleration', 'z_acceleration', 'temperature_in_lift']
}

class ElevatorJsonDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length, encoder=None, is_train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(csv_file)
        df[CONFIG['features']] = df[CONFIG['features']].fillna(0)
        df['alarm_type'] = df['alarm_type'].astype(str)
        df = df[df['alarm_type'].isin(CONFIG['valid_classes'])].copy()

        self.texts = []
        for _, row in df.iterrows():
            data_dict = {feat: row[feat] for feat in CONFIG['features']}
            json_str = json.dumps(data_dict, ensure_ascii=False)

            prompt = (
                f"Below is the sensor data of an elevator in JSON format.\n"
                f"Data: {json_str}\n"
                f"Please analyze the acceleration and pressure patterns to classify the alarm type.\n"
                f"Response:"
            )
            self.texts.append(prompt)

        if is_train:
            self.encoder = LabelEncoder()
            self.labels = self.encoder.fit_transform(df['alarm_type'].values)
        else:
            if encoder is None:
                raise ValueError("Encoder needed for test")
            self.encoder = encoder
            self.labels = self.encoder.transform(df['alarm_type'].values)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

class QwenLoRAClassification(nn.Module):
    def __init__(self, model_path, num_classes):
        super().__init__()
        print(f"Loading Qwen model from {model_path}...")

        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=CONFIG['device']
        )

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        self.qwen = get_peft_model(self.qwen, peft_config)
        self.qwen.print_trainable_parameters()

        hidden_size = self.qwen.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden_state = outputs.hidden_states[-1]

        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        sentence_embeddings = last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]

        sentence_embeddings = sentence_embeddings.to(torch.float32)

        logits = self.classifier(sentence_embeddings)
        return logits

def train():
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_path'],
        trust_remote_code=True,
        padding_side='right'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Preparing JSON Datasets...")
    train_dataset = ElevatorJsonDataset(CONFIG['train_file'], tokenizer, CONFIG['max_length'], is_train=True)
    if os.path.exists(CONFIG['test_file']):
        test_dataset = ElevatorJsonDataset(CONFIG['test_file'], tokenizer, CONFIG['max_length'],
                                           encoder=train_dataset.encoder, is_train=False)
    else:
        test_dataset = None
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    print(f"Classes map: {dict(zip(range(len(train_dataset.encoder.classes_)), train_dataset.encoder.classes_))}")

    model = QwenLoRAClassification(CONFIG['model_path'], CONFIG['num_classes'])
    model.to(CONFIG['device'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])

    criterion = nn.CrossEntropyLoss()

    print("Start Training...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | Train Loss: {avg_loss:.4f}")

        if test_dataset:
            evaluate(model, test_loader, train_dataset.encoder)

def evaluate(model, dataloader, encoder):
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])

            logits = model(input_ids, attention_mask)
            predicted_ids = torch.argmax(logits, dim=1)

            preds.extend(predicted_ids.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1_macro = f1_score(true_labels, preds, average='macro')
    f1_weighted = f1_score(true_labels, preds, average='weighted')

    print(f"Test Accuracy:    {acc:.4f}")
    print(f"Test Macro F1:    {f1_macro:.4f}")
    print(f"Test Weighted F1: {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=encoder.classes_))

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA not found. Running Qwen on CPU will be extremely slow.")

    train()