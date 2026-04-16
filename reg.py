import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import gc
import warnings

warnings.filterwarnings('ignore')

CONFIG = {
    'gpu_id': 0,
    'model_path': 'your_model_path_here',
    'data_file': 'predictive-maintenance-dataset.csv',
    'batch_size': 32,
    'grad_accumulation_steps': 32,
    'max_length': 256,
    'epochs': 100,
    'lr': 2e-4,
    'weight_decay': 0.01,
    'features': ['revolutions', 'humidity', 'x1', 'x2', 'x3', 'x4', 'x5'],
    'target': 'vibration'
}

device_str = f"cuda:{CONFIG['gpu_id']}" if torch.cuda.is_available() else "cpu"
CONFIG['device'] = device_str
print(f"Running on device: {CONFIG['device']}")

torch.cuda.empty_cache()
gc.collect()

class MaintenanceRegressionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df = df.reset_index(drop=True)

        self.texts = []
        self.labels = []

        for _, row in self.df.iterrows():
            prompt = (
                f"Predictive Maintenance Sensor Data:\n"
                f"- Revolutions: {row['revolutions']:.4f}\n"
                f"- Humidity: {row['humidity']:.4f}\n"
                f"- Sensor X1: {row['x1']:.4f}\n"
                f"- Sensor X2: {row['x2']:.4f}\n"
                f"- Sensor X3: {row['x3']:.4f}\n"
                f"- Sensor X4: {row['x4']:.4f}\n"
                f"- Sensor X5: {row['x5']:.4f}\n"
                f"Task: Predict the vibration value based on the sensor readings.\n"
                f"Vibration Prediction:"
            )
            self.texts.append(prompt)
            self.labels.append(float(row[CONFIG['target']]))

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
            'label': torch.tensor(label, dtype=torch.float32)
        }

class QwenLoRARegressor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        print(f"Loading Qwen model from {model_path}...")

        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=CONFIG['device']
        )

        self.qwen.gradient_checkpointing_enable()
        self.qwen.enable_input_require_grads()

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.qwen = get_peft_model(self.qwen, peft_config)
        self.qwen.print_trainable_parameters()

        hidden_size = self.qwen.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
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
        embedding = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]

        embedding = embedding.to(torch.float32)

        output = self.regressor(embedding)
        return output.squeeze(-1)

def train():
    print("Loading data...")
    df = pd.read_csv(CONFIG['data_file'])

    initial_len = len(df)
    df = df.dropna(subset=[CONFIG['target']])
    print(f"Dropped {initial_len - len(df)} rows (NaN target). Final size: {len(df)}")

    train_size = int(0.8 * len(df))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_path'], trust_remote_code=True, padding_side='right')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    train_ds = MaintenanceRegressionDataset(train_df, tokenizer, CONFIG['max_length'])
    test_ds = MaintenanceRegressionDataset(test_df, tokenizer, CONFIG['max_length'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    model = QwenLoRARegressor(CONFIG['model_path'])
    model.to(CONFIG['device'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

    criterion = nn.MSELoss()

    print("Start Regression Training...")
    best_rmse = float('inf')

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])

            preds = model(input_ids, attention_mask)

            loss = criterion(preds, labels)

            loss = loss / CONFIG['grad_accumulation_steps']
            loss.backward()

            if (step + 1) % CONFIG['grad_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * CONFIG['grad_accumulation_steps']

            if step % 50 == 0 and step > 0:
                print(f"  Step {step} | Loss: {loss.item() * CONFIG['grad_accumulation_steps']:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | Train MSE Loss: {avg_loss:.4f}")

        mae, rmse, r2 = evaluate(model, test_loader)

        if rmse < best_rmse:
            best_rmse = rmse
            print(f"--> New Best RMSE: {best_rmse:.4f}")

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])

            preds = model(input_ids, attention_mask)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)

    print(f"Test Evaluation:")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2  : {r2:.4f}")
    print("-" * 30)

    return mae, rmse, r2

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA not found.")
    train()