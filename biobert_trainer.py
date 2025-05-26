# biobert_trainer.py

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import joblib
import os

model_dir = "biobert_disease_model"
file_path = "D:/IMP.DATA/Downloads/GEN_AI/synthetic_medical_symptoms_dataset.xlsx"
df = pd.read_excel(file_path)

# Encode disease labels
le_disease = LabelEncoder()
df["disease_encoded"] = le_disease.fit_transform(df["disease"])
joblib.dump(le_disease, "disease_label_encoder.pkl")

# Combine symptoms into single string
df["combined_symptoms"] = df[["symptom1", "symptom2", "symptom3"]].astype(str).agg(" ".join, axis=1)

tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# === Dataset class ===
class SymptomDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

# === Prediction function for GUI ===
def predict_disease_and_recurrence(user_symptoms, past_diseases):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    le = joblib.load("disease_label_encoder.pkl")

    model.eval()
    combined_input = " ".join(user_symptoms)
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_disease = le.inverse_transform([predicted_class])[0]

    # === Get specialist for predicted disease from dataset ===
    matched_row = df[df["disease"].str.lower() == predicted_disease.lower()]
    if not matched_row.empty:
        predicted_specialist = matched_row["specialist"].values[0]
    else:
        predicted_specialist = "Specialist not found"

    # === Recurrence logic ===
    recurrence_probs = {}
    if past_diseases:
        base_prob = 100 / len(user_symptoms)
        for disease in past_diseases:
            match_count = 0
            disease_rows = df[df["disease"].str.lower() == disease.lower()]
            disease_symptoms = disease_rows[["symptom1", "symptom2", "symptom3"]].values
            for row in disease_symptoms:
                for symptom in row:
                    if str(symptom).lower().strip() in [s.lower().strip() for s in user_symptoms]:
                        match_count += 1
            recurrence_probs[disease] = round(base_prob * match_count, 2)

        total = sum(recurrence_probs.values())
        if total > 100:
            for d in recurrence_probs:
                recurrence_probs[d] = round((recurrence_probs[d] / total) * 100, 2)

    return{
        "predicted_disease": predicted_disease,
        "predicted_specialist": predicted_specialist,
        "recurrence_probabilities": recurrence_probs
    }

# === Training block: Only runs if this file is run directly ===
def train_model():
    model = BertForSequenceClassification.from_pretrained(
        "dmis-lab/biobert-base-cased-v1.1",
        num_labels=len(df["disease_encoded"].unique())
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["combined_symptoms"], df["disease_encoded"], test_size=0.2, random_state=42
    )

    train_dataset = SymptomDataset(train_texts.tolist(), train_labels.tolist())
    val_dataset = SymptomDataset(val_texts.tolist(), val_labels.tolist())

    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == "__main__":
    print("Training BioBERT model...")
    train_model()
