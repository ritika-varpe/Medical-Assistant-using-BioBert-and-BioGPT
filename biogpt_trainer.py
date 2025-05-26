import os
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 1
EPOCHS = 5
LR = 5e-5

# Your structured medical examples
sample_texts = [
    "Symptoms: persistent cough, weight loss, night sweats. History: none. Specialist: Pulmonologist. Recommendations: Recommend chest X-ray to rule out tuberculosis. Consider CBC and sputum test.",

    "Symptoms: frequent urination, burning sensation, blood in urine. History: UTI and kidney stone. Specialist: Urologist. Recommendations: Suspect recurrent UTI. Urinalysis and urine culture suggested. Consider starting empirical antibiotics.",

    "Symptoms: fatigue, palpitations, pale skin. History: none. Specialist: Hematologist. Recommendations: Suspect anemia. Order CBC and iron studies. Consider dietary advice and iron supplements.",

    "Symptoms: lower abdominal pain, unusual vaginal discharge. History: none. Specialist: Gynecologist. Recommendations: Recommend pelvic examination. Consider vaginal swab test and urine culture.",

    "Symptoms: headache, blurry vision, nausea. History: migraines. Specialist: Neurologist. Recommendations: Suspect migraine recurrence. Suggest MRI if symptoms worsen. Consider pain relief and hydration.",

    "Symptoms: chest pain, shortness of breath, dizziness. History: hypertension. Specialist: Cardiologist. Recommendations: Recommend ECG and blood pressure monitoring. Rule out myocardial infarction.",

    "Symptoms: itchy rash, dry skin. History: eczema. Specialist: Dermatologist. Recommendations: Suspect eczema flare-up. Use topical corticosteroids. Keep skin moisturized. Avoid allergens.",

    "Symptoms: joint pain, stiffness, swelling in knees. History: none. Specialist: Rheumatologist. Recommendations: Consider osteoarthritis. Recommend X-ray and anti-inflammatory meds.",

    "Symptoms: sore throat, fever, swollen lymph nodes. History: none. Specialist: General Physician. Recommendations: Suspect viral pharyngitis. Rest, fluids, and symptomatic treatment recommended.",

    "Symptoms: abdominal bloating, diarrhea, fatigue. History: lactose intolerance. Specialist: Gastroenterologist. Recommendations: Recommend lactose-free diet. Consider hydrogen breath test.",

    "Symptoms: missed period, breast tenderness, nausea. History: none. Specialist: Gynecologist. Recommendations: Recommend urine pregnancy test. Consider prenatal counseling.",

    "Symptoms: numbness in hands, trouble gripping objects. History: none. Specialist: Neurologist. Recommendations: Suspect carpal tunnel syndrome. Recommend nerve conduction studies.",

    "Symptoms: blood in stool, abdominal pain, weight loss. History: none. Specialist: Gastroenterologist. Recommendations: Recommend colonoscopy. Rule out colorectal cancer.",

    "Symptoms: dry eyes, blurry vision. History: diabetes. Specialist: Ophthalmologist. Recommendations: Recommend fundus examination. Monitor for diabetic retinopathy.",

    "Symptoms: fever, body aches, dry cough. History: flu shot not taken. Specialist: General Physician. Recommendations: Suspect influenza. Recommend rest, hydration, and paracetamol.",

    # New examples:
    "Symptoms: tremors, slow movement, muscle stiffness. History: none. Specialist: Neurologist. Recommendations: Suspect early Parkinson’s. Recommend neurological evaluation and MRI.",

    "Symptoms: excessive thirst, frequent urination, blurred vision. History: family history of diabetes. Specialist: Endocrinologist. Recommendations: Recommend fasting glucose test and HbA1c. Monitor blood sugar regularly.",

    "Symptoms: painful urination, pelvic pain. History: none. Specialist: Urologist. Recommendations: Suspect urinary tract infection. Recommend urinalysis and antibiotics.",

    "Symptoms: difficulty breathing at night, snoring. History: obesity. Specialist: Sleep Specialist. Recommendations: Suspect sleep apnea. Recommend sleep study and CPAP therapy if confirmed.",

    "Symptoms: skin redness, warmth, pus drainage. History: none. Specialist: Dermatologist. Recommendations: Suspect skin infection (cellulitis). Recommend antibiotics and wound care.",

    "Symptoms: excessive hair loss, brittle nails. History: thyroid issues. Specialist: Endocrinologist. Recommendations: Check thyroid hormone levels. Consider TSH and T3/T4 tests.",

    "Symptoms: pain while chewing, earache, jaw clicking. History: none. Specialist: Dentist. Recommendations: Suspect TMJ disorder. Recommend dental X-ray and jaw alignment evaluation."
]

# Load BioGPT tokenizer and model
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer.pad_token = tokenizer.eos_token  # Make sure padding is handled

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dataset class
class MedicalTextDataset(Dataset):
    def __init__(self, texts):
        self.inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")

    def __len__(self):
        return self.inputs["input_ids"].shape[0]

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx]
        attention_mask = self.inputs["attention_mask"][idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

dataset = MedicalTextDataset(sample_texts)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
model.train()
print("Training BioGPT started...")
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# Save the fine-tuned BioGPT model
save_path = "./biogpt_medical_finetuned"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Fine-tuned BioGPT model saved to {save_path}")
