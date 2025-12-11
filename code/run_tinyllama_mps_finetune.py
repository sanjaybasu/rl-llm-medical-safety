import os, json
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

annotated_csv = "results/labeling_round2/realworld_labelset_train_candidates_annotated.csv"
out_dir = "models/tinyllama_round2_cls"
os.makedirs(out_dir, exist_ok=True)

def to_bin(action):
    a = int(action)
    if a in (1, 2): return 0
    if a in (3, 4): return 1
    if a in (5, 6): return 2
    return 3

df = pd.read_csv(annotated_csv)
df = df[df["clinician_action_mapped"].notna()].copy()
df["label"] = df["clinician_action_mapped"].apply(to_bin)
df["text"] = df["message"]

train_df = df[df["is_val"] != 1].copy()
val_df   = df[df["is_val"] == 1].copy()

def to_ds(frame):
    return Dataset.from_list(frame[["text","label"]].to_dict(orient="records"))

train_ds = to_ds(train_df)
val_ds   = to_ds(val_df)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
    out["labels"] = batch["label"]
    return out

train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    pad_token_id=tokenizer.pad_token_id
)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

args = TrainingArguments(
    output_dir=out_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=1,  # keep batch=1 since no native pad in base model
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    remove_unused_columns=True,
    report_to="none",
    use_mps_device=True if device=="mps" else False,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
metrics = trainer.evaluate()

with open(os.path.join(out_dir, "tinyllama_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
trainer.save_model(out_dir)
tokenizer.save_pretrained(out_dir)

preds = trainer.predict(val_ds)
pred_labels = preds.predictions.argmax(axis=1)
with open(os.path.join(out_dir, "tinyllama_val_predictions.csv"), "w") as f:
    f.write("text,label,pred\n")
    for rec, pred in zip(val_df.to_dict(orient="records"), pred_labels):
        text_safe = str(rec["text"]).replace('"', "'")
        f.write(f"\"{text_safe}\",{rec['label']},{pred}\n")
