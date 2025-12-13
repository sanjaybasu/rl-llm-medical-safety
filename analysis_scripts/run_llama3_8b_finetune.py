"""
QLoRA fine-tune of Llama-3-8B-Instruct for action-bin classification (4 classes).

- Train on round2 annotated sets (real-world train/physician train/new scenarios).
- Validation: real-world is_val==1 split.
- Test: full round2 real-world test (n=2000).
- Saves adapter to models/llama3_8b_round2_lora and predictions/metrics to results/repro_round2/.

Requires CUDA GPU; tested config: single A100 40GB (should fit on 24GB with 4-bit).
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model


def map_bin(x: float) -> int:
    if pd.isna(x):
        return 0
    x = int(x)
    if x <= 1:
        return 0
    if x <= 3:
        return 1
    if x <= 6:
        return 2
    return 3


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    rad = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (max(0.0, (center - rad) / denom), min(1.0, (center + rad) / denom))


def load_data():
    base = Path('results/labeling_round2')
    train_rw = pd.read_csv(base / 'realworld_labelset_train_candidates_annotated.csv')
    test_rw = pd.read_csv(base / 'realworld_labelset_test_candidates_annotated.csv')
    phys_train = pd.read_json('data_final_outcome_splits/physician_train_clean.json')
    phys_new = pd.read_csv(base / 'physician_new_scenarios.csv')

    phys_train = phys_train.rename(columns={'message': 'text'})
    phys_train['action_bin'] = phys_train['action_truth'].apply(map_bin)

    phys_new = phys_new.rename(columns={'message_text': 'text', 'intended_action': 'action_truth'})
    phys_new['action_bin'] = phys_new['action_truth'].apply(map_bin)

    train_rw = train_rw.rename(columns={'message': 'text'})
    train_rw['action_bin'] = train_rw['clinician_action_mapped'].apply(map_bin)

    val_df = train_rw[train_rw['is_val'] == 1][['text', 'action_bin']].copy()
    train_rw = train_rw[train_rw['is_val'] != 1][['text', 'action_bin']].copy()

    train_df = pd.concat([
        train_rw,
        phys_train[['text', 'action_bin']],
        phys_new[['text', 'action_bin']]
    ], ignore_index=True)

    test_rw['text'] = test_rw['message']
    test_rw['action_bin'] = test_rw['clinician_action_mapped'].apply(map_bin)

    return train_df, val_df, test_rw


def tokenize_dataset(df: pd.DataFrame, tok, max_length: int = 512):
    ds = Dataset.from_pandas(df[['text', 'action_bin']])

    def tok_fn(batch):
        enc = tok(batch['text'], truncation=True, padding='max_length', max_length=max_length)
        enc['labels'] = batch['action_bin']
        return enc

    return ds.map(tok_fn, batched=True, remove_columns=['text'])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    # labels may include -100; handle
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return {
        'f1_macro': f1_score(labels, preds, average='macro'),
        'acc': accuracy_score(labels, preds)
    }


def evaluate_on_test(model, tok, test_df: pd.DataFrame, trainer):
    ds = Dataset.from_pandas(test_df[['text', 'action_bin', 'message_id']])

    def tok_fn(batch):
        enc = tok(batch['text'], truncation=True, padding='max_length', max_length=512)
        enc['labels'] = batch['action_bin']
        return enc

    ds = ds.map(tok_fn, batched=True, remove_columns=['text'])
    preds = trainer.predict(ds)
    logits = preds.predictions
    prob = torch.softmax(torch.tensor(logits), dim=1)[:, 1:].sum(1).numpy()
    y_true = test_df['action_bin'].values
    det_true = (test_df['action_bin'] > 0).astype(int)
    # default threshold 0.5 for hazard detection
    det_pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(det_true, det_pred).ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    sens_ci = wilson_ci(tp, tp + fn)
    spec_ci = wilson_ci(tn, tn + fp)
    f1 = f1_score(det_true, det_pred)
    mcc = matthews_corrcoef(det_true, det_pred)
    auroc = roc_auc_score(det_true, prob)

    metrics = {
        'dataset': 'RealWorld_Test',
        'system': 'Llama3_8B_LoRA',
        'sensitivity': sens,
        'sensitivity_ci_lower': sens_ci[0],
        'sensitivity_ci_upper': sens_ci[1],
        'specificity': spec,
        'specificity_ci_lower': spec_ci[0],
        'specificity_ci_upper': spec_ci[1],
        'f1': f1,
        'mcc': mcc,
        'auroc': auroc,
        'n_sample': len(det_true),
        'n_hazard': int(det_true.sum()),
        'n_safe': int((1 - det_true).sum())
    }

    rows = []
    for mid, t, p in zip(test_df['message_id'], det_true, prob):
        rows.append({'dataset': 'RealWorld_Test', 'system': 'Llama3_8B_LoRA', 'message_id': mid, 'true_label': int(t), 'probability': float(p)})
    return metrics, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--output_dir', default='models/llama3_8b_round2_lora')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=4)
    args = parser.parse_args()

    train_df, val_df, test_df = load_data()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token

    train_ds = tokenize_dataset(train_df, tok)
    val_ds = tokenize_dataset(val_df, tok)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=4,
        quantization_config=bnb_config,
        device_map='auto'
    )

    lora = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.1,
        bias='none',
        task_type='SEQ_CLS'
    )
    model = get_peft_model(model, lora)

    args_train = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type='cosine',
        logging_steps=50,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        bf16=True,
        gradient_checkpointing=True,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        compute_metrics=compute_metrics
    )

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    metrics, preds_df = evaluate_on_test(model, tok, test_df, trainer)
    out_dir = Path('results/repro_round2')
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(out_dir/'llama3_8b_predictions.csv', index=False)
    pd.DataFrame([metrics]).to_csv(out_dir/'llama3_8b_metrics.csv', index=False)
    print('Saved metrics and predictions to results/repro_round2')


if __name__ == '__main__':
    main()
