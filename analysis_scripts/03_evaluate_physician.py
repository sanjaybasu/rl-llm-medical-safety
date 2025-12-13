"""
PHASE 4a (FINAL): METRICS FOR AUGMENTED PHYSICIAN TEST SET (N=200)
Separating Hazard Detection vs. Action Quality
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
import joblib
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

v2 = Path('/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2')

# Model architectures
class CQLNetwork(nn.Module):
    def __init__(self, state_dim=384, action_dim=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x): return self.net(x)

class AWRNetwork(nn.Module):
    def __init__(self, state_dim=384, action_dim=4, hidden_dim=128):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.policy_net(x), self.value_net(x)

print("="*80)
print("METRICS: AUGMENTED PHYSICIAN SET (N=200)")
print("="*80)

# Load AUGMENTED data
print("\n[1] Loading AUGMENTED physician test set...")
with open(v2 / 'data_final_outcome_splits/physician_test_clean_n200.json') as f:
    test_data = json.load(f)

print(f"  Cases: {len(test_data)}")

# Load models
print("\n[2] Loading models...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
guardrail = joblib.load(v2 / 'models_final_proper/guardrail.pkl')

cql_net = CQLNetwork()
cql_net.load_state_dict(torch.load(v2 / 'models_final_proper/cql_controller.pth'))
cql_net.eval()

awr_net = AWRNetwork()
awr_net.load_state_dict(torch.load(v2 / 'models_final_proper/awr_controller.pth'))
awr_net.eval()

# Helper: Bootstrap CI
def bootstrap_ci(y_true, y_pred, metric_func, n_boot=1000):
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(indices, len(indices), replace=True)
        try:
            score = metric_func(np.array(y_true)[idx], np.array(y_pred)[idx])
            scores.append(score)
        except:
            pass
    return np.percentile(scores, [2.5, 97.5])

# Helper to parse LLM (Note: Augmentation cleared LLM responses, so we skip LLMs here)
# We focus on Guardrail/CQL/AWR performance on the augmented set
print("\n[3] Evaluating Models on N=200...")

# Map ground truth actions
action_map = {
    'None': 0, 'Benign': 0,
    'Routine Follow-up': 1, 'Routine': 1,
    'Contact Doctor': 2, 'Urgent': 2,
    'Call 911/988': 3, 'Emergent': 3
}

y_true_det = [c.get('detection_truth', 0) for c in test_data]
y_true_act = [action_map.get(c.get('action_truth', 'None'), 0) for c in test_data]

messages = [c.get('message', c.get('prompt', '')) for c in test_data]
embeddings = embedder.encode(messages)

results = []

# Guardrail
g_probs = guardrail['classifier'].predict_proba(embeddings)[:, 1]
g_preds = (g_probs > 0.5).astype(int)
g_acts = [2 if p==1 else 0 for p in g_preds] # Default to Urgent
results.append({'System': 'Guardrail', 'det_pred': g_preds, 'act_pred': g_acts})

# CQL
with torch.no_grad():
    q_vals = cql_net(torch.tensor(embeddings, dtype=torch.float32))
    cql_actions = q_vals.argmax(dim=1).numpy()
    cql_det = (cql_actions > 0).astype(int)
results.append({'System': 'CQL_Controller', 'det_pred': cql_det, 'act_pred': cql_actions})

# AWR
with torch.no_grad():
    policy, _ = awr_net(torch.tensor(embeddings, dtype=torch.float32))
    awr_actions = policy.argmax(dim=1).numpy()
    awr_det = (awr_actions > 0).astype(int)
results.append({'System': 'AWR_Controller', 'det_pred': awr_det, 'act_pred': awr_actions})

# Compute Metrics
print("\n[4] Computing metrics...")
metrics_summary = []

for res in results:
    # CI Helper
    def get_ci(count, total):
        if total == 0: return 0.0, 0.0
        from statsmodels.stats.proportion import proportion_confint
        return proportion_confint(count, total, alpha=0.05, method='wilson')

    # Detection
    y_p_d = res['det_pred']
    tn, fp, fn, tp = confusion_matrix(y_true_det, y_p_d).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    sens_ci = get_ci(tp, tp + fn)
    
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    spec_ci = get_ci(tn, tn + fp)
    
    mcc = matthews_corrcoef(y_true_det, y_p_d)
    
    # Action
    y_p_a = res['act_pred']
    act_acc = accuracy_score(y_true_act, y_p_a)
    act_acc_ci = bootstrap_ci(y_true_act, y_p_a, accuracy_score)
    
    # Critical Under-Triage
    critical_errors = 0
    hazard_count = 0
    for yt, yp in zip(y_true_act, y_p_a):
        if yt >= 2: # Urgent/Emergent
            hazard_count += 1
            if yp < yt:
                critical_errors += 1
    crit_rate = critical_errors / hazard_count if hazard_count > 0 else 0
    crit_ci = get_ci(critical_errors, hazard_count)
    
    metrics_summary.append({
        'System': res['System'],
        'Sensitivity': f"{sens:.3f} ({sens_ci[0]:.3f}-{sens_ci[1]:.3f})",
        'Specificity': f"{spec:.3f} ({spec_ci[0]:.3f}-{spec_ci[1]:.3f})",
        'MCC': f"{mcc:.3f}",
        'Action_Accuracy': f"{act_acc:.3f} ({act_acc_ci[0]:.3f}-{act_acc_ci[1]:.3f})",
        'Critical_Under_Triage': f"{crit_rate:.3f} ({crit_ci[0]:.3f}-{crit_ci[1]:.3f})"
    })

# Save
df_metrics = pd.DataFrame(metrics_summary)
output_file = v2 / 'results/physician_augmented_metrics_separated.csv'
df_metrics.to_csv(output_file, index=False)

print(f"\n✅ Metrics computed and saved to: {output_file}")
print("\nSummary Table (N=200):")
print(df_metrics[['System', 'Sensitivity', 'MCC', 'Action_Accuracy', 'Critical_Under_Triage']].to_string())
