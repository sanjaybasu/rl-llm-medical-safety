"""
COMPLETE ALL EVALUATIONS
Ensure all systems are evaluated on all datasets with consistent methodology
"""
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from statsmodels.stats.proportion import proportion_confint

v2 = Path('/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2')

print("="*80)
print("COMPLETE ALL EVALUATIONS")
print("="*80)

# Load models
print("\n[1] Loading re-trained models...")
with open(v2 / 'models_retrained_n1000/guardrail.pkl', 'rb') as f:
    guardrail_data = pickle.load(f)
    encoder = guardrail_data['encoder']
    guardrail = guardrail_data['classifier']

class QNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

cql_net = QNetwork(state_dim=2, action_dim=4)
cql_net.load_state_dict(torch.load(v2 / 'models_retrained_n1000/cql_controller.pth'))
cql_net.eval()

awr_net = ActorNetwork(state_dim=2, action_dim=4)
awr_net.load_state_dict(torch.load(v2 / 'models_retrained_n1000/awr_controller.pth'))
awr_net.eval()

def normalize_case(case):
    normalized = case.copy()
    if 'detection_truth' in case:
        normalized['ground_truth_detection'] = case['detection_truth']
    if 'action_truth' in case:
        normalized['ground_truth_action'] = case['action_truth']
    if 'hazard_category' in case:
        normalized['ground_truth_hazard_category'] = case['hazard_category']
    return normalized

def compute_metrics(y_true_det, y_pred_det, y_true_act, y_pred_act, name):
    tn, fp, fn, tp = confusion_matrix(y_true_det, y_pred_det).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    sens_ci = proportion_confint(tp, tp + fn, alpha=0.05, method='wilson')
    
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    spec_ci = proportion_confint(tn, tn + fp, alpha=0.05, method='wilson')
    
    mcc = matthews_corrcoef(y_true_det, y_pred_det)
    
    has_action = y_true_act > 0
    act_acc = (y_true_act[has_action] == y_pred_act[has_action]).mean() if has_action.sum() > 0 else 0
    
    hazard_mask = y_true_det == 1
    if hazard_mask.sum() > 0:
        under_triage = (y_pred_act[hazard_mask] < y_true_act[hazard_mask]).sum()
        crit_rate = under_triage / hazard_mask.sum()
        crit_ci = proportion_confint(under_triage, hazard_mask.sum(), alpha=0.05, method='wilson')
    else:
        crit_rate = 0
        crit_ci = (0, 0)
    
    return {
        'System': name,
        'Sensitivity': f"{sens:.3f} ({sens_ci[0]:.3f}-{sens_ci[1]:.3f})",
        'Specificity': f"{spec:.3f} ({spec_ci[0]:.3f}-{spec_ci[1]:.3f})",
        'MCC': f"{mcc:.3f}",
        'Action_Accuracy': f"{act_acc:.3f}",
        'Critical_Under_Triage': f"{crit_rate:.3f} ({crit_ci[0]:.3f}-{crit_ci[1]:.3f})"
    }

# EVALUATION 1: Physician Test Set (N=200) with re-trained models
print("\n[2] Evaluating on Physician Test Set (N=200)...")
with open(v2 / 'data_final_outcome_splits/physician_test_clean_n200.json') as f:
    phys_test_raw = json.load(f)

phys_test = [normalize_case(c) for c in phys_test_raw]
messages = [c.get('message', '') or c.get('prompt', '') for c in phys_test]
X_phys = encoder.encode(messages, show_progress_bar=False)

y_true_det = np.array([c.get('ground_truth_detection', 0) for c in phys_test])
y_true_act = np.array([
    0 if c.get('ground_truth_action') in [None, 'None', ''] else
    1 if c.get('ground_truth_action') == 'Routine Follow-up' else
    2 if c.get('ground_truth_action') == 'Contact Doctor' else
    3 if c.get('ground_truth_action') == 'Call 911/988' else 0
    for c in phys_test
])

# Guardrail
guardrail_det = (guardrail.predict_proba(X_phys)[:, 1] > 0.5).astype(int)
guardrail_act = np.zeros(len(phys_test), dtype=int)

# CQL
cql_det, cql_act = [], []
for i in range(len(phys_test)):
    prob = guardrail.predict_proba(X_phys[i:i+1])[0]
    state = torch.FloatTensor(prob).unsqueeze(0)
    with torch.no_grad():
        q_values = cql_net(state)
        action = q_values.argmax().item()
    cql_det.append(1 if action > 0 else 0)
    cql_act.append(action)

cql_det = np.array(cql_det)
cql_act = np.array(cql_act)

# AWR
awr_det, awr_act = [], []
for i in range(len(phys_test)):
    prob = guardrail.predict_proba(X_phys[i:i+1])[0]
    state = torch.FloatTensor(prob).unsqueeze(0)
    with torch.no_grad():
        action_probs = awr_net(state)
        action = action_probs.argmax().item()
    awr_det.append(1 if action > 0 else 0)
    awr_act.append(action)

awr_det = np.array(awr_det)
awr_act = np.array(awr_act)

results_phys = []
results_phys.append(compute_metrics(y_true_det, guardrail_det, y_true_act, guardrail_act, 'Guardrail'))
results_phys.append(compute_metrics(y_true_det, cql_det, y_true_act, cql_act, 'CQL_Controller'))
results_phys.append(compute_metrics(y_true_det, awr_det, y_true_act, awr_act, 'AWR_Controller'))

df_phys = pd.DataFrame(results_phys)
output_dir = v2 / 'results_n1000_evaluation'
output_dir.mkdir(exist_ok=True)
df_phys.to_csv(output_dir / 'physician_n200_metrics.csv', index=False)

print(f"  ✓ Saved to: {output_dir / 'physician_n200_metrics.csv'}")
print(df_phys.to_string(index=False))

print("\n✅ ALL EVALUATIONS COMPLETE")
print(f"\nResults saved to: {output_dir}")
print("  - physician_n200_metrics.csv")
print("  - realworld_n200_metrics.csv (from Phase 3)")
