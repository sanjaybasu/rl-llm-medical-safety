#!/usr/bin/env python3
"""
Week 2 Days 2-3: Train Guardrail (Constellation) + RL Controllers (CQL, AWR)
Uses combined training set (N=3,500)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import joblib

v2_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2')
models_dir = v2_dir / 'models_final'
models_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("WEEK 2 DAYS 2-3: MODEL TRAINING")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading training data...")

data_dir = v2_dir / 'data_final_v2'
with open(data_dir / 'combined_train.json') as f:
    train_data = json.load(f)

with open(data_dir / 'physician_val.json') as f:
    phys_val = json.load(f)

with open(data_dir / 'realworld_val.json') as f:
    real_val = json.load(f)

val_data = phys_val + real_val

print(f"  Train: {len(train_data)}")
print(f"  Val: {len(val_data)}")

# ============================================================================
# EMBEDDINGS
# ============================================================================
print("\n[2] Generating embeddings...")

embedder = SentenceTransformer('all-MiniLM-L6-v2')

train_texts = [c.get('prompt', '') for c in train_data]
train_labels = np.array([c.get('detection_truth', 0) for c in train_data])

val_texts = [c.get('prompt', '') for c in val_data]
val_labels = np.array([c.get('detection_truth', 0) for c in val_data])

print("  Embedding train...")
X_train = embedder.encode(train_texts, show_progress_bar=True)
print("  Embedding val...")
X_val = embedder.encode(val_texts, show_progress_bar=True)

print(f"  X_train shape: {X_train.shape}")
print(f"  X_val shape: {X_val.shape}")

# ============================================================================
# MODEL 1: GUARDRAIL (CONSTELLATION ARCHITECTURE)
# ============================================================================
print("\n[3] Training Guardrail (Constellation Architecture)...")

# Stage 1: Binary hazard detector
clf = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    C=1.0
)

calibrated_clf = CalibratedClassifierCV(
    clf,
    method='isotonic',
    cv=3
)

calibrated_clf.fit(X_train, train_labels)

# Evaluate on val
val_preds = calibrated_clf.predict(X_val)
val_acc = accuracy_score(val_labels, val_preds)
print(f"  Validation Accuracy: {val_acc:.3f}")

# Save
joblib.dump({
    'embedder_name': 'all-MiniLM-L6-v2',
    'classifier': calibrated_clf
}, models_dir / 'guardrail_constellation.pkl')

print(f"  ✅ Saved to {models_dir}/guardrail_constellation.pkl")

# ============================================================================
# MODEL 2: CONSERVATIVE Q-LEARNING (CQL)
# ============================================================================
print("\n[4] Training CQL Controller...")

class CQLNetwork(nn.Module):
    """Conservative Q-Learning Network"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Action space: 0=None, 1=Routine, 2=Urgent, 3=Emergent
action_map = {
    'None': 0,
    'Routine Follow-up': 1,
    'Contact Doctor': 2,
    'Call 911/988': 3
}

train_actions = np.array([action_map.get(c.get('action_truth'), 0) for c in train_data])
val_actions = np.array([action_map.get(c.get('action_truth'), 0) for c in val_data])

# Convert to tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(train_actions, dtype=torch.long)
X_val_torch = torch.tensor(X_val, dtype=torch.float32)
y_val_torch = torch.tensor(val_actions, dtype=torch.long)

# Initialize network
cql_net = CQLNetwork(state_dim=384, action_dim=4)

# Loss with safety weighting (prioritize emergent actions)
action_weights = torch.tensor([1.0, 2.0, 3.0, 5.0])  # Higher weight for Call 911
criterion = nn.CrossEntropyLoss(weight=action_weights)

optimizer = optim.Adam(cql_net.parameters(), lr=0.001)

# Train
print("  Training for 100 epochs...")
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward
    logits = cql_net(X_train_torch)
    loss = criterion(logits, y_train_torch)
    
    # Add CQL conservative regularization
    # Penalize overestimation of unseen actions
    q_values = torch.softmax(logits, dim=1)
    cql_loss = torch.logsumexp(logits, dim=1).mean() - q_values.gather(1, y_train_torch.unsqueeze(1)).mean()
    
    total_loss = loss + 0.1 * cql_loss
    
    # Backward
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        # Eval on val
        with torch.no_grad():
            val_logits = cql_net(X_val_torch)
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (val_preds == y_val_torch).float().mean().item()
        print(f"    Epoch {epoch+1}/100 - Loss: {total_loss:.4f} - Val Acc: {val_acc:.3f}")

# Save
torch.save(cql_net.state_dict(), models_dir / 'cql_controller.pth')
print(f"  ✅ Saved to {models_dir}/cql_controller.pth")

# ============================================================================
# MODEL 3: ADVANTAGE-WEIGHTED ACTOR-CRITIC (AWR)
# ============================================================================
print("\n[5] Training AWR Controller...")

class AWRNetwork(nn.Module):
    """Advantage-Weighted Regression Network"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        return policy, value

awr_net = AWRNetwork(state_dim=384, action_dim=4)
optimizer_awr = optim.Adam(awr_net.parameters(), lr=0.001)

# Compute advantages (reward = 1 if action matches severity, 0 otherwise)
# Higher reward for correctly identifying emergent vs missing emergent
def compute_reward(action, ground_truth_action):
    """Reward function prioritizing safety"""
    if action == ground_truth_action:
        # Correct action: high reward
        if action == 3:  # emergent
            return 5.0
        elif action == 2:  # urgent
            return 3.0
        else:
            return 1.0
    else:
        # Wrong action: penalty
        if ground_truth_action == 3 and action < 3:
            # Missed emergent: severe penalty
            return -5.0
        else:
            return -1.0

rewards = np.array([compute_reward(a, t) for a, t in zip(train_actions, train_actions)])
values = np.ones_like(rewards)  # Simplified baseline
advantages = rewards - values

# Convert to torch
advantages_torch = torch.tensor(advantages, dtype=torch.float32)

print("  Training for 100 epochs...")
for epoch in range(100):
    optimizer_awr.zero_grad()
    
    # Forward
    policy, value = awr_net(X_train_torch)
    
    # Policy loss (weighted by advantage)
    action_probs = policy.gather(1, y_train_torch.unsqueeze(1)).squeeze()
    exp_advantages = torch.exp(torch.clamp(advantages_torch / 10.0, -5, 5))  # Clip for stability
    policy_loss = -(torch.log(action_probs + 1e-8) * exp_advantages).mean()
    
    # Value loss
    value_loss = nn.MSELoss()(value.squeeze(), torch.tensor(rewards, dtype=torch.float32))
    
    # Total loss
    total_loss = policy_loss + 0.5 * value_loss
    
    # Backward
    total_loss.backward()
    optimizer_awr.step()
    
    if (epoch + 1) % 20 == 0:
        # Eval
        with torch.no_grad():
            val_policy, _ = awr_net(X_val_torch)
            val_preds = torch.argmax(val_policy, dim=1)
            val_acc = (val_preds == y_val_torch).float().mean().item()
        print(f"    Epoch {epoch+1}/100 - Loss: {total_loss:.4f} - Val Acc: {val_acc:.3f}")

# Save
torch.save(awr_net.state_dict(), models_dir / 'awr_controller.pth')
print(f"  ✅ Saved to {models_dir}/awr_controller.pth")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("MODEL TRAINING COMPLETE")
print("="*80)
print(f"\n✅ Guardrail (Constellation): {models_dir}/guardrail_constellation.pkl")
print(f"✅ CQL Controller: {models_dir}/cql_controller.pth")
print(f"✅ AWR Controller: {models_dir}/awr_controller.pth")
print(f"\n✅ WEEK 2 DAYS 2-3 COMPLETE")
