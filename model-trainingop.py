from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
# ============================================
# YOUR BUSINESS COSTS
# ============================================
COST_FN = 500  # Cost of missing a churner
COST_FP = 50   # Cost of false alarm



# Load your data
df = pd.read_csv('churn_engineered1.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split and scale (same as before)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# GRID SEARCH FOR HYPERPARAMETER TUNING
# ============================================

print("🔍 Starting Grid Search Optimization...")

# Define the parameter grid
param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [10, 12],           # Reduce slightly
    'min_samples_split': [50, 100],  # Increase more
    'min_samples_leaf': [20, 50],    # Increase more
    'max_features': ['sqrt', 'log2'],
    'max_samples': [0.7, 0.8]        # ADD THIS - restricts data per tree
}

# Create base model (THIS IS BAGGING - Random Forest)
rf_base = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    bootstrap=True  # YES - this enables bagging!
)

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc',  # Optimize for AUC
    n_jobs=-1,  # Use all CPU cores
    verbose=1  # Show progress
)

# Run the search (this may take a few minutes)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_

print("\n✅ BEST PARAMETERS FOUND:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\nBest CV Score: {grid_search.best_score_:.4f}")

# Evaluate on test set
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 OPTIMIZED MODEL PERFORMANCE:")
print(f"Accuracy: {best_model.score(X_test_scaled, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"True Negatives: {TN} | False Positives: {FP}")
print(f"False Negatives: {FN} | True Positives: {TP}")



#################################################################################################################################



# ============================================
# ============================================
# SEPARATE OPTIMIZATION BLOCK
# (Using the already trained best_model)
# ============================================
# ============================================

print("\n" + "=" * 70)
print("SEPARATE OPTIMIZATION BLOCK")
print("Threshold Tuning for Cost-Sensitive Predictions")
print("=" * 70)

# ============================================
# STEP 1: Get probabilities from your trained model
# ============================================
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# ============================================
# STEP 2: Function to find optimal threshold
# ============================================



    best_cm = None
    all_results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            total_cost = (FN * fn_cost) + (FP * fp_cost)
            all_results.append({
                'threshold': threshold,
                'cost': total_cost,
                'FN': FN,
                'FP': FP,
                'TP': TP,
                'TN': TN
            })

            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
                best_cm = (TN, FP, FN, TP)

    return best_threshold, best_cost, best_cm, all_results

# ============================================
# STEP 3: Find optimal threshold
# ============================================

optimal_threshold, min_cost, optimal_cm, all_results = find_optimal_threshold(
    y_test, y_proba, COST_FN, COST_FP
)

TN_opt, FP_opt, FN_opt, TP_opt = optimal_cm

print("\n" + "-" * 70)
print("OPTIMAL THRESHOLD FOUND")
print("-" * 70)
print(f"Default threshold (0.5) cost: ${(FN * COST_FN) + (FP * COST_FP):,.0f}")
print(f"Optimal threshold ({optimal_threshold:.4f}) cost: ${min_cost:,.0f}")
print(f"Savings: ${((FN * COST_FN) + (FP * COST_FP)) - min_cost:,.0f}")

print(f"\nConfusion Matrix at optimal threshold {optimal_threshold:.4f}:")
print(f"  True Negatives: {TN_opt}")
print(f"  False Positives: {FP_opt}")
print(f"  False Negatives: {FN_opt}")
print(f"  True Positives: {TP_opt}")

# ============================================
# STEP 4: Compare default vs optimal
# ============================================

print("\n" + "-" * 70)
print("COMPARISON: Default (0.5) vs Optimal Threshold")
print("-" * 70)

# Default predictions (threshold 0.5)
y_pred_default = (y_proba >= 0.5).astype(int)
cm_default = confusion_matrix(y_test, y_pred_default)
TN_def, FP_def, FN_def, TP_def = cm_default.ravel()

print(f"\n{'Metric':<25} {'Default (0.5)':<20} {'Optimal':<20} {'Change':<15}")
print("-" * 80)
print(f"{'Threshold':<25} {'0.5000':<20} {optimal_threshold:.4f}{'':<15}")
print(f"{'Total Cost ($)':<25} ${(FN_def * COST_FN) + (FP_def * COST_FP):<18,.0f} ${min_cost:<18,.0f} ${((FN_def * COST_FN) + (FP_def * COST_FP)) - min_cost:<14,.0f}")
print(f"{'False Negatives':<25} {FN_def:<20} {FN_opt:<20} {FN_def - FN_opt:<15}")
print(f"{'False Positives':<25} {FP_def:<20} {FP_opt:<20} {FP_def - FP_opt:<15}")
print(f"{'True Positives':<25} {TP_def:<20} {TP_opt:<20} {TP_opt - TP_def:<15}")
print(f"{'Recall':<25} {TP_def/(TP_def+FN_def):.4f}{'':<16} {TP_opt/(TP_opt+FN_opt):.4f}{'':<15} {(TP_opt/(TP_opt+FN_opt) - TP_def/(TP_def+FN_def)):+.4f}")
print(f"{'Precision':<25} {TP_def/(TP_def+FP_def):.4f}{'':<16} {TP_opt/(TP_opt+FP_opt):.4f}{'':<15} {(TP_opt/(TP_opt+FP_opt) - TP_def/(TP_def+FP_def)):+.4f}")

# ============================================
# STEP 5: Visualizations
# ============================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cost vs Threshold
ax1 = axes[0, 0]
thresholds = [r['threshold'] for r in all_results]
costs = [r['cost'] for r in all_results]
ax1.plot(thresholds, costs, 'b-', linewidth=2)
ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, label=f'Default (0.5): ${(FN_def * COST_FN) + (FP_def * COST_FP):,.0f}')
ax1.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal ({optimal_threshold:.3f}): ${min_cost:,.0f}')
ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('Total Cost ($)', fontsize=12)
ax1.set_title('Cost vs Threshold', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: FN and FP vs Threshold
ax2 = axes[0, 1]
fns = [r['FN'] for r in all_results]
fps = [r['FP'] for r in all_results]
ax2.plot(thresholds, fns, 'r-', linewidth=2, label='False Negatives (Missed Churners)')
ax2.plot(thresholds, fps, 'b-', linewidth=2, label='False Positives (False Alarms)')
ax2.axvline(x=optimal_threshold, color='green', linestyle='--', linewidth=2)
ax2.set_xlabel('Threshold', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Errors vs Threshold', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Probability Distribution
ax3 = axes[1, 0]
ax3.hist(y_proba[y_test == 0], bins=20, alpha=0.5, label='Actual No Churn', color='blue')
ax3.hist(y_proba[y_test == 1], bins=20, alpha=0.5, label='Actual Churn', color='red')
ax3.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, label='Default (0.5)')
ax3.axvline(x=optimal_threshold, color='green', linestyle='--', linewidth=2, label=f'Optimal ({optimal_threshold:.3f})')
ax3.set_xlabel('Probability of Churn', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Probability Distribution by Actual Class', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix Comparison
ax4 = axes[1, 1]
x = np.arange(2)
width = 0.35

default_errors = [FN_def, FP_def]
optimal_errors = [FN_opt, FP_opt]

bars1 = ax4.bar(x - width/2, default_errors, width, label='Default (0.5)', color='salmon')
bars2 = ax4.bar(x + width/2, optimal_errors, width, label=f'Optimal ({optimal_threshold:.3f})', color='steelblue')

ax4.set_xticks(x)
ax4.set_xticklabels(['False Negatives\n(Missed Churners)', 'False Positives\n(False Alarms)'])
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('Error Comparison', fontsize=14, fontweight='bold')
ax4.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# ============================================
# STEP 6: Final Recommendation
# ============================================

print("\n" + "=" * 70)
print("FINAL RECOMMENDATION")
print("=" * 70)

print(f"""
Based on your business costs:
- Cost of missing a churner (FN): ${COST_FN}
- Cost of false alarm (FP): ${COST_FP}
- Cost Ratio (FN/FP): {COST_FN/COST_FP:.1f}:1

✅ RECOMMENDED THRESHOLD: {optimal_threshold:.4f}

Why?
- Default threshold 0.5: Total Cost = ${(FN_def * COST_FN) + (FP_def * COST_FP):,.0f}
- Optimal threshold {optimal_threshold:.4f}: Total Cost = ${min_cost:,.0f}
- Your savings: ${((FN_def * COST_FN) + (FP_def * COST_FP)) - min_cost:,.0f}

Business Impact:
- You will catch {FN_def - FN_opt} MORE churners
- You will have {FP_opt - FP_def} MORE false alarms (acceptable trade-off)
- Net benefit: ${((FN_def * COST_FN) + (FP_def * COST_FP)) - min_cost:,.0f}
""")

# ============================================
# STEP 7: How to use the optimal threshold
# ============================================

print("\n" + "-" * 70)
print("HOW TO USE OPTIMAL THRESHOLD IN PRODUCTION")
print("-" * 70)
print(f"""
# Add this to your prediction code:

# Get probabilities from your model
y_proba = best_model.predict_proba(X_new_scaled)[:, 1]

# Apply optimal threshold
y_pred_optimized = (y_proba >= {optimal_threshold:.4f}).astype(int)

# Now y_pred_optimized is cost-optimized for your business!
""")

# ============================================
# STEP 8: Save optimal threshold for later use
# ============================================

import joblib

joblib.dump(optimal_threshold, 'optimal_threshold.pkl')
print(f"\n✅ Saved optimal threshold to 'optimal_threshold.pkl'")

# ============================================
# OPTIONAL: Test different cost scenarios
# ============================================

print("\n" + "-" * 70)
print("SENSITIVITY ANALYSIS: Different Cost Ratios")
print("-" * 70)

cost_ratios = [1, 5, 10, 20, 50]
print(f"\n{'FN Cost':<12} {'FP Cost':<12} {'Ratio':<10} {'Optimal Threshold':<20} {'Expected Cost':<15}")
print("-" * 70)

for ratio in cost_ratios:
    fn_test = COST_FP * ratio
    threshold_test, cost_test, _, _ = find_optimal_threshold(y_test, y_proba, fn_test, COST_FP)
    print(f"${fn_test:<11,.0f} ${COST_FP:<11,.0f} {ratio}:1{'':<6} {threshold_test:.4f}{'':<15} ${cost_test:,.0f}")

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETE!")
print("=" * 70)
