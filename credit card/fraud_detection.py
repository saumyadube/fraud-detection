 # Credit Card Fraud Detection — American Express Portfolio Project 
 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')

#Load the dataset 

df = pd.read_csv('creditcard.csv') 

# Always check these 4 things first

print("Shape:", df.shape) 
print("\nFirst 5 rows:")
print(df.head()) 
print("\nMissing values:", df.isnull().sum().sum()) 
print("\nClass distribution:")
print(df['Class'].value_counts()) 
print("\nFraud percentage:", round(df['Class'].mean() * 100, 4), "%")

#Visualisations 

counts = df['Class'].value_counts().sort_index()

#imbalance bar chart 

plt.figure(figsize=(7, 5)) 
bars = plt.bar( ['Legitimate', 'Fraud'], [counts[0], counts[1]], color=['#185FA5', '#D85A30'], width=0.5, edgecolor='white' )

# Add count labels on top of bars

for bar, val in zip(bars, [counts[0], counts[1]]): plt.text( bar.get_x() + bar.get_width() / 2, bar.get_height() + counts[0] * 0.01, f'{val:,}', ha='center', fontsize=11, fontweight='bold' ) 
plt.title('Class Imbalance — Only 0.17% Fraud', fontsize=13) 
plt.ylabel('Number of Transactions') 
plt.grid(axis='y', alpha=0.3) 
plt.tight_layout()
plt.savefig('plot1_class_imbalance.png', dpi=150) 
plt.close() 
print('✓ Plot 1 saved: plot1_class_imbalance.png') 

#Transaction amount distribution

plt.figure(figsize=(9, 5)) 
plt.hist( np.log1p(df[df['Class'] == 0]['Amount']), bins=60, alpha=0.6, color='#185FA5', label='Legitimate', density=True ) 
plt.hist( np.log1p(df[df['Class'] == 1]['Amount']), bins=60, alpha=0.8, color='#D85A30', label='Fraud', density=True ) 
plt.xlabel('log(Amount + 1)')
plt.ylabel('Density') 
plt.title('Transaction Amount Distribution by Class', fontsize=13) 
plt.legend(fontsize=10) 
plt.grid(alpha=0.3) 
plt.tight_layout()
plt.savefig('plot2_amount_distribution.png', dpi=150) 
plt.close()
print('✓ Plot 2 saved: plot2_amount_distribution.png') 

 #Feature correlations with fraud 
 
v_cols = [f'V{i}' for i in range(1, 29)] 
corr = ( df[v_cols + ['Amount', 'Class']] .corr()['Class'] .drop('Class') .sort_values() ) 

#negative + top 8 positive correlations

top_corr = pd.concat([corr.head(8), corr.tail(8)]) 
bar_colors = ['#D85A30' if v < 0 else '#185FA5' for v in top_corr.values] 
plt.figure(figsize=(10, 6)) 
plt.barh(top_corr.index, top_corr.values, color=bar_colors, edgecolor='white') 
plt.axvline(0, color='black', linewidth=0.8, linestyle='--') 
plt.xlabel('Pearson Correlation with Fraud Label') 
plt.title('Top 16 Features Correlated with Fraud', fontsize=13) 
plt.grid(axis='x', alpha=0.3) 
plt.tight_layout() 
plt.savefig('plot3_correlations.png', dpi=150) 
plt.close() 
print('✓ Plot 3 saved: plot3_correlations.png') 

#Feature 1: Amount Z-score

df['Amount_zscore'] = ( (df['Amount'] - df['Amount'].mean()) / df['Amount'].std() )
print("Feature 1: Amount_zscore created") 
print(" Fraud mean Z-score :", round(df[df['Class'] == 1]['Amount_zscore'].mean(), 2)) 
print(" Legit mean Z-score :", round(df[df['Class'] == 0]['Amount_zscore'].mean(), 2))

# Feature 2: Transaction velocity per hour

df_sorted = df.sort_values('Time').copy() 
df_sorted['Hour_bucket'] = (df_sorted['Time'] // 3600).astype(int)
df_sorted['Velocity'] = ( df_sorted.groupby('Hour_bucket').cumcount() + 1 ) 

# Merge Velocity back to original df order

df['Velocity'] = df_sorted.sort_index()['Velocity'].values 
print("\nFeature 2: Velocity created")
print(" Fraud mean velocity:", round(df[df['Class'] == 1]['Velocity'].mean(), 1)) 
print(" Legit mean velocity:", round(df[df['Class'] == 0]['Velocity'].mean(), 1))

# Feature 3: Log-transformed amount 

df['Amount_log'] = np.log1p(df['Amount']) 

# Feature 4: Hour of day (0 to 24)

df['Hour'] = (df['Time'] / 3600) % 24 

# Drop raw Amount and Time — replaced by engineered versions

df_model = df.drop( columns=['Amount', 'Time', 'Hour_bucket'], errors='ignore' ) 
print("\nFeature 3: Amount_log created") 
print("Feature 4: Hour created")

print(f" df_model shape : {df_model.shape}") 

print(f" Total features : {df_model.shape[1] - 1}")
print(f" New columns added: Amount_zscore, Velocity, Amount_log, Hour") 
print(f" Columns dropped : Amount, Time")

# STEP 4: TRAIN / TEST SPLIT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

X = df_model.drop('Class', axis=1)
y = df_model['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} rows | Fraud: {y_train.sum()}")
print(f"Test : {len(X_test):,} rows  | Fraud: {y_test.sum()}")

# STEP 5: SCALE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled  = scaler.transform(X_test)        # transform only
print("Scaling done — fit on train only, no leakage ✓")

# STEP 6: SMOTE — training only, NEVER test
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE — Fraud: {y_train_sm.sum():,} | Legit: {(y_train_sm==0).sum():,}")

# STEP 7: TRAIN 3 MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, recall_score,
    precision_score, f1_score)

results = {}

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
lr.fit(X_train_sm, y_train_sm)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
lr_pred = (lr_prob >= 0.5).astype(int)
results['Logistic Regression'] = {
    'prob': lr_prob, 'pred': lr_pred,
    'auc': roc_auc_score(y_test, lr_prob),
    'rec': recall_score(y_test, lr_pred),
    'pre': precision_score(y_test, lr_pred),
    'f1':  f1_score(y_test, lr_pred)
}
print("✓ Logistic Regression trained | AUC:",
      round(results['Logistic Regression']['auc'], 4))

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
     random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)
rf_prob = rf.predict_proba(X_test_scaled)[:, 1]
rf_pred = (rf_prob >= 0.5).astype(int)
results['Random Forest'] = {
    'prob': rf_prob, 'pred': rf_pred,
    'auc': roc_auc_score(y_test, rf_prob),
    'rec': recall_score(y_test, rf_pred),
    'pre': precision_score(y_test, rf_pred),
    'f1':  f1_score(y_test, rf_pred)
}
print("✓ Random Forest trained       | AUC:",
      round(results['Random Forest']['auc'], 4))

# Model 3: XGBoost
spw = float((y_train == 0).sum()) / float(y_train.sum())
xgb = XGBClassifier(n_estimators=200, max_depth=6,
      learning_rate=0.05, subsample=0.8,
      scale_pos_weight=spw, random_state=42, verbosity=0)
xgb.fit(X_train_sm, y_train_sm)
xgb_prob = xgb.predict_proba(X_test_scaled)[:, 1]
xgb_pred = (xgb_prob >= 0.5).astype(int)
results['XGBoost'] = {
    'prob': xgb_prob, 'pred': xgb_pred,
    'auc': roc_auc_score(y_test, xgb_prob),
    'rec': recall_score(y_test, xgb_pred),
    'pre': precision_score(y_test, xgb_pred),
    'f1':  f1_score(y_test, xgb_pred)
}
print("✓ XGBoost trained             | AUC:",
      round(results['XGBoost']['auc'], 4))

# Print comparison table
print(f"\n{'Model':<22} {'AUC':>7} {'Recall':>7} {'Precision':>10} {'F1':>7}")
print("-" * 55)
for name, m in results.items():
    print(f"{name:<22} {m['auc']:>7.4f} {m['rec']:>7.4f}"
          f" {m['pre']:>10.4f} {m['f1']:>7.4f}")

# STEP 8: VISUALISATIONS
from sklearn.metrics import roc_curve, confusion_matrix
import seaborn as sns

model_colors = {
    'Logistic Regression': '#7F77DD',
    'Random Forest':       '#1D9E75',
    'XGBoost':             '#D85A30'
}

# Plot 4: ROC curves
plt.figure(figsize=(8, 6))
for name, m in results.items():
    fpr, tpr, _ = roc_curve(y_test, m['prob'])
    plt.plot(fpr, tpr, label=f"{name}  AUC={m['auc']:.3f}",
             color=model_colors[name], linewidth=2)
plt.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.4, label='Random (0.500)')
plt.xlabel('False Positive Rate  (legitimate blocked)')
plt.ylabel('True Positive Rate  (fraud caught)')
plt.title('ROC Curves — All 3 Models', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig('plot4_roc_curves.png', dpi=150)
plt.close()


# Plot 5: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices at Threshold = 0.50', fontsize=13)

for ax, (name, m) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, m['pred'])
    labels = [
        [f"TN\n{cm[0,0]:,}", f"FP\n{cm[0,1]:,}"],
        [f"FN ⚠\n{cm[1,0]}", f"TP ✓\n{cm[1,1]}"]
    ]
    import numpy as np
    sns.heatmap(cm, annot=np.array(labels), fmt='',
                cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Pred: Legit','Pred: Fraud'],
                yticklabels=['Actual: Legit','Actual: Fraud'],
                linewidths=0.8)
    ax.set_title(f"{name}\nRecall={m['rec']:.3f}  "
                 f"Precision={m['pre']:.3f}")

plt.tight_layout()
plt.savefig('plot5_confusion_matrices.png', dpi=150)
plt.close()

# Plot 6: XGBoost feature importance
feat_imp = pd.Series(
    xgb.feature_importances_,
    index=X.columns
).sort_values().tail(20)

engineered = {'Amount_zscore','Velocity','Amount_log','Hour'}
bar_colors = ['#D85A30' if f in engineered
              else '#7F77DD' for f in feat_imp.index]

plt.figure(figsize=(10, 7))
bars = plt.barh(feat_imp.index, feat_imp.values,
                color=bar_colors, edgecolor='white')
plt.xlabel('Feature Importance (XGBoost gain)')
plt.title('Top 20 Features — Red = Engineered by me', fontsize=12)
plt.tight_layout()
plt.savefig('plot6_feature_importance.png', dpi=150)
plt.close()
print("✓ Plot 6 saved: plot6_feature_importance.png")


# STEP 9: ISOLATION FOREST
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve

iso = IsolationForest(
    n_estimators=100,
    contamination=0.002,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train_scaled)   # NO y_train — purely unsupervised

iso_raw   = iso.predict(X_test_scaled)
iso_pred  = np.where(iso_raw == -1, 1, 0)
iso_score = -iso.score_samples(X_test_scaled)
iso_auc   = roc_auc_score(y_test, iso_score)

print("✓ Isolation Forest trained (unsupervised — no labels used)")
print(f"  AUC-ROC   : {iso_auc:.4f}")
print(f"  Recall    : {recall_score(y_test, iso_pred):.4f}")
print(f"  Precision : {precision_score(y_test, iso_pred):.4f}")
print(f"  F1        : {f1_score(y_test, iso_pred):.4f}")
print(f"\n  XGBoost AUC    : {results['XGBoost']['auc']:.4f}  (supervised)")
print(f"  Isolation AUC  : {iso_auc:.4f}  (unsupervised — no labels!)")

# Plot 7: Isolation Forest plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Isolation Forest — Unsupervised Anomaly Detection',
             fontsize=13)

# Left — anomaly score distribution
ax = axes[0]
ax.hist(iso_score[y_test == 0], bins=60, alpha=0.55,
        color='#185FA5', label='Legitimate', density=True)
ax.hist(iso_score[y_test == 1], bins=60, alpha=0.85,
        color='#D85A30', label='Fraud', density=True)
ax.set_xlabel('Anomaly Score (higher = more suspicious)')
ax.set_ylabel('Density')
ax.set_title('Anomaly Score Distribution by Class')
ax.legend()

# Right — ROC comparison XGBoost vs Isolation Forest
ax = axes[1]
fpr_x, tpr_x, _ = roc_curve(y_test, results['XGBoost']['prob'])
fpr_i, tpr_i, _ = roc_curve(y_test, iso_score)
ax.plot(fpr_x, tpr_x, color='#D85A30', linewidth=2.5,
        label=f"XGBoost supervised  AUC={results['XGBoost']['auc']:.3f}")
ax.plot(fpr_i, tpr_i, color='#7F77DD', linewidth=2,
        linestyle='--',
        label=f"Isolation Forest    AUC={iso_auc:.3f}")
ax.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC: Supervised vs Unsupervised')
ax.legend()

plt.tight_layout()
plt.savefig('plot7_isolation_forest.png', dpi=150)
plt.close()
print("✓ Plot 7 saved: plot7_isolation_forest.png")

# STEP 10: THRESHOLD TUNER
thresholds = np.linspace(0.01, 0.99, 200)
th_precision, th_recall, th_f1, th_blocked = [], [], [], []
total_legit = (y_test == 0).sum()

for t in thresholds:
    pred = (xgb_prob >= t).astype(int)
    th_precision.append(precision_score(y_test, pred, zero_division=0))
    th_recall.append(recall_score(y_test, pred, zero_division=0))
    th_f1.append(f1_score(y_test, pred, zero_division=0))
    fp = ((pred == 1) & (y_test == 0)).sum()
    th_blocked.append(fp / total_legit * 100)

best_idx = int(np.argmax(th_f1))
best_t   = thresholds[best_idx]

print(f"✓ Optimal threshold by F1 : {best_t:.2f}")
print(f"  At threshold {best_t:.2f}:")
print(f"    Recall    : {th_recall[best_idx]:.3f}")
print(f"    Precision : {th_precision[best_idx]:.3f}")
print(f"    F1        : {th_f1[best_idx]:.3f}")
print(f"    Legit blocked : {th_blocked[best_idx]:.2f}%")

# Plot 8: Threshold tuner
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('XGBoost Threshold Tuner — Business Decision Tool',
             fontsize=13)

ax = axes[0]
ax.plot(thresholds, th_recall,    color='#D85A30', lw=2,
        label='Recall  (fraud caught)')
ax.plot(thresholds, th_precision, color='#185FA5', lw=2,
        label='Precision  (alert accuracy)')
ax.plot(thresholds, th_f1,        color='#EF9F27', lw=2.5,
        linestyle='--', label='F1 score')
ax.axvline(best_t, color='black', lw=1.2, linestyle=':',
           label=f'Best F1 threshold = {best_t:.2f}')
ax.axvline(0.50,   color='#C8CDD6', lw=1, linestyle='--',
           label='Default threshold = 0.50')
ax.set_xlabel('Decision threshold')
ax.set_ylabel('Score')
ax.set_title('Precision · Recall · F1 vs Threshold')
ax.legend()
ax.set_xlim(0, 1)

ax = axes[1]
ax.fill_between(thresholds, th_blocked, alpha=0.2, color='#D85A30')
ax.plot(thresholds, th_blocked, color='#D85A30', lw=2)
ax.axvline(best_t, color='black', lw=1.2, linestyle=':',
           label=f'Best F1 threshold = {best_t:.2f}')
ax.set_xlabel('Decision threshold')
ax.set_ylabel('Legitimate transactions blocked (%)')
ax.set_title('Business Cost — Customer Friction vs Threshold')
ax.legend()
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('plot8_threshold_tuner.png', dpi=150)
plt.close()
print("✓ Plot 8 saved: plot8_threshold_tuner.png")


# STEP 11: FINAL SUMMARY
print("\n" + "="*60)
print("  PROJECT COMPLETE — FRAUD DETECTION ENGINE")
print("="*60)

best = results['XGBoost']
print(f"""
  Dataset        : 284,807 real transactions
  Fraud rate     : 0.17% (severe class imbalance)

  BEST MODEL     : XGBoost

  METRICS (default threshold 0.50):
    AUC-ROC      : {best['auc']:.4f}
    Recall       : {best['rec']:.4f}
    Precision    : {best['pre']:.4f}
    F1           : {best['f1']:.4f}

  METRICS (optimal threshold {best_t:.2f}):
    Recall       : {th_recall[best_idx]:.4f}
    Precision    : {th_precision[best_idx]:.4f}
    F1           : {th_f1[best_idx]:.4f}
    Legit blocked: {th_blocked[best_idx]:.2f}%

  PLOTS SAVED    : 8 PNG files in your folder
""")

print("  RESUME BULLET:")
print(f"""
  Built an end-to-end fraud detection pipeline on 284,807
  transactions — applied SMOTE to address 0.17% class
  imbalance, engineered velocity and Z-score anomaly features,
  benchmarked Logistic Regression, Random Forest, and XGBoost;
  XGBoost achieved {best['auc']:.2f} AUC-ROC and
  {best['rec']*100:.0f}% Recall. Added Isolation Forest for
  unsupervised cold-start detection.
""")
print("="*60)
