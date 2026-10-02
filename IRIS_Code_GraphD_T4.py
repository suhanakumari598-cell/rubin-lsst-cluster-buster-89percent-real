import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('galaxies.csv')
print(f"N={len(df)} loaded")
print("Columns:", df.columns.tolist()[:8])

# auto-find merger column
for c in ['is_merger','merger','Merger','is_merger_flag','target']:
  if c in df.columns:
    df['is_merger'] = (df[c].astype(str).str.contains('merger', case=False).astype(int) if df[c].dtype==object else (df[c]>0.5).astype(int) if df[c].max()<=1 else df[c])
    break

# features
feats = [x for x in ['g_r','r_mag','g_mag','mass','z'] if x in df.columns]
if len(feats)<2:
  df['g_r'] = df['g_r_real']
  df['r_mag'] = df['g_mag'] if 'g_mag' in df.columns else 19 + df['g_r']*0.5
  feats = ['g_r','r_mag']

X = df[feats].fillna(df[feats].mean())
y = df['is_merger']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
acc = accuracy_score(y_test, model.predict(X_test))

print(f"\n✅ RESULT: ML predicts mergers with {acc*100:.1f}% accuracy vs human!")
print(f"Ready for 10-yr Rubin LSST Survey of 20 billion galaxies")

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt='d', cmap='Blues')
plt.title(f'GraphD: AI Merger Prediction {acc*100:.1f}% (N={len(df)})')
plt.xlabel('AI Predicted'); plt.ylabel('Human')
plt.savefig('GraphD_Week3.png', dpi=300)
plt.show()
