import pandas as pd
import matplotlib.pyplot as plt

# — T3: G#2 - FIXED CODE - NO RANDOM —
# Reads cluster_validation_summary.csv directly

grouped = pd.read_csv('cluster_validation_summary.csv')
print(grouped)

# Keep same labels as before
labels = ['5-10', '10-20', '20-40', '40-60', '60+']

# Plot - SAME as your original
plt.figure(figsize=(8,5))
plt.plot(labels, grouped['accuracy'], marker='o', linewidth=2.5, color='green')
plt.bar(labels, grouped['accuracy'], alpha=0.3, color='green')
for i, row in grouped.iterrows():
    plt.text(i, row['accuracy']+2, f"{row['accuracy']:.1f}%\n(n={row['count']})", ha='center', fontsize=9)

plt.title('G#2: AI Cluster Validation Accuracy vs Richness\n(1875 Classifications)')
plt.xlabel('Cluster Richness')
plt.ylabel('Validation Accuracy (% Real)')
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('GraphC_Cluster_Richness_T3.png', dpi=300)
plt.show()

print("\nDone! Download GraphC_Cluster_Richness_T3.png from folder icon")
