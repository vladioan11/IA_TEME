import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    balanced_accuracy_score, 
    classification_report, 
    confusion_matrix
)

# Setari vizuale pentru grafice academice
plt.rcParams['figure.dpi'] = 300
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 1. INCARCAREA SI EXPLORAREA DATELOR
# ==========================================
# Poti folosi fisierul descarcat de pe Kaggle ('winequality-red.csv') 
# sau URL-ul direct catre setul oficial Cortez et al.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
try:
    df = pd.read_csv('winequality-red.csv')
    if df.shape[1] == 1:  # in caz ca separatorul e ';'
        df = pd.read_csv('winequality-red.csv', sep=';')
except FileNotFoundError:
    df = pd.read_csv(url, sep=';')

print(f"Dimensiunea setului de date: {df.shape[0]} instante, {df.shape[1]} coloane")
print("\nDistributia claselor (scorul de calitate):")
print(df['quality'].value_counts().sort_index())

# Grafic 1: Distributia dezechilibrata a claselor
plt.figure(figsize=(8, 4.5))
ax = sns.countplot(x='quality', data=df, palette='viridis', hue='quality', legend=False)
plt.title('Distribuția claselor în setul de date Red Wine Quality (Dezechilibru de clasă)', fontsize=12, fontweight='bold')
plt.xlabel('Calitatea vinului (Clasa țintă)', fontsize=11)
plt.ylabel('Număr de instanțe', fontsize=11)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('1_distributia_claselor.png')
plt.show()

# ==========================================
# 2. PREPROCESAREA DATELOR
# ==========================================
X = df.drop('quality', axis=1)
y = df['quality']

# Impartire stratificata 80% antrenare - 20% testare
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardizare Z-score (esentiala pentru calculul distantelor in kNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. DEFINIREA REGULILOR DE CLASIFICARE
# ==========================================
# Functie custom pentru ponderare Gaussiana (Exponential Decay)
def gaussian_weights(distances):
    sigma = 1.5
    return np.exp(-(distances ** 2) / (2 * (sigma ** 2)))

methods = {
    'Standard kNN (Vot Majoritar / Uniform)': 'uniform',
    'Alternativa 1: Inverse-Distance kNN (1/d)': 'distance',
    'Alternativa 2: Gaussian Kernel kNN': gaussian_weights
}

# ==========================================
# 4. EXPERIMENT COMPARATIV PE PLAJA LUI K
# ==========================================
k_values = range(1, 51, 2)  # Valori impare de la 1 la 49
results_acc = {name: [] for name in methods}
results_f1 = {name: [] for name in methods}

for k in k_values:
    for name, weight_rule in methods.items():
        clf = KNeighborsClassifier(n_neighbors=k, weights=weight_rule, metric='euclidean')
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        results_acc[name].append(accuracy_score(y_test, y_pred))
        results_f1[name].append(f1_score(y_test, y_pred, average='macro'))

# Grafic 2: Evolutia Acuratetii si a scorului F1-Macro in functie de k
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

styles = ['--', '-', '-.']
markers = ['o', 's', '^']
colors = ['#d95f02', '#1b9e77', '#7570b3']

for (name, acc_list), style, marker, color in zip(results_acc.items(), styles, markers, colors):
    ax1.plot(k_values, acc_list, linestyle=style, marker=marker, markersize=5, color=color, label=name, linewidth=2)

ax1.set_title('Acuratețea în funcție de numărul de vecini (k)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Valoarea lui k (Număr de vecini)', fontsize=11)
ax1.set_ylabel('Acuratețe (Accuracy)', fontsize=11)
ax1.legend(fontsize=9.5)

for (name, f1_list), style, marker, color in zip(results_f1.items(), styles, markers, colors):
    ax2.plot(k_values, f1_list, linestyle=style, marker=marker, markersize=5, color=color, label=name, linewidth=2)

ax2.set_title('F1-Score (Macro) în funcție de numărul de vecini (k)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Valoarea lui k (Număr de vecini)', fontsize=11)
ax2.set_ylabel('Macro F1-Score (sensibil la clasele minoritare)', fontsize=11)
ax2.legend(fontsize=9.5)

plt.tight_layout()
plt.savefig('2_comparatie_performanta_k.png')
plt.show()

# ==========================================
# 5. TABEL COMPARATIV DETALIAT (PENTRU K=15 SI K=35)
# ==========================================
summary_rows = []
for k_eval in [5, 15, 35]:
    for name, weight_rule in methods.items():
        start_time = time.perf_counter()
        clf = KNeighborsClassifier(n_neighbors=k_eval, weights=weight_rule, metric='euclidean')
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        summary_rows.append({
            'k': k_eval,
            'Metodă (Regulă Clasificare)': name,
            'Accuracy (%)': round(accuracy_score(y_test, y_pred) * 100, 2),
            'Balanced Acc (%)': round(balanced_accuracy_score(y_test, y_pred) * 100, 2),
            'F1-Macro (%)': round(f1_score(y_test, y_pred, average='macro') * 100, 2),
            'F1-Weighted (%)': round(f1_score(y_test, y_pred, average='weighted') * 100, 2),
            'Timp Execuție (ms)': round(elapsed_ms, 2)
        })

summary_df = pd.DataFrame(summary_rows)
print("\n=== TABEL COMPARATIV DE PERFORMANTA ===")
print(summary_df.to_string(index=False))

# ==========================================
# 6. MATRICI DE CONFUZIE LA K=15
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
classes = sorted(y.unique())

for ax, (name, weight_rule) in zip(axes, methods.items()):
    clf = KNeighborsClassifier(n_neighbors=15, weights=weight_rule)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False, ax=ax)
    ax.set_title(f'{name}\n(k = 15)', fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Clasa Prezisă')
    ax.set_ylabel('Clasa Reală')

plt.tight_layout()
plt.savefig('3_matrici_confuzie_k15.png')
plt.show()
