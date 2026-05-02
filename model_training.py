# ============================================================
# IMPORTS
# ============================================================
from sklearn.model_selection import RandomizedSearchCV, train_test_split
# RandomizedSearchCV est préféré à GridSearchCV :
# il tire aléatoirement n_iter combinaisons au lieu de tester toutes → ~3x plus rapide
# pour des espaces de paramètres larges, il trouve de meilleures solutions en moins de temps

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from scipy.stats import randint  # pour définir des distributions continues de paramètres

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
df = pd.read_csv('churn_engineered1.csv')

X = df.drop('Churn', axis=1)  # features : toutes les colonnes sauf la cible
y = df['Churn']               # variable cible : 0 = pas de churn, 1 = churn

# ============================================================
# SPLIT TRAIN / TEST
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% pour le test, 80% pour l'entraînement
    random_state=42,    # reproductibilité des résultats
    stratify=y          # conserve la même proportion de churn dans train et test
)

# ⚡ OPTIMISATION #1 : StandardScaler SUPPRIMÉ
# Random Forest est un modèle basé sur des seuils de découpage (splits).
# Il est totalement insensible à l'échelle des features.
# Scaler les données pour RF est une perte de temps, voire perturbateur.
# → X_train et X_test sont utilisés directement (sans transformation)

# ============================================================
# DÉFINITION DE L'ESPACE DE RECHERCHE
# ============================================================
param_dist = {
    'n_estimators': randint(100, 500),
    # Nombre d'arbres dans la forêt.
    # randint(100, 500) tire des entiers uniformément entre 100 et 499.
    # Plus d'arbres = plus stable, mais rendements décroissants après ~300.

    'max_depth': [10, 20, 30, None],
    # Profondeur maximale de chaque arbre.
    # None = arbres complets (risque de sur-apprentissage).
    # Valeurs plus faibles régularisent le modèle.

    'min_samples_split': randint(2, 15),
    # Nombre minimum d'échantillons pour diviser un nœud interne.
    # Des valeurs élevées empêchent les divisions sur très peu d'exemples (régularisation).

    'min_samples_leaf': randint(1, 8),
    # Nombre minimum d'échantillons requis dans chaque feuille.
    # Évite les feuilles pures sur 1 ou 2 exemples (sur-apprentissage).

    'max_features': ['sqrt', 'log2', 0.5],
    # Nombre de features à considérer à chaque split.
    # 'sqrt' = √(nb_features), classique pour la classification.
    # 0.5 = 50% des features → plus de diversité entre arbres.

    'class_weight': ['balanced', 'balanced_subsample']
    # ⚡ OPTIMISATION #2 : class_weight intégré dans la grille
    # 'balanced' : pondère globalement selon la fréquence des classes.
    # 'balanced_subsample' : recalcule les poids à chaque bootstrap sample
    #   → souvent meilleur pour le churn (classes très déséquilibrées).
}

# ============================================================
# MODÈLE DE BASE
# ============================================================
rf_base = RandomForestClassifier(
    random_state=42,    # seed pour la reproductibilité des arbres
    bootstrap=True,     # active le bagging : chaque arbre s'entraîne sur un échantillon avec remise
    n_jobs=-1           # ⚡ OPTIMISATION #3 : parallélisation de l'entraînement sur tous les cœurs CPU
    # (n_jobs était mis dans GridSearchCV avant, mais l'idéal est ici sur le modèle lui-même)
)

# ============================================================
# RANDOMIZED SEARCH AVEC VALIDATION CROISÉE
# ============================================================
random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,  # espace de paramètres (distributions ou listes)
    n_iter=50,           # ⚡ OPTIMISATION #4 : 50 tirages aléatoires au lieu de 162 combinaisons
                         # → divise par ~3 le temps de calcul pour une qualité équivalente
    cv=5,                # 5-fold cross-validation stratifiée : évalue chaque config sur 5 plis
    scoring='roc_auc',   # métrique d'optimisation = AUC-ROC
                         # idéale pour le churn : gère le déséquilibre des classes
    n_jobs=-1,           # parallélisation des folds sur tous les cœurs
    random_state=42,     # reproductibilité de l'ordre des tirages
    verbose=1,           # affiche la progression (sans verbosité excessive)
    refit=True           # ré-entraîne automatiquement le meilleur modèle sur tout X_train
)

# ============================================================
# LANCEMENT DE LA RECHERCHE
# ============================================================
print("Démarrage du Randomized Search (50 itérations × 5 folds = 250 entraînements)...")
random_search.fit(X_train, y_train)
# Chaque combinaison est évaluée via CV sur X_train uniquement (X_test n'est jamais vu ici)

# ============================================================
# MEILLEUR MODÈLE
# ============================================================
best_model = random_search.best_estimator_
# Modèle ré-entraîné sur l'intégralité de X_train avec les meilleurs hyperparamètres

print("\nMEILLEURS PARAMÈTRES TROUVÉS :")
for param, value in random_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\nMeilleur score CV (AUC): {random_search.best_score_:.4f}")

# ============================================================
# ÉVALUATION SUR LE JEU DE TEST
# ============================================================
y_pred       = best_model.predict(X_test)           # classe prédite (0 ou 1)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # probabilité d'être classe 1 (churn)
# [:, 1] sélectionne la colonne de la classe positive

print("\nPERFORMANCE DU MODÈLE OPTIMISÉ :")
print(f"Accuracy : {best_model.score(X_test, y_test):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
print(f"\nMatrice de confusion :")
print(f"  Vrais Négatifs  : {TN}  |  Faux Positifs : {FP}")
print(f"  Faux Négatifs   : {FN}  |  Vrais Positifs : {TP}")

# ============================================================
# ⚡ OPTIMISATION #5 : ANALYSE DE L'IMPORTANCE DES FEATURES
# ============================================================
# Random Forest calcule nativement l'importance de chaque feature
# (réduction moyenne d'impureté sur tous les arbres).
# C'est une information précieuse pour la sélection de variables.

feature_importance = pd.DataFrame({
    'feature'  : X_train.columns,          # noms des colonnes
    'importance': best_model.feature_importances_  # score d'importance (somme = 1)
}).sort_values('importance', ascending=False)  # tri décroissant

print("\nTOP 10 FEATURES LES PLUS IMPORTANTES :")
print(feature_importance.head(10).to_string(index=False))

# ⚡ OPTIMISATION #6 (optionnelle) : ÉLAGAGE DES FEATURES PEU IMPORTANTES
# Supprimer les features avec importance < 0.01 peut réduire le temps d'inférence
# et améliorer la généralisation en réduisant le bruit.
threshold = 0.01
important_features = feature_importance[
    feature_importance['importance'] >= threshold
]['feature'].tolist()

print(f"\n{len(important_features)} features conservées sur {X_train.shape[1]} "
      f"(seuil importance >= {threshold})")
# Pour réentraîner avec ces features seulement :
# X_train_slim = X_train[important_features]
# X_test_slim  = X_test[important_features]


import joblib
import os

os.makedirs("saved_model", exist_ok=True)
joblib.dump(best_model, "saved_model/best_rf_model.pkl")
joblib.dump(list(X_train.columns), "saved_model/feature_names.pkl")

print("✅ Model saved successfully!")