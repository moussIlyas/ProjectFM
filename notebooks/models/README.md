# Utiliser les modèles UC5 — Jalon 1

Ce dossier contient les modèles entraînés par `notebooks/jalon1_final.ipynb`.
Tous prédisent la **RUL** (durée de vie restante en cycles) d'une pompe de type A.

## Fichiers

| Fichier | Entrées | Perte | Rôle |
|---|---|---|---|
| **`modele_retenu.keras`** | 9 | asymétrique ×3 | **modèle livré** — le seul qui respecte les 4 cibles du test (= fenêtre `k=1`) |
| `reference_asym_3entrees.keras` | 3 | asymétrique ×3 | référence 3 entrées (`capteur_4, capteur_5, charge`), ablation 6bis |
| `reference_9ent_k5.keras` | 45 | asymétrique ×3 | modèle retenu, fenêtre de 5 cycles (Q4) |
| `reference_9ent_k15.keras` | 135 | asymétrique ×3 | modèle retenu, fenêtre de 15 cycles (Q4) |
| `reference_9ent_k30.keras` | 270 | asymétrique ×3 | modèle retenu, fenêtre de 30 cycles (Q4) |

Entrées **9** = `capteur_0 … capteur_7` + `charge`.
Entrées **`9×k`** = les 9 caractéristiques sur les `k` derniers cycles, aplatis
(cycle le plus ancien → le plus récent).

## Convention

- **Sortie** : RUL normalisée dans `[0, 1]`. RUL = `clip(125 × sortie, 0, 125)`.
- **Entrée** : les capteurs sont **déjà normalisés par régime de charge** (`charge` reste
  brute). Appliquer la normalisation avant de prédire (voir ci-dessous).
- **Seuils** : pompe **signalée** si RUL prédite < `30` ; fausse alerte si RUL réelle > `60`.

## Chargement

Charger avec `compile=False` : les pertes personnalisées ne sont pas sérialisées et
feraient échouer la compilation. Prédire n'a pas besoin de la perte.

```python
import keras
modele = keras.models.load_model("notebooks/models/modele_retenu.keras", compile=False)
```

## Exemple complet (9 entrées, reproduit `RMSE test = 22,68`)

À lancer **depuis la racine du projet**.

```python
import numpy as np, pandas as pd, keras
from pathlib import Path

DATA, MODELS = Path("data_UC5/uc5_flotteA"), Path("notebooks/models")
RUL_MAX, SEUIL_ALERTE = 125, 30
CAPTEURS = [f"capteur_{i}" for i in range(8)]
FEATURES = CAPTEURS + ["charge"]
FEATURES_3 = ["capteur_4", "capteur_5", "charge"]

def charger(split):
    capteurs = np.load(DATA / f"capteurs_{split}.npy")
    charge = np.round(np.load(DATA / f"charge_{split}.npy").astype(float), 1)
    d = pd.DataFrame({"pompe": np.load(DATA / f"pompe_{split}.npy"),
                      "cycle": np.load(DATA / f"cycle_{split}.npy"), "charge": charge})
    return pd.concat([d, pd.DataFrame(capteurs, columns=CAPTEURS)], axis=1
                     ).sort_values(["pompe", "cycle"]).reset_index(drop=True)

# 1) Normalisation par régime, apprise sur les cycles sains du train
sains = charger("train")
sains["rul_cap"] = np.minimum(np.load(DATA / "rul_train.npy"), RUL_MAX)
couches = {}
for c in [0.6, 0.8, 1.0]:
    s = sains[(sains["charge"] == c) & (sains["rul_cap"] == RUL_MAX)]
    couches[c] = keras.layers.Normalization(axis=-1)
    couches[c].adapt(s[CAPTEURS].values.astype("float32"))

def normaliser(d):
    d = d.copy()
    for c, couche in couches.items():
        m = d["charge"] == c
        d.loc[m, CAPTEURS] = couche(d.loc[m, CAPTEURS].values.astype("float32")).numpy()
    return d

# 2) Prédiction avec le modèle retenu (une ligne = un cycle)
def predire_rul(d, nom="modele_retenu"):
    modele = keras.models.load_model(MODELS / f"{nom}.keras", compile=False)
    X = normaliser(d)[FEATURES].values.astype("float32")
    return np.clip(RUL_MAX * modele.predict(X, verbose=0).ravel(), 0, RUL_MAX)

test = charger("test")
rul_fin = np.minimum(np.load(DATA / "rul_fin_test.npy"), RUL_MAX).astype(float)
dernier = test.groupby("pompe").tail(1)
pred = predire_rul(dernier)
print("RMSE test :", round(np.sqrt(((pred - rul_fin) ** 2).mean()), 2),
      "| pompes signalées :", int((pred < SEUIL_ALERTE).sum()))
```

### Modèles à fenêtre (`reference_9ent_k*`)

Même normalisation ; l'entrée est la fenêtre aplatie des `k` derniers cycles des
**9 caractéristiques** (ordre : cycle le plus ancien → le plus récent). Une prédiction par
pompe, au dernier cycle observé :

```python
def predire_rul_fenetre(d, k, nom=None):
    nom = nom or f"reference_9ent_k{k}"
    n = normaliser(d).tail(k)          # les k derniers cycles de la pompe
    if len(n) < k:                     # amorce : 1er cycle répété
        n = pd.concat([n.iloc[[0]]] * (k - len(n)) + [n])
    X = n[FEATURES].values.astype("float32").ravel()[None, :]
    m = keras.models.load_model(MODELS / f"{nom}.keras", compile=False)
    return np.clip(RUL_MAX * m.predict(X, verbose=0).ravel(), 0, RUL_MAX)
```