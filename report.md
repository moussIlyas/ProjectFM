# Rapport — UC5 Jalon 1 : estimation de la RUL (flotte A)

Compilation de tous les resultats obtenus (`notebooks/mlp4.ipynb`,
`notebooks/journal_jalon1_flotteA.csv`, experiences M1-M6). Les resultats de
validation sont exprimes en **cycles** ; le **score NASA** est une somme (plus bas =
mieux).

---

## 0. Synthese (meilleur par section)

| Section | Meilleure configuration (metrique) | RMSE | score NASA | anticipation | FA |
|---|---|---|---|---|---|
| 4. Modele de base | base 9 entrees | 19.81 | 129 100 | 0.778 | 0.0 |
| 5. 3 entrees vs 9 | 9 entrees (metrique) / 3 entrees (retenu) | 19.81 | 129 100 | 0.778 | 0.0 |
| 6. Normalisation | globale (metrique) / regime (retenu) | 20.51 | 139 801 | 0.644 | 0.0 |
| 7. Fonction de perte | **asymetrique** | 22.99 | 63 378 | 0.822 | 0.0 |
| 8. M6 (Swish + Huber) | **M6** | 24.61 | 53 336 | 0.822 | 0.0 |
| 9. M4 (L2 + dropout) | **M4** | 22.51 | 62 107 | 0.822 | 0.0 |
| 10. Fenetre k (M4) | **k = 30** (score/RMSE) | 20.16 | 46 794 | 0.756 | 0.0 |
| 11. Fenetre k (M6) | **k = 15** (compromis) | 22.12 | 48 254 | **0.844** | 0.0 |

**Meilleur score global** : M6 + fenetre **k = 5** (score 44 072).
**Meilleur compromis global** : M6 + fenetre **k = 15** (score 48 254, anticipation 0.844, RMSE 22.12) — meilleure anticipation de toutes les configurations.

---

## 1. Cadre experimental

- **Donnees** : flotte A, decoupage par pompe (aucun recouvrement) — 70 pompes train (19 220 cycles), 15 val (4 170), 30 test (6 538).
- **Entrees** : 8 capteurs + `charge`. Jeu reduit retenu : `capteur_4`, `capteur_5`, `charge`. EDA : `capteur_4`/`capteur_5` les plus correles a la RUL, `charge` module les mesures.
- **Cible** : RUL **plafonnee a 125 cycles** puis ramenee dans `[0, 1]` (activation de sortie sigmoide).
- **Normalisation** : z-score **par regime de charge** (statistiques des cycles sains), ou globale (ablation).
- **Metriques** : RMSE (cycles), score NASA (somme, penalise surtout la sur-estimation), anticipation (taux d'alerte sur RUL reelle <= 30), fausses alertes (alerte sur RUL reelle > 60), bruit (ecart-type moyen des predictions intra-pompe).
- **Cibles du cahier des charges** : RMSE <= 23, score <= 1 200, anticipation >= 0.90, fausses alertes <= 0.10. *(Non atteintes au jalon 1 : voir Limites.)*

---

## 4. Modele de base (9 entrees)

MLP `64-32-1`, normalisation par regime.

| modele | RMSE | score | anticipation | FA |
|---|---|---|---|---|
| **base 9 entrees** | **19.81** | **129 100** | 0.778 | 0.0 |

**Meilleur : base 9 entrees.** Bon sur les cycles sains, se degrade pres de la panne (tendance a sur-estimer la RUL, les predictions en retard).

---

## 5. Modele a 3 entrees vs 9 entrees

| modele | RMSE | score | anticipation | FA |
|---|---|---|---|---|
| **base 9 entrees** | **19.81** | **129 100** | **0.778** | 0.0 |
| reduit 3 entrees | 20.61 | 143 754 | 0.556 | 0.0 |

**Meilleur sur les metriques : 9 entrees.** Le modele a **3 entrees est retenu** pour sa compacite (reseau bien plus leger), avec une perte de performance moderee. *(Correction : la conclusion de `mlp4` disait "au moins aussi bons" — en realite 9 entrees est legerement meilleur.)*

---

## 6. Normalisation : globale vs par regime (3 entrees)

| normalisation | RMSE | score | anticipation | FA |
|---|---|---|---|---|
| **globale** | **20.51** | **139 801** | **0.644** | 0.0 |
| regime | 20.61 | 143 754 | 0.556 | 0.0 |

**Meilleure sur les metriques : globale** (meilleure sur les trois colonnes). La **normalisation par regime est retenue** pour son sens physique (retire le niveau propre a chaque charge). Ecart faible, sensible au run.

---

## 7. Fonction de perte (3 entrees, regime)

| perte | RMSE | score | anticipation | FA |
|---|---|---|---|---|
| mse | 20.61 | 143 754 | 0.556 | 0.0 |
| **asymetrique (MSE x3 sur sur-estimation)** | 22.99 | **63 378** | **0.822** | 0.0 |
| inspiree du score (exponentielle) | **22.11** | 321 865 | 0.822 | 0.0 |

**Meilleure : asymetrique** — divise le score par ~2.3 en penalisant les predictions en retard, au prix d'une RMSE un peu plus elevee. La perte exponentielle "inspiree du score" est instable (dominee par quelques outliers).

---

## 8. M6 — MLP residuel Swish + perte de Huber asymetrique

Architecture : deux blocs residuels pre-activation (Dense -> BatchNorm -> Swish -> Dropout -> Dense -> BatchNorm -> Add du skip -> Swish), tete `Dense(32, Swish) -> Dense(1, sigmoid)`.
Perte Huber asymetrique (`delta = 0.5`, sur-estimation `x4`), **AdamW** (lr `5e-4`, weight decay `1e-4`), early stopping sur `val_loss`, `ReduceLROnPlateau(0.5, 15)`. ~41 857 parametres.

| modele | RMSE | score | anticipation | FA |
|---|---|---|---|---|
| reduit 3 - perte asymetrique | 22.99 | 63 378 | 0.822 | 0.0 |
| **M6 residuel Swish + Huber asym** | 24.61 | **53 336** | 0.822 | 0.0 |

**Meilleur : M6** sur le score NASA (-16 %), anticipation identique, RMSE plus elevee.
Test 3 graines (moyenne) : base asymetrique `RMSE 22.69 ± 0.16 / score 65 282 ± 120` ; M6 `RMSE 24.02 ± 0.55 / score 55 482 ± 2 697`.

Deux corrections necessaires par rapport a l'enonce initial :
1. la penalite porte sur la **sur-estimation** (`y_hat - y > 0`), pas la sous-estimation, pour alleger le score NASA et l'anticipation ;
2. l'early stopping surveille **`val_loss`** : le score NASA etant une *somme* dominee par quelques outliers, le surveiller directement donne un arret premature tres degrade (RMSE ~54, score ~5e5).

---

## 9. M4 — MLP regularise (L2 + dropout), meilleur compromis RMSE / score

Architecture `128-64-32`, ReLU, regularisation **L2** (`1e-3`) + **dropout** (`0.4`),
perte asymetrique, Adam `1e-3`, early stopping sur `val_loss`. ~10 881 parametres.

| modele | RMSE | score | anticipation | FA |
|---|---|---|---|---|
| reduit 3 - perte asymetrique | 22.99 | 63 378 | 0.822 | 0.0 |
| **M4 (128-64-32, L2 + dropout 0.4)** | **22.51** | **62 107** | 0.822 | 0.0 |

**Meilleur : M4** — domine le modele asymetrique seul sur la RMSE *et* le score, avec une anticipation identique.

---

## 10. Fenetre temporelle aplatie k (M4)

Concept (`mlp2` section 7) : donner au reseau les **`k` derniers cycles concatenes** (entree `k x 3`, premier cycle repete en amorce). Mesure en plus du **bruit** (ecart-type moyen des predictions intra-pompe).

| k | RMSE | score | anticipation | FA | bruit |
|---|---|---|---|---|---|
| 1 | 22.51 | 62 107 | **0.822** | 0.0 | 30.99 |
| 5 | 20.73 | 50 635 | 0.778 | 0.0 | 31.34 |
| 15 | 20.75 | 53 308 | 0.778 | 0.0 | 31.64 |
| **30** | **20.16** | **46 794** | 0.756 | 0.0 | 31.78 |

**Meilleur : k = 30** (score et RMSE), mais l'anticipation decroit avec k et le bruit **augmente** legerement (pas de lissage). Aucune cible de bruit atteinte. `k = 5` capte deja l'essentiel du gain.

---

## 11. Fenetre temporelle aplatie k (M6)

Fenetre appliquee a M6 (meme protocole que la section 10).

| k | RMSE | score | anticipation | FA | bruit |
|---|---|---|---|---|---|
| 1 | 24.61 | 53 336 | 0.822 | 0.0 | 30.33 |
| **5** | 23.06 | **44 072** | 0.778 | 0.0 | 29.87 |
| **15** | **22.12** | 48 254 | **0.844** | 0.0 | 30.69 |
| 30 | 22.22 | 44 736 | 0.778 | 0.0 | 30.99 |

**Meilleur score : k = 5** (44 072). **Meilleur compromis et meilleure anticipation : k = 15** (0.844). La fenetre ameliore nettement M6 (score 53 336 -> ~44 000, RMSE 24.6 -> ~22.1).

---

## 12. Modeles retenus et recommandation

| Rang | Configuration | RMSE | score | anticipation |
|---|---|---|---|---|
| Score minimal | **M6 + k = 5** | 23.06 | **44 072** | 0.778 |
| Meilleur compromis | **M6 + k = 15** | 22.12 | 48 254 | **0.844** |
| Meilleure RMSE (fenetre) | M4 + k = 30 | **20.16** | 46 794 | 0.756 |
| Reference profonde simple | M4 | 22.51 | 62 107 | 0.822 |
| Reference asymetrique | reduit 3 asymetrique | 22.99 | 63 378 | 0.822 |

**Recommandation** : **M6 + fenetre k = 15** (meilleure anticipation 0.844 et score faible). Si l'on privilegie uniquement le score NASA, prendre **M6 + k = 5**.

---

## 13. Hyperparametres

| modele | architecture | perte | optimiseur | arret | params |
|---|---|---|---|---|---|
| base 64-32-1 | 64-32-1 sigmoid | MSE | Adam 1e-3, 150 ep, batch 256 | - | 2 369 |
| reduit 3 | 64-32-1 sigmoid | MSE | Adam 1e-3, 150 ep | - | 2 369 |
| asymetrique | 64-32-1 sigmoid | MSE x3 (sur-est.) | Adam 1e-3 | - | 2 369 |
| M4 | 128-64-32, L2 1e-3, dropout 0.4 | MSE x3 | Adam 1e-3 | val_loss, patience 25, RLROP 0.5/10 | 10 881 |
| M6 | res 128-128/64-64, Swish, dropout 0.25 | Huber d=0.5 x4 | AdamW 5e-4, wd 1e-4 | val_loss, patience 40, RLROP 0.5/15 | ~41 857 |

Fenetre k : entree `k x 3`, premier cycle repete en amorce (train et val normalises par regime avant fenetrage).

---

## 14. Limites et precautions

- **Runs uniques** (hors test M6 a 3 graines) : la variance run-a-run du score atteint ~2 000-2 700. Les ecarts `k = 5 / 15 / 30` sont dans le bruit ; seul le saut `k = 1 -> k >= 5` est robuste.
- **Reseau non deterministe** (dropout, BatchNorm, oneDNN) : re-executer un notebook donne des chiffres legerement differents.
- **Ablations 5 et 6** (3 vs 9 entrees, globale vs regime) dependent du run : conclusions a confirmer par moyenne multi-graines.
- **Cibles du cahier des charges non atteintes** (score <= 1 200, anticipation >= 0.90) : attendu au jalon 1 (30 pompes de test, modele simple, RUL plafonnee, score domine par les predictions en retard).
- Les architectures **M1-M5** (annexe) proviennent d'une experience separee (notebook supprime), a titre exploratoire seulement.

---

## Annexe A. Architectures exploratoires M1-M5 (run unique, indicatif)

Memes donnees (3 entrees, regime, perte asymetrique), seules l'architecture et la
regularisation changent.

| modele | architecture | params | epochs | RMSE | score | anticipation |
|---|---|---|---|---|---|---|
| M1 | 128-64-32, dropout 0.2 | 11 649 | 32 | 22.77 | 62 848 | 0.778 |
| **M2** | 256-128-64-32, dropout 0.3 | 45 825 | 47 | 24.05 | **54 132** | 0.533 |
| M3 | 5 x 128, dropout 0.15 | 69 249 | 31 | 22.69 | 67 245 | 0.822 |
| M4 | 128-64-32, L2 1e-3, dropout 0.4 | 10 881 | 106 | 22.72 | 61 053 | 0.822 |
| M5 | 128-64-16-64-1, dropout 0.2 | 11 985 | 40 | 22.45 | 66 167 | 0.822 |

**Lecture** : M2 donne le meilleur score brut mais degrede fortement l'anticipation ; M3/M5 sur-apprennent (score degrade) ; **M4 est le meilleur compromis** (retenu et integre a `mlp4`).

---

## Annexe B. Fichiers

- `notebooks/mlp4.ipynb` — notebook principal (sections 1-11).
- `notebooks/journal_jalon1_flotteA.csv` — journal des experiences.
- `notebooks/mlflow_jalon1.db` — suivi MLflow.
- `report.md` — ce rapport.
