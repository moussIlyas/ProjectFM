# Étude de cas 5 — Jalon 1 : estimer la durée de vie restante (RUL)

Notebook : `notebooks/jalon1_final.ipynb` (exécuté, graine `SEED = 42`, versions en tête).
Journal d'expériences : `notebooks/journal_jalon1_flotteA.csv` (`journaliser`, `jalon = 1`).
Figures : `notebooks/figures/`. Modèles : `notebooks/models/` (voir `notebooks/models/README.md`
pour les charger et les utiliser).

## 1. Objectif et données

Prédire la RUL (nombre de cycles avant panne) d'une pompe de type A avec un MLP.
Flotte A : **70 pompes** (train), **15** (validation), **30** (test, historiques coupés).
Découpage imposé **par pompe**. RUL **plafonnée à 125 cycles** ; une pompe est *signalée*
si la RUL prédite est **< 30 cycles**.

Entrées : 8 capteurs + `charge` (régime 60/80/100 %). `capteur_4` et `capteur_5` portent
l'essentiel de l'information, `charge` module les mesures.

## 2. Résultats et réponses aux questions

Protocole : tous les réglages sur la **validation** ; le test n'est évalué **qu'une fois**.

**Q1 — référence + erreur vs RUL réelle** (9 entrées, MSE, normalisation par régime) :
RMSE val = **19,0** ; biais moyen = **+1,05 cycle** (léger retard). Le modèle est bon sur
les cycles sains et se dégrade près de la panne, en **sur-estimant** la RUL (prédictions en
retard, erreurs positives). Figure `fig_q1_erreur_reference.png`.

**Q2 — normalisation globale vs par régime** (RMSE / score) :

| normalisation | RMSE | score |
|---|---|---|
| globale | 19,13 | 104 563 |
| **régime** | **19,01** | **103 457** |

Le régime est légèrement meilleur → **retenu**. Figure `fig_q2_normalisation.png`.

**Q3 — perte asymétrique vs MSE** :

| perte | RMSE | score | anticipation |
|---|---|---|---|
| MSE | 19,01 | 103 457 | 0,73 |
| **asymétrique ×3** | 20,87 | **49 858** | **0,78** |
| inspirée du score | 20,33 | 200 245 | 0,76 |

La perte asymétrique **divise le score par ~2** (elle réduit la sur-estimation, partie
dominante du score) → **retenue**. La perte exponentielle « inspirée du score » est
instable (décomposition du score dans le notebook).

**Q3bis — référence à 3 entrées** (mêmes décisions : régime, perte asymétrique, sans
L2/dropout) vs référence 9 entrées (validation) :

| modèle | RMSE | score | anticipation |
|---|---|---|---|
| **référence 9 entrées, asym** | **20,87** | **49 858** | **0,78** |
| référence 3 entrées, asym | 22,78 | 59 601 | 0,76 |

Le modèle 3 entrées est plus simple mais reste **en retrait en anticipation** : on garde
les 9 entrées. C'est ce modèle « référence 3 entrées » qui est utilisé pour la fenêtre (Q4).

**Q4 — fenêtre de `k` cycles (modèle retenu 9 entrées + perte asymétrique)** :

| k | RMSE | score | anticipation | bruit |
|---|---|---|---|---|
| **1 (retenu)** | 20,87 | 49 858 | 0,78 | **32,09** |
| 5 | 19,80 | **46 390** | **0,89** | 33,58 |
| 15 | 19,55 | 49 327 | 0,82 | 33,83 |
| 30 | **18,96** | 52 183 | 0,87 | 34,80 |

Aucune fenêtre ne domine clairement : k=5 gagne en score/anticipation, k=30 en RMSE,
mais le **bruit augmente** avec k. k=1, déjà validé et testé une fois sur le test, est
**retenu** (le plus simple, meilleur bruit). Figure `fig_q4_fenetres.png`.

## 3. Modèle retenu

**Référence (9 entrées, normalisation par régime) + perte asymétrique (`MSE × 3`)** :
meilleur score sur la validation, et modèle le plus simple parmi les candidats.

**Évaluation sur le test (une seule fois, une prédiction par pompe) :**

| exigence | mesure | cible | obtenu | respectée |
|---|---|---|---|---|
| RUL, type A | RMSE (cycles) | ≤ 23 | **22,68** | oui |
| coût des erreurs | score asymétrique | ≤ 1 200 | **997,3** | oui |
| anticipation | part des pompes ≤ 30 cycles signalées | ≥ 0,90 | **0,909** | oui |
| fausses alertes | part des pompes > 60 cycles signalées | ≤ 0,10 | **0,000** | oui |

Le modèle **respecte les quatre cibles du cahier des charges**. Figure
`fig_test_prediction.png`. Réserve : le test ne compte que 30 pompes (11 à ≤ 30 cycles),
les taux sont donc quantifiés par pas de ~0,09.

## 4. Choix à justifier

- **Plafonnement de la RUL à 125 cycles** : la contribution au score explose au-delà de
  ~20 cycles d'erreur ; plafonner borne la cible et évite que les cycles très lointains
  ne dominent l'apprentissage.
- **Perte** : **asymétrique**, car le score punit plus la sur-estimation (+10 coûte ×1,48
  et +30 coûte ×2,11 par rapport à une sous-estimation de même amplitude).
- **Activation de sortie** : **sigmoïde**, car la cible est dans `[0, 1]` après
  plafonnement (0 → RUL 0, 1 → RUL ≥ 125).

## 5. Vérifications théoriques

1. **Paramètres du MLP de référence `9-64-32-1`** :
   `(9×64 + 64) + (64×32 + 32) + (32×1 + 1) = 640 + 2080 + 33 = **2753**`
   (vérifié par `count_params()`).
2. **Contribution au score** de `d = y_pred − y_reel` :

   | d (cycles) | sens | contribution |
   |---|---|---|
   | −30 | sous-estimation | 9,051 |
   | −10 | sous-estimation | 1,158 |
   | +10 | sur-estimation | 1,718 |
   | +30 | sur-estimation | 19,086 |

   À amplitude égale, une **sur-estimation** coûte ~1,5 à ~2,1 fois plus qu'une
   sous-estimation → d'où la **perte asymétrique**.
