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

**Q4 — fenêtre de `k` cycles (M4, 3 entrées)** :

| k | RMSE | score | anticipation | bruit |
|---|---|---|---|---|
| **1** | 20,47 | 144 682 | 0,56 | **32,19** |
| 5 | 18,93 | 104 218 | 0,53 | 32,49 |
| 15 | 18,84 | 110 375 | 0,53 | 32,19 |
| 30 | 18,47 | 98 510 | 0,56 | 33,00 |

Agrandir la fenêtre améliore un peu la RMSE/le score mais **ne réduit pas le bruit** et
n'apporte pas de gain décisif → **k = 1** retenu (le plus simple). Figure `fig_q4_fenetres.png`.

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
