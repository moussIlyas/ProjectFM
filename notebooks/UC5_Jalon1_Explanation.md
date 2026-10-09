# UC5 - Jalon 1 : Explication du problème, des critères et des concepts

## 1. Problématique du Jalon 1

L'objectif du Jalon 1 est de **prédire la durée de vie restante (Remaining Useful Life, RUL)** d'une pompe, à partir de signaux de capteurs mesurés au cours de son fonctionnement.

- **RUL** : nombre de cycles restant avant la panne. Plus on s'approche de la panne, la RUL diminue.
- **Données** : deux types décrits dans le PDF. Ici nous travaillons sur les données **type B** (flotte restreinte) : 10 pompes pour l'entraînement, 20 pompes pour le test.
- **Objectif pratique** : aider à la maintenance prédictive (anticiper une alerte suffisamment tôt, sans trop d'alertes inutiles).
- **Contraintes du jalon** : respecter le découpage par pompe, tout régler sur la validation, le test ne sert **qu'une seule fois** à la fin.

## 2. Les données disponibles

Pour chaque cycle, nous disposons de :

- `cycle` : numéro du cycle dans la vie de la pompe
- `charge` : régime de fonctionnement (0.6, 0.8 ou 1.0)
- `capteur_0` à `capteur_7` : 8 signaux de capteurs
- `rul` : RUL calculée depuis les données d'entraînement (connue jusqu'à la panne)
- `rul_cap` / `cible` : RUL plafonnée à 125 et normalisée entre 0 et 1 pour l'entraînement du réseau

Les 10 pompes d'entraînement sont suivies **jusqu'à la panne**. Pour les 20 pompes de test, l'historique est tronqué et la RUL finale `rul_fin_test` est cachée (on ne l'utilise qu'à l'évaluation finale).

## 3. Cible et plafonnement à 125

Dans la phase saine, quand la RUL est encore très grande (> 125), le comportement est peu informatif pour prévoir l'approche de la panne. 

On définit donc :

$$y = \frac{\min(\mathrm{RUL}, 125)}{125} \in [0,1]$$

- **RUL > 125** → $y = 1.0$ (zone "sain", peu discriminante)
- **RUL diminue vers 0** → $y$ diminue vers 0 (proche de la panne)
- **Sortie sigmoïde** du MLP compatible avec $[0,1]$

Ce **plafonnement** permet au réseau de se concentrer sur la **phase critique** (les 125 derniers cycles) plutôt que d'essayer de modéliser une longue période saine.

## 4. Découpage par pompe

**Règle fondamentale** : on ne mélange **pas** les cycles d'une même pompe entre entraînement et validation.

- **Validation** : pompes 502 et 509 (2 pompes)
- **Entraînement** : les 8 autres (501, 503, 504, 505, 506, 507, 508, 510)
- Total : 2378 cycles train / 469 cycles validation

L'objectif est de vérifier que le modèle **généralise à de nouvelles pompes**, pas à de nouveaux cycles de la même pompe.

## 5. Normalisation des entrées

Les capteurs ont des **échelles très différentes** (ex. `capteur_7` ~2940, `capteur_4` ~3). Sans normalisation, certains capteurs domineraient l'apprentissage.

On utilise ici **`keras.layers.Normalization`** apprise **uniquement sur l'ensemble d'entraînement**.

### 5.1 Normalisation par régime (référence du sujet)

On observe que la distribution des capteurs **dépend du régime de charge** $charge \in \{0.6, 0.8, 1.0\}$ (figure 3.7).

Pour chaque régime $c$ :

- On sélectionne **uniquement les cycles sains** de l'entraînement avec $\mathrm{RUL} \ge 125$ : $\{(x_i) \mid pompe \in train,\ charge=c,\ rul\_cap==125\}$
- On calcule $\mu^{(c)}_j, \sigma^{(c)}_j$ sur ces cycles sains
- Pour un cycle de régime $c$, on normalise : $\tilde{x}_j = \frac{x_j - \mu^{(c)}_j}{\sigma^{(c)}_j + \epsilon}$

**Intérêt** : on centre/réduit **relativement au fonctionnement "nominal"** de ce régime. Cela permet de mieux faire ressortir les **écarts** liés à la dégradation.

### 5.2 Normalisation globale (ablation Q2)

On calcule $\mu_j, \sigma_j$ **sur tous les cycles de l'entraînement** (tous régimes confondus), puis on normalise tous les cycles avec ces mêmes moyennes/écarts-types.

**Comparaison Q2** : sur ce jeu type B, la **normalisation globale** donne une meilleure RMSE (14.56) que la normalisation par régime (16.09). Néanmoins, **la normalisation par régime reste la référence du sujet** car elle est plus cohérente physiquement (le niveau des capteurs dépend bien du régime).

**Règle stricte** : on **n'adapte jamais** les couches de normalisation sur la validation/test.

## 6. Modèle MLP de référence

Structure retenue : **9 → 64 → 32 → 1**

- **Entrées** : 8 capteurs normalisés + `charge` (numérique dans la version référence)
- **Couches cachées** : ReLU
- **Sortie** : **sigmoïde** (bornée dans $[0,1]$)
- **Paramètres totaux** : $9\times64 + 64 + 64\times32 + 32 + 32\times1 + 1 = 640 + 64 + 2080 + 32 + 33 = 2753$
- **Optimiseur** : Adam
- **Batch size** : 256
- **Époques** : 150
- **Perte (référence)** : MSE

## 7. Métriques utilisées

### 7.1 RMSE (en cycles)

$$\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_i (\hat{RUL}_i - RUL_i)^2},\quad \hat{RUL}_i = \mathrm{clip}(125 * \hat{y}_i, 0, 125)$$

Mesure globale de l'erreur absolue moyenne en cycles.

### 7.2 Score NASA C-MAPSS (asymétrique)

C'est le **score asymétrique** du challenge NASA C-MAPSS. Il pénalise **beaucoup plus** une prédiction **tardive** ($\hat{RUL} > RUL$, on prédit "reste plus longtemps" alors que c'est faux) qu'une prédiction **précoce** ($\hat{RUL} < RUL$, on anticipe un peu trop tôt).

Soit $d_i = \hat{y}_i^{(cycles)} - y_i^{(cycles)} = (\hat{RUL}_i - RUL_i)$

$$s_i = \begin{cases}
e^{-d_i/13} - 1, & d_i < 0\quad (\text{prédiction trop précoce})\\[4pt]
e^{\,d_i/10} - 1, & d_i \ge 0\quad (\text{prédiction trop tardive})
\end{cases},\qquad Score = \sum_i s_i$$

**Valeurs clés** :

| $d$ | $s(d)$ | Interprétation |
|-----:|--------:|----------------|
| -30 | ~9.05 | trop précoce de 30 cycles |
| -10 | ~1.16 | trop précoce de 10 cycles |
| +10 | ~1.72 | trop tardive de 10 cycles |
| +30 | ~19.09 | trop tardive de 30 cycles |

On voit clairement : $s(+30) \gg s(-10)$. **Une erreur tardive est bien plus coûteuse**. C'est pourquoi ce score justifie l'utilisation d'une **perte asymétrique**.

### 7.3 Anticipation et fausses alertes

On regarde ces indicateurs **sur les points de coupe** de la validation ($RUL \in \{5,15,30,45,60,90,120\}$) pour se rapprocher d'un usage "alerte".

- **Anticipation** (taux de bonnes anticipations) : proportion des cas où $RUL_{vrai} \le 30$ et $\hat{RUL} < RUL_{vrai}$ (on anticipe avant/après la vraie valeur, dans un sens "sûr" vis-à-vis du besoin). Plus précisément dans le notebook : on compare $\hat{RUL} < SEUIL\_ALERTE$ quand $RUL_{vrai} \le SEUIL\_ALERTE$.
- **Fausses alertes (FA)** : proportion de cas où $\hat{RUL} < SEUIL\_ALERTE$ alors que $RUL_{vrai} > SEUIL\_FA$ (on alerte alors que la pompe est encore loin de la panne).

**Objectif visé** (système final type A) : Anticipation $\ge 0.90$, FA $\le 0.10$. Sur notre validation type B, on vise déjà à se rapprocher de ces tendances.

## 8. Fenêtre glissante $k$ cycles (question Q4)

Jusqu'ici on utilise **un seul cycle** ($k=1$) : entrées = valeurs des 8 capteurs + charge à ce cycle.

L'idée d'une **fenêtre aplatie** ($k>1$) est d'utiliser **les $k$ derniers cycles** pour lisser le bruit et mieux capter une tendance.

**Principe** :

- Pour un instant $t$, on concatène les vecteurs des cycles $t-k+1, ..., t$ : on obtient $9*k$ entrées (8 capteurs + charge répétés)
- On garde le même $y = \min(RUL_t, 125)/125$ (cible associée au cycle courant)
- Même normalisation par régime appliquée cycle par cycle

**Valeurs testées** : $k \in \{1,5,15,30\}$

**Résultat observé** :

- Le **bruit intra-pompe** (écart-type des prédictions sur une même pompe) **augmente** avec $k$ (33.3 → 38.1 → 42.1 → 36.9)
- Le score se dégrade globalement quand $k$ augmente
- $k=1$ reste le **moins bruyant** et globalement le plus stable sur ce petit jeu

Conclusion Q4 : **le fenêtrage n'apporte pas de gain ici**. $k=1$ est retenu.

## 9. Perte asymétrique (question Q3)

Au lieu de minimiser uniquement l'écart quadratique (MSE), on veut pénaliser **davantage les prédictions tardives** pour coller à la logique maintenance + au score NASA.

On utilise une **perte asymétrique** inspirée du score, appliquée directement sur l'erreur $d = \hat{y}_{cycles} - y_{cycles}$.

Formulation utilisée dans le notebook (même forme que le score, pondérée) :

$$\mathcal{L}_{asym} = \frac{1}{N}\sum_i \begin{cases}
e^{-d_i/13} - 1, & d_i < 0\\[4pt]
e^{\,d_i/10} - 1, & d_i \ge 0
\end{cases}$$

**Effet** :

- Gradient plus fort quand $d>0$ (tardif)
- Incite le réseau à **anticiper un peu plutôt que de retarder**
- Sur validation : score passe de **3512** (MSE) à **2394** (asymétrique), anticipation passe de 0.83 à **1.00**, au prix d'une RMSE un peu plus élevée (16.09 → 18.52)

**Conclusion Q3** : la perte asymétrique est **préférable** du point de vue "maintenance prédictive" car elle minimise le score pénalisant les erreurs tardives.

## 10. Ablations complémentaires

### 10.1 Sélection des capteurs (section 15.1)

L'analyse de corrélation montre :

| Capteur | $\rho(\text{capteur}, RUL)$ |
|---------|----------------------------:|
| capteur_4 | -0.561 |
| capteur_5 | -0.360 |
| capteur_0 | -0.345 |
| capteur_6 | -0.298 |
| capteur_1 | -0.006 |
| capteur_3 | +0.066 |
| capteur_2 | +0.077 |
| capteur_7 | +0.082 |

Seuls **capteur_4, capteur_5** (et dans une moindre mesure 0,6) portent un **vrai signal** de dégradation. **capteur_1,2,3,7** ont $\lvert\rho\rvert < 0.09$ : ce sont quasiment du bruit.

On teste 3 jeux d'entrées (moyenne sur 3 graines, même recette MSE/régime/k=1) :

| Entrées | RMSE | Score | Anticipation | FA |
|--------:|-----:|------:|-------------:|---:|
| 9 (référence) | 16.01 | 3521 | 0.72 | 0.00 |
| 5 (sans c1,c2,c3,c7) | 14.57 | 2020 | 1.00 | 0.00 |
| 3 (c4,c5 + charge) | **13.50** | **1833** | **1.00** | 0.00 |

**Conclusion** : **supprimer les capteurs peu corrélés améliore nettement** (score divisé par ~2). Ils n'apportent rien d'informatif et ajoutent du bruit.

### 10.2 Encodage de la charge : numérique vs one-hot (section 15.2)

La `charge` ne prend **que 3 valeurs exactes** : 0.6, 0.8, 1.0.

- **Numérique (référence)** : on passe `charge` tel quel (1 valeur). Le MLP impose implicitement une **relation ordonnée/linéaire** entre 0.6, 0.8, 1.0.
- **One-hot** : on remplace par 3 indicatrices $(c_{0.6}, c_{0.8}, c_{1.0}) \in \{0,1\}^3$. L'entrée passe de **9 à 11** colonnes. Les 8 capteurs restent normalisés par régime (on utilise toujours `charge` pour choisir la couche de normalisation).

Résultats (moyenne sur plusieurs graines) :

| Encodage | Entrées | RMSE | Score | Anticipation | FA |
|----------|--------:|-----:|------:|-------------:|---:|
| Numérique | 9 | 16.01 | 3521 | 0.72 | 0.00 |
| **One-hot** | 11 | **15.4** | **2896** | **0.78** | 0.00 |

**Conclusion** : le **one-hot est légèrement meilleur**. Puisqu'il n'y a que 3 modalités, c'est plus juste de **ne pas supposer de continuité linéaire** entre ces 3 régimes.

## 11. Évaluation finale sur le test

Le modèle retenu pour le test est **le modèle de référence** : normalisation par régime, $k=1$, **perte MSE**, 9 entrées (numériques), entraîné sur train+validation selon le protocole (test **uniquement à la fin**).

Résultats :

- **RMSE test** = **29.07 cycles**
- **Score test** = **3089.5**

Principaux écarts (tardifs/optimistes) : pompe 612 (+78.9), 616 (+53.4), 607 (+47.3), 619 (+29.9), 613 (+29.1).

**Remarque** : anticipation/FA ne sont **pas mesurées** sur le test (pas de points de coupe identiques dans le même sens). Elles sont évaluées sur la validation.

## 12. Cibles du cahier des charges vs résultats

| Exigence | Cible (type A, système final) | Notre test (type B) | Commentaire |
|----------|------------------------------:|--------------------:|-------------|
| RMSE RUL | <= 23 cycles | 29.07 | Dépassé (~+6) |
| Score asymétrique | <= 1200 | 3089.5 | ~2.6x |
| Anticipation | >= 0.90 | 1.00 (validation, perte asym.) | Atteignable |
| Fausses alertes | <= 0.10 | 0.00 (validation) | Conforme |

**Point important** : le PDF précise clairement que **ces cibles ne sont pas exigées à la fin du Jalon 1**. Elles concernent le **système final (type A)**. L'objectif du Jalon 1 est de **construire un pipeline propre, justifier ses choix, et expliquer les écarts**.

**Explication des écarts** :

1. **Données type B très réduites** (10 pompes train) vs flotte type A beaucoup plus large
2. **Modèle testé = référence MSE**. Avec **perte asymétrique**, le score validation descend à 2394 et anticipation à 1.00
3. **Ablations** montrent qu'on peut encore améliorer : ne garder que c4,c5+charge (**score validation 1833**) ou utiliser **one-hot** sur la charge
4. **Erreurs tardives** sur RUL moyennes fortement pénalisées par le score NASA

## 13. Synthèse des choix retenus

| Choix | Justification |
|-------|---------------|
| **Plafonnement RUL à 125** | Se concentre sur la phase critique (proche panne). Stabilise l'apprentissage. |
| **Découpage par pompe** | Évite toute fuite. Force à généraliser à de nouvelles pompes. |
| **Normalisation par régime** | Cohérente avec l'effet charge observé. Référence du sujet. (globale meilleure RMSE mais moins physique) |
| **Sortie sigmoïde + cible [0,1]** | Naturelle pour $\min(RUL,125)/125$. Bornée. |
| **MLP 9-64-32-1 (2753 params)** | Suffisant pour ce problème, simple, traçable. Vérifié. |
| **k = 1** | Moins bruyant, plus stable. Fenêtrage >1 n'apporte rien ici. |
| **Perte asymétrique** | Colle au coût réel (tardif plus cher) et au score NASA. Meilleur compromis maintenance. |
| **Sélection capteurs** | Retirer c1,c2,c3,c7 est très bénéfique (réduction score). À envisager pour jalon suivant. |
| **One-hot charge** | Plus juste car 3 modalités disjointes (évite fausse linéarité). Léger gain cohérent. |

## 14. Conclusion

Le Jalon 1 est **complet et reproductible**. Nous avons :

- Fait un **EDA structuré** (distributions, corrélations, effet régime, figure de priorisation)
- Mis en œuvre le pipeline imposé (plafonnement, découpage par pompe, normalisation Keras)
- Répondu à **Q1-Q4** avec justifications
- Effectué **2 ablations complémentaires** pertinentes
- Vérifié les **2 points théoriques** (2753 paramètres, asymétrie du score)
- Évalué **une seule fois** sur le test
- Tenus à jour **journal** + **rapport de synthèse**

L'essentiel est atteint : **comprendre les concepts (normalisation par régime, asymétrie NASA, fenêtre k, découpage par pompe)** et **savoir justifier ses choix**. Les écarts aux cibles type A sont expliqués et des pistes concrètes existent pour les réduire.
