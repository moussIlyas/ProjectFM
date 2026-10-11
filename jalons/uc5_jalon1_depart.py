# Étude de cas 5 - Jalon 1 : code de démarrage
# À exécuter à la suite du notebook de démarrage, exécuté en entier.

# 1. Erreur en fonction de la RUL réelle (tous les cycles de validation)
Xv1, yv1 = xy(NA["val"])
ecart = RUL_MAX * (mlp.predict(Xv1, verbose=0).ravel() - yv1)
plt.scatter(RUL_MAX * yv1, ecart, s=2, alpha=0.3)
plt.axhline(0, color="r", ls=":")
plt.xlabel("RUL réelle (plafonnée)")
plt.ylabel("RUL prédite - RUL réelle")
plt.show()

# 2. Normalisation globale (sans tenir compte du régime), pour l'ablation
mu_g, sd_g = TA["train"][CAPTEURS].mean(), TA["train"][CAPTEURS].std()

def normaliser_global(df):
    out = df.copy()
    out[CAPTEURS] = (out[CAPTEURS] - mu_g) / sd_g
    return out

# 3. Fenêtre des k derniers cycles, aplatie (k x 9 entrées) ; au début
#    de l'historique, le premier cycle est répété
def fenetres_aplaties(df, k):
    X, y, dernier = [], [], []
    for _, g in df.sort_values(["pompe", "cycle"]).groupby("pompe"):
        v = g[CAPTEURS + ["charge"]].values.astype("float32")
        v = np.vstack([np.repeat(v[:1], k - 1, axis=0), v])
        X.append(np.stack([v[i:i + k].ravel() for i in range(len(g))]))
        dernier.append(np.arange(len(g)) == len(g) - 1)
        if "rul" in g:
            y.append(np.minimum(g.rul.values, RUL_MAX) / RUL_MAX)
    y = np.concatenate(y).astype("float32") if y else None
    return np.concatenate(X), y, np.concatenate(dernier)

X5, y5, _ = fenetres_aplaties(NA["train"], 5)
Xt5, _, fin_t5 = fenetres_aplaties(NA["test"], 5)   # prédire sur Xt5[fin_t5]
print(X5.shape, Xt5[fin_t5].shape)
