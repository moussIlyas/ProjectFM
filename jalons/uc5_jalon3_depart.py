# Étude de cas 5 - Jalon 3 : code de démarrage
# À exécuter à la suite du notebook de démarrage, exécuté en entier.
# Point de départ : le MLP de référence « mlp » du notebook (type A).
from sklearn.model_selection import GroupKFold

def rmse_cycles(model, df_norm):
    # RMSE (en cycles) sur tous les cycles d'historiques complets
    X, y = xy(df_norm)
    d = model.predict(X, verbose=0).ravel() - y
    return float(RUL_MAX * np.sqrt(np.mean(d ** 2)))

# 1. Type B normalisé avec les statistiques du type A, ou avec les siennes
NB_statsA = {s: normaliser(df, statsA) for s, df in TB.items()}
variantes = [("normalisation de A", NB_statsA), ("normalisation de B", NB)]
for nom, N in variantes:
    print("modèle A tel quel,", nom, ":", rmse_cycles(mlp, N["train"]))

# 2. Validation croisée par pompe sur les dix historiques du type B
pompes_B = NB["train"].pompe.values
plis = list(GroupKFold(n_splits=5).split(NB["train"], groups=pompes_B))

# 3. Spectrogrammes des deux types et étiquettes « sain » / modes
def etiquette_mode(D, s, seuil=60):
    # 0 = sain, 1 = roulement, 2 = désalignement, 3 = cavitation
    y = np.where(D["rul_" + s] > seuil, 0, 1 + D["mode_" + s])
    return y.astype("int32")

SPEC_A = A["spectro_train"][..., None].astype("float32") / 255.0
SPEC_B = B["spectro_train"][..., None].astype("float32") / 255.0
Y_MODE_A, Y_MODE_B = etiquette_mode(A, "train"), etiquette_mode(B, "train")
