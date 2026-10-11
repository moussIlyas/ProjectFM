# Étude de cas 5 - Jalon 2 : code de démarrage
# À exécuter à la suite du notebook de démarrage, exécuté en entier.
from scipy.stats import spearmanr

# 1. Spectrogrammes (N, 32, 32, 1) dans [0, 1] : lignes = fréquences
#    (0 à 500 Hz, de bas en haut), colonnes = temps (une seconde)
SPEC = {s: A["spectro_" + s][..., None].astype("float32") / 255.0
        for s in ["train", "val", "test"]}

# 2. Règle d'étiquetage, à justifier et à faire varier : « sain » tant
#    que la RUL dépasse un seuil, mode de défaillance ensuite
def etiquette_mode(D, s, seuil=60):
    # 0 = sain, 1 = roulement, 2 = désalignement, 3 = cavitation
    y = np.where(D["rul_" + s] > seuil, 0, 1 + D["mode_" + s])
    return y.astype("int32")

Y_MODE = {s: etiquette_mode(A, s) for s in ["train", "val"]}
print("effectifs par classe (train) :", np.bincount(Y_MODE["train"]))

# 3. Indicateur de santé non supervisé : cycles sains (RUL > 150, comme
#    pour la normalisation par régime), capteurs normalisés et charge
sains = NA["train"].rul.values > 150
X_sains = NA["train"][CAPTEURS + ["charge"]].values[sains]
print("cycles sains pour l'auto-encodeur :", X_sains.shape)
# après entraînement de l'auto-encodeur (fenêtres de cycles sains) :
# rho = spearmanr(erreur_de_reconstruction, rul).correlation
