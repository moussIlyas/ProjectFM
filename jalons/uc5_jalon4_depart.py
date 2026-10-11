# Étude de cas 5 - Jalon 4 : code de démarrage
# À exécuter à la suite du notebook de démarrage, exécuté en entier.
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view

def sequences(df, W, remplissage="repetition"):
    # une fenêtre (W cycles x 9 variables) par cycle observé ; au début de
    # l'historique : répétition du 1er cycle, ou zéros (layers.Masking)
    X, y, dernier = [], [], []
    for _, g in df.sort_values(["pompe", "cycle"]).groupby("pompe"):
        v = g[CAPTEURS + ["charge"]].values.astype("float32")
        if remplissage == "repetition":
            debut = np.repeat(v[:1], W - 1, axis=0)
        else:
            debut = np.zeros((W - 1, v.shape[1]), "float32")
        v = np.vstack([debut, v])
        X.append(sliding_window_view(v, W, axis=0).transpose(0, 2, 1))
        dernier.append(np.arange(len(g)) == len(g) - 1)
        if "rul" in g:
            y.append(np.minimum(g.rul.values, RUL_MAX) / RUL_MAX)
    y = np.concatenate(y).astype("float32") if y else None
    return np.concatenate(X), y, np.concatenate(dernier)

W = 30
X_seq, y_seq, _ = sequences(NA["train"], W)
Xv_seq, yv_seq, _ = sequences(NA["val"], W)
Xt_seq, _, fin_t = sequences(NA["test"], W)    # prédire sur Xt_seq[fin_t]
print(X_seq.shape, Xt_seq[fin_t].shape)

def normes_gradient(model, X, y):
    # norme du gradient de chaque variable, sur un lot
    with tf.GradientTape() as tape:
        perte = keras.ops.mean((model(X, training=True)[:, 0] - y) ** 2)
    grads = tape.gradient(perte, model.trainable_variables)
    return {v.path: float(tf.norm(g))
            for v, g in zip(model.trainable_variables, grads)}
