"""Partición sin fuga: todas las reseñas de una sede quedan en la misma parte."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def partir_por_grupo(tabla, proporcion, grupo='sede_id', etiqueta='sentimiento', clases=None, semilla=42, intentos=200):
    """Separa grupos completos (sedes) en dos lados.

    De `intentos` repartos al azar conserva el que deja más parecida la mezcla de clases en los dos
    lados y más cerca el tamaño pedido. Devuelve las posiciones (iloc) de cada lado.
    """
    clases = clases or sorted(tabla[etiqueta].unique())
    y = tabla[etiqueta].map({c: i for i, c in enumerate(clases)}).to_numpy()

    def mezcla(idx):
        return np.bincount(y[idx], minlength=len(clases)) / len(idx)

    mejor = None
    repartos = GroupShuffleSplit(n_splits=intentos, test_size=proporcion, random_state=semilla)
    for lado_a, lado_b in repartos.split(tabla, groups=tabla[grupo]):
        desbalance = np.abs(mezcla(lado_a) - mezcla(lado_b)).sum() + abs(len(lado_b) / len(tabla) - proporcion)
        if mejor is None or desbalance < mejor[0]:
            mejor = (desbalance, lado_a, lado_b)
    return mejor[1], mejor[2]
