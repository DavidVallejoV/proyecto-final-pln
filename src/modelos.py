"""Baseline TF-IDF + regresión logística y utilidades de explicación."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def crear_baseline(semilla=42, class_weight='balanced', **tfidf):
    """TF-IDF de palabras (unigramas y bigramas) + regresión logística balanceada, en un solo pipeline."""
    params = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    params.update(tfidf)
    return Pipeline([('tfidf', TfidfVectorizer(**params)),
                     ('clf', LogisticRegression(max_iter=1500, class_weight=class_weight, random_state=semilla))])


def evidencia(pipeline, texto, clase, k=6):
    """Términos del texto que más empujan hacia `clase` (peso TF-IDF × coeficiente de la clase).

    Devuelve una lista de (término, aporte), del mayor al menor aporte positivo.
    """
    tfidf, clf = pipeline.named_steps['tfidf'], pipeline.named_steps['clf']
    x = tfidf.transform([texto])
    i = list(clf.classes_).index(clase)
    columnas = x.indices
    aportes = x.data * clf.coef_[i, columnas]
    terminos = tfidf.get_feature_names_out()[columnas]
    orden = np.argsort(aportes)[::-1]
    return [(str(terminos[j]), float(aportes[j])) for j in orden[:k] if aportes[j] > 0]


def palabras_desconocidas(pipeline, texto):
    """Proporción de palabras del texto que no están en el vocabulario del baseline (solo unigramas)."""
    tfidf = pipeline.named_steps['tfidf']
    palabras = [w for w in tfidf.build_analyzer()(texto) if ' ' not in w]
    if not palabras:
        return 1.0
    return float(np.mean([w not in tfidf.vocabulary_ for w in palabras]))
