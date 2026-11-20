"""Limpieza, anonimización y utilidades de texto.

Las mismas funciones se usan al preparar el corpus, al entrenar y al clasificar en la demo,
para que el modelo vea siempre el texto transformado de la misma manera.
"""
import re
import unicodedata

PATRON_URL = re.compile(r'https?://\S+|www\.\S+')
PATRON_EMAIL = re.compile(r'[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}')          # no toca el «@» inclusivo (chic@s)
PATRON_TEL = re.compile(
    r'(?<!\d)(?:(?:\+34[\s.-]?)?[6789]\d{2}[\s.-]?\d{3}[\s.-]?\d{3}'     # España
    r'|(?:\+57[\s.-]?)?3\d{2}[\s.-]?\d{3}[\s.-]?\d{4})(?!\d)'             # celular de Colombia
)
PATRON_CORTE = re.compile(r'…\s*$')


def normalizar(texto):
    """Limpieza mínima, común a todos los modelos.

    Unifica la codificación (NFC), reemplaza enlaces, correos y teléfonos por un marcador,
    quita la marca de corte «…» que dejó el recolector y une espacios. Conserva palabras,
    mayúsculas, tildes y signos, porque llevan sentimiento.
    """
    s = unicodedata.normalize('NFC', str(texto))
    s = PATRON_URL.sub(' [URL] ', s)
    s = PATRON_EMAIL.sub(' [EMAIL] ', s)
    s = PATRON_TEL.sub(' [TEL] ', s)
    s = PATRON_CORTE.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def a_secuencia(texto):
    """Texto → tokens separados por un espacio, para las redes: minúsculas y cada signo reemplazado por un espacio."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]|_', ' ', str(texto).lower())).strip()


def sin_tildes(texto):
    """Quita las tildes y la diéresis (la ñ se conserva)."""
    return str(texto).translate(str.maketrans('áéíóúüÁÉÍÓÚÜ', 'aeiouuAEIOUU'))


def detectar_idioma(texto):
    """Código ISO del idioma (langdetect, con semilla fija). Devuelve '??' si no se puede detectar."""
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    try:
        return detect(str(texto))
    except Exception:
        return '??'


def palabras_de_nombre(nombre):
    """'H10_Casa_de_la_Plata' → ['casa', 'de', 'la', 'plata'] (se omiten códigos y la palabra hotel)."""
    partes = [p for p in re.split(r'[_\W]+', str(nombre).lower()) if p]
    return [p for p in partes if p not in {'hotel', 'hostal', 'apartamentos', 'h10'} and not p.isdigit()]


def enmascarar_sede(texto, nombre_sede):
    """Reemplaza por [SEDE] las menciones del nombre completo de la sede (sin distinguir mayúsculas ni tildes).

    Así el modelo no puede aprender la reputación de una sede por su nombre.
    """
    palabras = palabras_de_nombre(nombre_sede)
    if len(palabras) < 2:
        return texto
    patron = r'\b' + r'[\s_-]+'.join(re.escape(sin_tildes(p)) for p in palabras) + r'\b'
    plano = sin_tildes(texto)
    salida, fin = [], 0
    for m in re.finditer(patron, plano, flags=re.IGNORECASE):    # mismas posiciones: sin_tildes no cambia la longitud
        salida.append(texto[fin:m.start()]); salida.append('[SEDE]'); fin = m.end()
    salida.append(texto[fin:])
    return ''.join(salida)


class Anonimizador:
    """Reemplaza por [NOMBRE] los nombres de personas (por ejemplo, del personal de la sede).

    Usa el reconocedor de entidades de spaCy (es_core_news_sm), que en reseñas marca como persona
    muchas palabras comunes al inicio de frase («Además», «Ubicación»). Para ganar precisión, solo se
    enmascara una palabra marcada como persona si: empieza en mayúscula, tiene al menos 3 letras,
    casi nunca aparece en minúscula en el corpus de entrenamiento (`lexico_minusculas`) y no es un
    lugar ni parte del nombre de una sede (`excluir`).
    """

    def __init__(self, lexico_minusculas, excluir=(), modelo='es_core_news_sm'):
        import spacy
        self.nlp = spacy.load(modelo, disable=['parser', 'lemmatizer'])
        self.lexico = set(lexico_minusculas)
        self.excluir = {e.lower() for e in excluir}

    def _es_nombre(self, tok):
        w = tok.text
        return (w[:1].isupper() and w.isalpha() and len(w) >= 3
                and w.lower() not in self.lexico and w.lower() not in self.excluir)

    def enmascarar(self, textos, batch_size=256):
        """Devuelve (textos enmascarados, cantidad de palabras enmascaradas por texto, palabras enmascaradas)."""
        salida, conteos, palabras = [], [], []
        for doc in self.nlp.pipe(list(textos), batch_size=batch_size):
            marcar = {tok.i for ent in doc.ents if ent.label_ == 'PER' for tok in ent if self._es_nombre(tok)}
            partes, previo = [], False
            for tok in doc:
                if tok.i in marcar:
                    if not previo:                          # «Adrián Garrido» → un solo [NOMBRE]
                        partes.append('[NOMBRE]' + (tok.whitespace_ if tok.i + 1 not in marcar else ' '))
                    elif tok.i + 1 not in marcar:
                        partes[-1] = '[NOMBRE]' + tok.whitespace_
                    previo = True
                else:
                    partes.append(tok.text_with_ws); previo = False
            salida.append(''.join(partes))
            conteos.append(len(marcar))
            palabras.append([doc[i].text for i in sorted(marcar)])
        return salida, conteos, palabras


def construir_lexico(textos, minimo=2):
    """Palabras que aparecen escritas en minúscula al menos `minimo` veces: son palabras comunes, no nombres propios."""
    from collections import Counter
    cuenta = Counter(w for t in textos for w in re.findall(r'\b[^\W\d_]+\b', str(t)) if w.islower())
    return sorted(w for w, c in cuenta.items() if c >= minimo)
