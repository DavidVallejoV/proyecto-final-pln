"""Clasificación de una reseña con la política de decisión del proyecto.

Flujo: texto → limpieza y anonimización → controles de entrada (vacío, idioma, palabras desconocidas)
→ probabilidades del modelo → decisión:
  - «alerta al coordinador»  si P(negativa) ≥ τ (regla de la clase crítica)
  - «revisión humana»        si la entrada no es válida, la confianza es baja o la predicción es neutra
  - «automática»             en el resto de los casos
Los umbrales vienen de `config/politica.json`, que el notebook principal genera con el conjunto de validación.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.modelos import evidencia, palabras_desconocidas
from src.texto import Anonimizador, a_secuencia, detectar_idioma, normalizar

AUTOMATICA, REVISION, ALERTA = 'automática', 'revisión humana', 'alerta al coordinador de sede'


def _n(x, d=2):
    """Número con coma decimal para los mensajes."""
    return f'{x:.{d}f}'.replace('.', ',')


def huella(ruta, n=12):
    """Primeros caracteres del SHA-256 del archivo: identifica la versión exacta del modelo."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()[:n]


class Clasificador:
    def __init__(self, ruta_modelo=config.RUTA_BASELINE, ruta_politica=config.RUTA_POLITICA,
                 anonimizar_nombres=True):
        self.modelo = joblib.load(ruta_modelo)
        self.politica = json.loads(Path(ruta_politica).read_text(encoding='utf-8'))
        self.version = huella(ruta_modelo)
        self.clases = list(self.modelo.classes_)
        self.anonimizador = None
        if anonimizar_nombres:
            try:
                lexico = Path(config.RUTA_LEXICO).read_text(encoding='utf-8').split()
                self.anonimizador = Anonimizador(lexico, excluir=self.politica.get('excluir_de_nombres', []))
            except Exception:          # sin spaCy o sin el léxico: se enmascaran solo correos, teléfonos y enlaces
                self.anonimizador = None

    def preparar(self, titulo, resena):
        """Mismo texto de entrada que en el entrenamiento: título + '. ' + reseña, limpio y anonimizado."""
        titulo, resena = normalizar(titulo or ''), normalizar(resena or '')
        texto = f'{titulo}. {resena}' if titulo else resena
        if self.anonimizador is not None and texto:
            texto = self.anonimizador.enmascarar([texto])[0][0]
        return texto

    def clasificar(self, titulo, resena):
        p = self.politica
        texto = self.preparar(titulo, resena)
        r = {'texto_procesado': texto, 'version_modelo': self.version, 'alertas': [],
             'probabilidades': None, 'clase': None, 'confianza': None, 'evidencia': []}

        if not a_secuencia(texto):
            r.update(decision=REVISION, motivo='La reseña está vacía: no hay nada que clasificar.')
            return r

        idioma = detectar_idioma(resena or texto)
        if idioma not in p['idiomas_permitidos']:
            r['alertas'].append(f'Idioma detectado: «{idioma}». El modelo solo se validó en español.')
        desconocidas = palabras_desconocidas(self.modelo, texto)
        r['palabras_desconocidas'] = desconocidas
        if desconocidas > p['max_palabras_desconocidas']:
            r['alertas'].append(f'{desconocidas:.0%} de las palabras no están en el vocabulario del modelo '.replace('%', ' %') +
                                f'(límite {p["max_palabras_desconocidas"]:.0%}): posible texto fuera de dominio.'.replace('%', ' %'))

        prob = self.modelo.predict_proba([texto])[0]
        i = int(np.argmax(prob))
        r.update(probabilidades=dict(zip(self.clases, map(float, prob))), clase=self.clases[i], confianza=float(prob[i]))
        p_neg = float(prob[self.clases.index(config.CLASE_CRITICA)])
        r['evidencia'] = evidencia(self.modelo, texto, r['clase'])

        if r['alertas']:
            r.update(decision=REVISION, motivo='La entrada no pasó los controles: la decide una persona.')
        elif p_neg >= p['tau_negativa']:
            r.update(decision=ALERTA, motivo=f'Posible queja: P(negativa) = {_n(p_neg)} ≥ τ = {_n(p["tau_negativa"], 3)}.')
        elif r['confianza'] < p['umbral_confianza']:
            r.update(decision=REVISION, motivo=f'Confianza {_n(r["confianza"])} menor que el umbral {_n(p["umbral_confianza"])}.')
        elif r['clase'] == 'neutra' and p.get('neutra_a_revision', True):
            r.update(decision=REVISION, motivo='Predicción neutra: es la clase menos confiable, la revisa una persona.')
        else:
            r.update(decision=AUTOMATICA, motivo=f'Confianza {_n(r["confianza"])} ≥ {_n(p["umbral_confianza"])} y P(negativa) = {_n(p_neg)} < τ.')
        return r
