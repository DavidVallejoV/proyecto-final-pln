"""Rutas y parámetros del proyecto.

Todo se define con rutas relativas a la raíz del repositorio. Los valores se pueden cambiar con
variables de entorno (ver `.env.example`); ninguno es un secreto.
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

try:                                    # el archivo .env es opcional
    from dotenv import load_dotenv
    load_dotenv(RAIZ / '.env')
except ImportError:
    pass

SEMILLA = int(os.getenv('SEMILLA', '42'))

RUTA_DATOS_CRUDOS = RAIZ / os.getenv('RUTA_DATOS_CRUDOS', 'datos/raw/Big_AHR.csv')
RUTA_CORPUS = RAIZ / os.getenv('RUTA_CORPUS', 'datos/procesado/corpus_bodytech_anonimizado.csv')
DIR_MODELOS = RAIZ / os.getenv('DIR_MODELOS', 'modelos')
DIR_RESULTADOS = RAIZ / os.getenv('DIR_RESULTADOS', 'resultados')
DIR_FIGURAS = DIR_RESULTADOS / 'figuras'
RUTA_POLITICA = RAIZ / os.getenv('RUTA_POLITICA', 'config/politica.json')

RUTA_BASELINE = DIR_MODELOS / 'baseline_tfidf_lr.joblib'
RUTA_LEXICO = DIR_MODELOS / 'lexico_minusculas.txt'       # usado por el anonimizador de nombres
DIR_RED = DIR_MODELOS / 'bilstm'                       # mejor variante de BiLSTM (por validación)

CLASES = ['negativa', 'neutra', 'positiva']
CLASE_CRITICA = 'negativa'
