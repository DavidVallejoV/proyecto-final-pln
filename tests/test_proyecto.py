"""Pruebas del preprocesamiento, la partición y la política de decisión.

Ejecutar desde la raíz del repositorio:  pytest -q
Las pruebas de la política usan el modelo y config/politica.json que genera el notebook principal.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.particion import partir_por_grupo  # noqa: E402
from src.texto import a_secuencia, enmascarar_sede, normalizar  # noqa: E402


def test_normalizar_enmascara_datos_personales():
    s = normalizar('Escríbanme a ana.perez@correo.com o al 300 123 4567, info en https://ejemplo.com …')
    assert '[EMAIL]' in s and '[TEL]' in s and '[URL]' in s
    assert 'ana.perez' not in s and '4567' not in s and '…' not in s


def test_normalizar_respeta_arroba_inclusiva_y_tildes():
    s = normalizar('Gracias a tod@s, la atención fue excelente')
    assert 'tod@s' in s and 'atención' in s


def test_enmascarar_sede():
    s = enmascarar_sede('Me encantó el Hotel Casa de la Plata, volveré', 'H10_Casa_de_la_Plata')
    assert '[SEDE]' in s and 'Plata' not in s


def test_a_secuencia_separa_signos():
    assert a_secuencia('Muy bien.Volveré!!') == 'muy bien volveré'


def test_particion_sin_sedes_compartidas():
    rng = np.random.default_rng(0)
    tabla = pd.DataFrame({'sede_id': np.repeat([f'S{i:02d}' for i in range(40)], 10),
                          'sentimiento': rng.choice(['negativa', 'neutra', 'positiva'], 400)})
    a, b = partir_por_grupo(tabla, 0.25, semilla=42)
    assert set(tabla.iloc[a].sede_id).isdisjoint(set(tabla.iloc[b].sede_id))
    assert 0.15 < len(b) / len(tabla) < 0.35


requiere_modelo = pytest.mark.skipif(not (config.RUTA_BASELINE.exists() and config.RUTA_POLITICA.exists()),
                                     reason='Ejecute primero el notebook principal (genera el modelo y la política)')


@pytest.fixture(scope='module')
def clf():
    from src.inferencia import Clasificador
    return Clasificador(anonimizar_nombres=False)


@requiere_modelo
def test_resena_vacia_va_a_revision(clf):
    from src.inferencia import REVISION
    r = clf.clasificar('', '    ')
    assert r['decision'] == REVISION and r['clase'] is None


@requiere_modelo
def test_otro_idioma_va_a_revision(clf):
    from src.inferencia import REVISION
    r = clf.clasificar('Bad', 'The showers were dirty and the staff was rude. We will not come back.')
    assert r['decision'] == REVISION and r['alertas']


@requiere_modelo
def test_queja_clara_genera_alerta(clf):
    from src.inferencia import ALERTA
    r = clf.clasificar('Pésimo', 'Todo sucio, el personal fue grosero y nadie nos atendió. Horrible, no volveremos nunca.')
    assert r['decision'] == ALERTA


@requiere_modelo
def test_elogio_claro_es_automatico(clf):
    from src.inferencia import AUTOMATICA
    r = clf.clasificar('Excelente', 'Todo perfecto, el personal muy amable y todo muy limpio. Muy recomendable, volveremos sin duda.')
    assert r['decision'] == AUTOMATICA and r['clase'] == 'positiva'
