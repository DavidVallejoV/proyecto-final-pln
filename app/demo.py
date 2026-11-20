"""Demo del clasificador de reseñas por sede (caso Bodytech).

Ejecutar desde la raíz del repositorio:
    streamlit run app/demo.py

Muestra el flujo completo: entrada → limpieza y anonimización → controles de entrada → probabilidades →
decisión (automática, revisión humana o alerta al coordinador) → evidencia. Cada clasificación queda
registrada en resultados/registro_demo.csv SIN el texto de la reseña (solo su huella), por privacidad.
"""
import csv
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import config  # noqa: E402
from src.inferencia import ALERTA, AUTOMATICA, REVISION, Clasificador  # noqa: E402

EJEMPLOS = {
    'Queja clara': ('Pésima experiencia', 'Las duchas nunca tienen agua caliente, las máquinas están dañadas hace semanas y nadie '
                                          'responde en recepción. Estoy pensando en cancelar el plan.'),
    'Elogio': ('Excelente sede', 'El personal es muy amable, todo está limpio y las clases son excelentes. La recomiendo sin duda.'),
    'Opinión mixta': ('Bien, pero mejorable', 'La ubicación es buena y los entrenadores atentos, pero en la noche no hay cupo en '
                                               'las clases y los vestidores están sucios.'),
    'Ironía': ('Genial', 'Genial, solo tardaron dos horas en atenderme. Maravilloso.'),
    'Con datos personales': ('Excelente atención', 'Quiero agradecer a Andrea de recepción y al entrenador Julián, me ayudaron '
                                                   'muchísimo con el cambio de plan. Mi correo es ana.perez@correo.com y mi celular '
                                                   '3001234567.'),
    'Otro idioma': ('Bad', 'The showers were dirty and the staff was rude. We will not come back.'),
    'Fuera de dominio': ('', 'La fotosíntesis depende de la luz solar y del agua.'),
    'Vacía': ('', '   '),
}
COLOR = {AUTOMATICA: 'success', REVISION: 'warning', ALERTA: 'error'}


@st.cache_resource
def cargar_clasificador():
    return Clasificador()


def registrar(resultado, titulo, resena):
    """Trazabilidad sin guardar el texto: huella SHA-256 de la entrada, versión del modelo y decisión."""
    ruta = config.DIR_RESULTADOS / 'registro_demo.csv'
    nueva = not ruta.exists()
    with open(ruta, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if nueva:
            w.writerow(['fecha', 'huella_entrada', 'version_modelo', 'clase', 'confianza', 'decision'])
        huella = hashlib.sha256(f'{titulo}|{resena}'.encode('utf-8')).hexdigest()[:16]
        w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), huella, resultado['version_modelo'], resultado['clase'],
                    None if resultado['confianza'] is None else round(resultado['confianza'], 4), resultado['decision']])


st.set_page_config(page_title='Clasificador de reseñas por sede', layout='wide')
clf = cargar_clasificador()
pol = clf.politica

st.title('Clasificador de reseñas por sede')
num = lambda x, d=2: f'{x:.{d}f}'.replace('.', ',')
st.caption(f'Prototipo académico · caso Bodytech (entrenado con reseñas de hoteles como proxy) · modelo {pol["modelo"]} '
           f'· versión {clf.version} · umbrales elegidos en validación: confianza ≥ {num(pol["umbral_confianza"])}, '
           f'alerta si P(negativa) ≥ {num(pol["tau_negativa"], 3)}')

entrada, salida = st.columns([3, 2], gap='large')
with entrada:
    ejemplo = st.selectbox('Cargar un ejemplo', ['(escribir mi propia reseña)'] + list(EJEMPLOS))
    t0, r0 = EJEMPLOS.get(ejemplo, ('', ''))
    titulo = st.text_input('Título de la reseña', value=t0)
    resena = st.text_area('Reseña', value=r0, height=180)
    if st.button('Clasificar', type='primary'):
        st.session_state['resultado'] = clf.clasificar(titulo, resena)
        registrar(st.session_state['resultado'], titulo, resena)

with salida:
    r = st.session_state.get('resultado')
    if r is None:
        st.info('Escriba una reseña o cargue un ejemplo y presione «Clasificar».')
    else:
        getattr(st, COLOR[r['decision']])(f"**Decisión: {r['decision']}**  \n{r['motivo']}")
        if r['clase'] is not None:
            c1, c2 = st.columns(2)
            c1.metric('Clase predicha', r['clase'])
            c2.metric('Confianza', num(r['confianza']))
            st.bar_chart(pd.Series(r['probabilidades'], name='probabilidad'), horizontal=True, height=180)
        for alerta in r['alertas']:
            st.warning(alerta)
        if r['evidencia']:
            st.markdown('**Evidencia:** términos que más empujan hacia la clase predicha')
            st.dataframe(pd.DataFrame(r['evidencia'], columns=['término', 'aporte']).round(3), hide_index=True,
                         use_container_width=True)
        with st.expander('Texto que recibió el modelo (limpio y anonimizado)'):
            st.write(r['texto_procesado'] or '(vacío)')

st.divider()
st.caption('La decisión final sobre una reseña la toma una persona cuando el sistema lo indica. Este prototipo no debe usarse con '
           'reseñas reales de clientes sin reentrenarlo con datos propios y sin la aprobación del dueño del modelo.')
