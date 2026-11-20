"""Caso de valor del piloto (caso Bodytech simulado): supuestos, beneficio neto y sensibilidad.

Las tasas del modelo (qué parte de las quejas genera alerta, cuántas reseñas van a revisión) salen de la
evaluación en test del notebook principal (resultados/resumen_resultados.json). Los demás valores son
SUPUESTOS del equipo, a medir durante el piloto. La misma lógica está en docs/Caso_de_Valor.xlsx con fórmulas.

Ejecutar desde la raíz del repositorio:  python docs/caso_de_valor.py
"""
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]


def tasas_del_modelo(modelo='tfidf_lr (baseline)'):
    """Tasas de la política de un modelo en test (por defecto, el baseline), separadas para quejas reales y para el resto."""
    r = json.loads((RAIZ / 'resultados' / 'resumen_resultados.json').read_text(encoding='utf-8'))
    f = next(x for x in r['politica_tabla'] if x['modelo'] == modelo and x['parte'] == 'test')
    corpus = pd.read_csv(RAIZ / 'datos' / 'procesado' / 'corpus_bodytech_anonimizado.csv', usecols=['parte', 'sentimiento'])
    test = corpus[corpus.parte == 'test']
    n, n_neg = len(test), int((test.sentimiento == 'negativa').sum())
    alerta_neg = f['negativas con alerta']
    revision_neg = f['negativas que ve una persona (alerta o revisión)'] - alerta_neg
    alertas, revisiones = f['alertas por 1.000'] / 1000 * n, f['a revisión por 1.000'] / 1000 * n
    return {'alerta_si_queja': alerta_neg, 'revision_si_queja': revision_neg,
            'alerta_si_no_queja': (alertas - alerta_neg * n_neg) / (n - n_neg),
            'revision_si_no_queja': (revisiones - revision_neg * n_neg) / (n - n_neg)}


TASAS = tasas_del_modelo()

# (valor base, valor bajo, valor alto, descripción, formato)
SUPUESTOS = {
    'sedes': (86, 86, 86, 'Sedes Bodytech en Colombia (Portafolio, 21-ene-2026)', '{:.0f}'),
    'comentarios_por_sede_mes': (100, 50, 200, 'Comentarios por sede al mes (Google, app, encuestas)', '{:.0f}'),
    'pct_quejas': (0.14, 0.10, 0.20, 'Comentarios que son quejas', '{:.0%}'),
    'contacto_hoy': (0.30, 0.10, 0.50, 'Quejas que hoy se contactan en menos de 24 h', '{:.0%}'),
    'alerta_si_queja': (TASAS['alerta_si_queja'], 0.80, 0.95, 'Quejas que llegan como alerta (recall en test)', '{:.0%}'),
    'p_cancela_sin_contacto': (0.30, 0.20, 0.40, 'Afiliados con queja sin contacto que cancelan en 3 meses', '{:.0%}'),
    'reduccion_por_contacto': (0.25, 0.10, 0.40, 'Cancelaciones que evita un contacto a tiempo', '{:.0%}'),
    'cuota_mensual': (200_000, 150_000, 295_000, 'Cuota mensual promedio (pesos)', '${:,.0f}'),
    'meses_retenidos': (6, 3, 12, 'Meses adicionales que se queda un afiliado retenido', '{:.0f}'),
    'margen': (0.60, 0.40, 0.80, 'Margen de contribución de la cuota', '{:.0%}'),
    'min_por_alerta': (10, 5, 15, 'Minutos del coordinador por alerta (leer y contactar)', '{:.0f}'),
    'min_por_revision': (1, 0.5, 2, 'Minutos por reseña en revisión humana', '{:.1f}'),
    'costo_hora': (30_000, 25_000, 40_000, 'Costo de una hora de coordinador (pesos)', '${:,.0f}'),
    'costo_fijo_mes': (1_360_000, 1_000_000, 3_000_000, 'Costo fijo mensual: servidor, monitoreo y etiquetado (pesos)', '${:,.0f}'),
    'inversion_inicial': (60_000_000, 40_000_000, 120_000_000, 'Inversión inicial: integración, etiquetado y piloto (pesos)', '${:,.0f}'),
}
BASE = {k: v[0] for k, v in SUPUESTOS.items()}


def beneficio_neto(s, tasas=None):
    """Beneficio neto mensual en pesos y sus componentes (con las tasas del baseline, salvo que se pasen otras)."""
    tasas = tasas or TASAS
    comentarios = s['sedes'] * s['comentarios_por_sede_mes']
    quejas = comentarios * s['pct_quejas']
    no_quejas = comentarios - quejas
    alertas = quejas * s['alerta_si_queja'] + no_quejas * tasas['alerta_si_no_queja']
    revisiones = quejas * tasas['revision_si_queja'] + no_quejas * tasas['revision_si_no_queja']
    contactadas_extra = quejas * max(s['alerta_si_queja'] - s['contacto_hoy'], 0)
    retenidos = contactadas_extra * s['p_cancela_sin_contacto'] * s['reduccion_por_contacto']
    valor_retenido = s['cuota_mensual'] * s['meses_retenidos'] * s['margen']
    beneficio = retenidos * valor_retenido
    costo_alertas = alertas * s['min_por_alerta'] / 60 * s['costo_hora']
    costo_revision = revisiones * s['min_por_revision'] / 60 * s['costo_hora']
    costo = costo_alertas + costo_revision + s['costo_fijo_mes']
    neto = beneficio - costo
    return dict(comentarios=comentarios, quejas=quejas, alertas=alertas, revisiones=revisiones,
                automaticas=comentarios - alertas - revisiones, contactadas_extra=contactadas_extra, retenidos=retenidos,
                valor_retenido=valor_retenido, beneficio_mensual=beneficio, costo_alertas=costo_alertas,
                costo_revision=costo_revision, costo_fijo=s['costo_fijo_mes'], costo_mensual=costo, neto_mensual=neto,
                neto_anual=12 * neto - s['inversion_inicial'],
                recuperacion_meses=s['inversion_inicial'] / neto if neto > 0 else float('inf'),
                retenidos_equilibrio=costo / valor_retenido)


def es(texto):
    """Formato numérico en español: punto para miles, coma para decimales y espacio antes de %."""
    return texto.replace(',', '§').replace('.', ',').replace('§', '.').replace('%', ' %')


def sensibilidad():
    """Beneficio neto mensual al mover un supuesto a la vez a su valor bajo y alto (ordenado por impacto)."""
    filas = []
    for k, (b, bajo, alto, desc, fmt) in SUPUESTOS.items():
        if bajo == alto or k == 'inversion_inicial':
            continue
        nb, na = (beneficio_neto({**BASE, k: v})['neto_mensual'] for v in (bajo, alto))
        filas.append({'clave': k, 'supuesto': desc, 'neto_bajo': nb, 'neto_alto': na, 'rango': abs(na - nb),
                      'etiqueta_bajo': es(fmt.format(bajo)), 'etiqueta_alto': es(fmt.format(alto))})
    return sorted(filas, key=lambda f: -f['rango'])


if __name__ == '__main__':
    r = beneficio_neto(BASE)
    print('Tasas del modelo (test):', {k: round(v, 4) for k, v in TASAS.items()})
    for k, v in r.items():
        print(f'{k:22s} {v:,.1f}')
    print('\nSensibilidad (beneficio neto mensual, millones):')
    for f in sensibilidad():
        print(f"  {f['supuesto'][:60]:60s} {f['etiqueta_bajo']:>10s} → {f['neto_bajo'] / 1e6:7.1f} | {f['etiqueta_alto']:>10s} → {f['neto_alto'] / 1e6:7.1f}")
