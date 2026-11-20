"""Figuras del informe y de la presentación, calculadas a partir de resultados/ (no se escriben cifras a mano).

Ejecutar desde la raíz del repositorio, después del notebook principal:  python docs/figuras_documentos.py
Genera en docs/figuras/: politica_1000.png, diferencia_f1.png y sensibilidad.png.
"""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'docs'))
from caso_de_valor import BASE as SUPUESTOS, beneficio_neto, sensibilidad  # noqa: E402

SALIDA = RAIZ / 'docs' / 'figuras'
SALIDA.mkdir(parents=True, exist_ok=True)
R = json.loads((RAIZ / 'resultados' / 'resumen_resultados.json').read_text(encoding='utf-8'))
TAB = pd.read_csv(RAIZ / 'resultados' / 'tablero_test.csv', index_col=0).to_dict(orient='index')   # precisión completa

# Paleta de referencia (modo claro) y tinta
TINTA, TINTA2, TENUE, REJILLA, EJE, FONDO = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
AZUL, NARANJA = '#2a78d6', '#eb6834'
BIEN, ATENCION, CRITICO = '#0ca30c', '#fab219', '#d03b3b'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': TINTA, 'axes.edgecolor': EJE,
                     'axes.labelcolor': TINTA2, 'xtick.color': TINTA2, 'ytick.color': TINTA2, 'figure.facecolor': FONDO,
                     'axes.facecolor': FONDO, 'savefig.facecolor': FONDO})


def coma(x, d=1, signo=False):
    """Redondeo «hacia arriba en el medio» (como se lee en el notebook) y coma decimal."""
    v = float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
    return (f'{v:+.{d}f}' if signo else f'{v:.{d}f}').replace('.', ',').replace('-', '−')


def limpiar_ejes(ax):
    for lado in ['top', 'right', 'left']:
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0)


# 1. Qué pasa con las reseñas de sedes nuevas (política del baseline en test)
fila = next(f for f in R['politica_tabla'] if f['modelo'] == 'tfidf_lr (baseline)' and f['parte'] == 'test')
todas = [fila['automáticas por 1.000'] / 10, fila['a revisión por 1.000'] / 10, fila['alertas por 1.000'] / 10]
neg_alerta = fila['negativas con alerta'] * 100
neg_persona = fila['negativas que ve una persona (alerta o revisión)'] * 100
negativas = [100 - neg_persona, neg_persona - neg_alerta, neg_alerta]
fig, ax = plt.subplots(figsize=(8.6, 2.8))
colores, nombres = [BIEN, ATENCION, CRITICO], ['Registro automático', 'Revisión humana', 'Alerta al coordinador']
for y, valores in [(1, todas), (0, negativas)]:
    izq = 0
    for v, col in zip(valores, colores):
        ax.barh(y, v, left=izq, color=col, height=0.56, edgecolor=FONDO, linewidth=2)
        if v >= 6:
            ax.text(izq + v / 2, y, f'{coma(v, 0)} %', ha='center', va='center',
                    color=TINTA if col == ATENCION else 'white', fontsize=10, fontweight='bold')
        izq += v
ax.annotate(f'{coma(negativas[0])} % de las quejas se registra sin que nadie la lea', xy=(0.3, -0.29), xytext=(1.5, -0.62),
            fontsize=8.5, color=TINTA2, va='center', arrowprops=dict(arrowstyle='-', color=TINTA2, lw=0.8))
ax.set_yticks([1, 0]); ax.set_yticklabels(['Todas las reseñas', 'Solo las quejas reales'], fontsize=10, color=TINTA)
ax.set_xlim(0, 100); ax.set_ylim(-0.85, 1.4); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0 %', '25 %', '50 %', '75 %', '100 %'])
ax.xaxis.grid(True, color=REJILLA, lw=0.8); ax.set_axisbelow(True); limpiar_ejes(ax)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col) for col in colores], labels=nombres, ncol=3, frameon=False,
          loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=9)
plt.tight_layout(); plt.savefig(SALIDA / 'politica_1000.png', dpi=200, bbox_inches='tight'); plt.close()

# 2. Diferencia de F1 macro en test frente al baseline, con intervalo de confianza del 95 % (bootstrap pareado)
ic = R['ic']
base = TAB['tfidf_lr (baseline)']['F1 macro test']
filas = [('BiLSTM + Word2Vec\n(semilla desplegada)', TAB[R['avanzado']]['F1 macro test'] - base, ic['diferencia'])]
if 'beto' in ic and 'beto (notebook 02)' in TAB:
    filas.append(('BETO\n(notebook 02)', TAB['beto (notebook 02)']['F1 macro test'] - base, ic['beto']))
fig, ax = plt.subplots(figsize=(8.6, 1.3 + 0.75 * len(filas)))
for i, (nombre, d, (a, b)) in enumerate(filas[::-1]):
    ax.plot([a, b], [i, i], color=AZUL, lw=2, solid_capstyle='round')
    ax.plot(d, i, 'o', ms=9, color=AZUL, mec=FONDO, mew=2)
    ax.text(b + 0.003, i, f'{coma(d, 3, True)}  (IC 95 %: {coma(a, 3, True)} a {coma(b, 3, True)})', va='center', fontsize=9, color=TINTA2)
ax.axvline(0, color=TINTA2, lw=1); ax.axvline(0.02, color=BIEN, lw=1.2, ls='--')
arriba = len(filas) - 0.3
ax.text(0.001, arriba, 'baseline', ha='left', fontsize=8.5, color=TINTA2)
ax.text(0.021, arriba, 'meta: +0,02', ha='left', fontsize=8.5, color='#006300')
ax.set_yticks(range(len(filas))); ax.set_yticklabels([f[0] for f in filas[::-1]], color=TINTA)
ax.set_ylim(-0.6, len(filas) - 0.05)
ax.set_xlim(min(-0.04, min(f[2][0] for f in filas) - 0.01), max(0.09, max(f[2][1] for f in filas) + 0.075))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: coma(v, 2, v != 0)))
ax.set_xlabel('Diferencia de F1 macro en test frente al baseline TF-IDF')
ax.xaxis.grid(True, color=REJILLA, lw=0.8); ax.set_axisbelow(True); limpiar_ejes(ax)
plt.tight_layout(); plt.savefig(SALIDA / 'diferencia_f1.png', dpi=200, bbox_inches='tight'); plt.close()

# 3. Sensibilidad del caso de valor: beneficio neto mensual al mover un supuesto a la vez
base_neto = beneficio_neto(SUPUESTOS)['neto_mensual'] / 1e6
sens = sensibilidad()[:8]
fig, ax = plt.subplots(figsize=(8.8, 4.4))
for i, s in enumerate(sens[::-1]):
    for valor, etiqueta, col in [(s['neto_bajo'] / 1e6, s['etiqueta_bajo'], NARANJA), (s['neto_alto'] / 1e6, s['etiqueta_alto'], AZUL)]:
        ax.barh(i, valor - base_neto, left=base_neto, color=col, height=0.6, edgecolor=FONDO, linewidth=2)
        derecha = valor >= base_neto
        if not derecha and valor < 9 and base_neto - valor > 10:      # cerca de la línea de equilibrio: la etiqueta va dentro de la barra
            ax.text(valor + 1.2, i, etiqueta, va='center', ha='left', fontsize=8, color='white', fontweight='bold')
        else:
            ax.text(valor + (1.2 if derecha else -1.2), i, etiqueta, va='center', ha='left' if derecha else 'right', fontsize=8, color=TINTA2)
ax.plot([base_neto, base_neto], [-0.6, len(sens) - 0.4], color=TINTA, lw=1.2)
ax.plot([0, 0], [-0.6, len(sens) - 0.4], color=CRITICO, lw=1, ls='--')
ax.text(base_neto, len(sens) - 0.25, f'escenario base: {coma(base_neto, 1)} M', ha='center', fontsize=8.5, color=TINTA)
ax.text(0, len(sens) - 0.25, 'equilibrio (0)', ha='center', fontsize=8.5, color=CRITICO)
ax.set_yticks(range(len(sens))); ax.set_yticklabels([s['supuesto'] for s in sens[::-1]], fontsize=9, color=TINTA)
ax.set_ylim(-0.7, len(sens) + 0.15)
ax.set_xlabel('Beneficio neto mensual (millones de pesos)')
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=NARANJA), plt.Rectangle((0, 0), 1, 1, color=AZUL)],
          labels=['supuesto en su valor bajo', 'supuesto en su valor alto'], frameon=False, ncol=2, loc='lower left',
          bbox_to_anchor=(0, 1.0), fontsize=9)
ax.xaxis.grid(True, color=REJILLA, lw=0.8); ax.set_axisbelow(True); limpiar_ejes(ax)
ax.set_xlim(min(-12, min(s['neto_bajo'] for s in sens) / 1e6 - 14), max(s['neto_alto'] for s in sens) / 1e6 + 16)
plt.tight_layout(); plt.savefig(SALIDA / 'sensibilidad.png', dpi=200, bbox_inches='tight'); plt.close()
print('Figuras en', SALIDA.relative_to(RAIZ), [p.name for p in sorted(SALIDA.glob('*.png'))])
