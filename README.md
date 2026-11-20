# Clasificador de reseñas por sede · Caso Bodytech

**Proyecto final de Procesamiento de Lenguaje Natural** · Maestría en Ciencia de Datos, Universidad Ean · Docente: Gerson José Moreno Morales

**Equipo:** Jesús Joseph Suescun (Gerencia en IA) · Fabián David Sánchez (Machine Learning) · David Vallejo Villegas (Ciencia de Datos)

## Problema
Bodytech recibe reseñas de sus usuarios sobre cada sede (app, Google y encuestas). Si una reseña negativa no llega a tiempo al
coordinador de la sede, nadie contacta al usuario y aumenta el riesgo de que cancele su plan. El sistema clasifica cada reseña como
**negativa, neutra o positiva**, envía una **alerta al coordinador** cuando hay una posible queja y manda a **revisión humana** lo que
no puede decidir con confianza.

> **Datos:** no existe un corpus público de reseñas de gimnasios en español. Se usan 18.172 reseñas reales de hoteles de Andalucía
> (TripAdvisor) como proxy: el hotel hace de sede y la provincia de ciudad. El caso Bodytech es **simulado** (ver `DATA_LICENSE.md`).

<!-- RESULTADOS -->
## Resultados (test: 2.827 reseñas de 176 sedes que el modelo nunca vio)
| Modelo | F1 macro | Recall de quejas | ECE | F1 prueba externa | Latencia p95 (ms) | Memoria (MB) |
|---|---|---|---|---|---|---|
| TF-IDF + regresión logística (baseline, desplegado) | 0,723 | 0,818 | 0,119 | 0,744 | 0,9 | 160 |
| BiLSTM, embedding al azar | 0,714 | 0,795 | 0,037 | 0,734 | 13,0 | — |
| BiLSTM + Word2Vec | 0,727 | 0,853 | 0,044 | 0,746 | 10,7 | 662 |

> **BETO:** el notebook 02 está listo; sus cifras se agregan a esta tabla cuando se ejecute en Colab con GPU y sus archivos se suban a `resultados/beto/`.

- **Sin evidencia de que la BiLSTM supere al baseline:** +0,004 de F1 con intervalo de confianza del 95 % de −0,015 a +0,023 (la meta prerregistrada era +0,02), y 11 veces más latencia.
- **Política de decisión** (umbrales elegidos en validación): de cada 1.000 reseñas, 646 se registran solas (96 % de acierto), 177 van a revisión humana y 178 generan alerta. Una persona ve el **99,5 % de las quejas**; las alertas cubren el 87,9 % (meta 90 %, no se cumple en test).
- **Partición:** quitar 26,6 % de duplicados y partir por sede evita reportar un F1 de 0,802 que en sedes nuevas es de 0,72.
- **Caso de valor (supuestos explícitos):** beneficio neto de unos $27,7 millones al mes; basta retener 14 afiliados al mes en toda la red para cubrir la operación.
- **Recomendación:** pilotear el baseline con la política de decisión y revisión humana; no adoptar la BiLSTM.
<!-- /RESULTADOS -->

## Estructura del repositorio
```
├── notebooks/
│   ├── 01_proyecto_final_bodytech.ipynb   ← notebook principal: EDA, preparación, baseline, BiLSTM, evaluación, errores, conclusiones
│   └── 02_beto_finetuning_colab.ipynb      ← fine-tuning de BETO en Colab con GPU (misma partición)
├── src/                                     ← código reutilizable: limpieza y anonimización, partición, baseline, inferencia
├── app/demo.py                              ← demo en Streamlit: entrada → decisión, confianza, evidencia y manejo de errores
├── tests/test_proyecto.py                   ← pruebas automáticas (pytest)
├── datos/
│   ├── raw/Big_AHR.csv                      ← dataset original (público, Kaggle)
│   ├── procesado/corpus_bodytech_anonimizado.csv   ← corpus sin duplicados y anonimizado, con la partición por sede
│   ├── procesado/diccionario_datos.csv
│   └── README.md                            ← data card
├── modelos/                                 ← baseline (.joblib), BiLSTM (.keras), léxico del anonimizador
├── config/politica.json                     ← umbrales de decisión elegidos en validación
├── resultados/                              ← tablero, registro de experimentos, errores, figuras y resultados de BETO
├── docs/                                    ← informe, presentación, matriz de riesgos, caso de valor, model card, declaración de uso de IA
├── requirements.txt / requirements-beto.txt
├── .env.example                             ← parámetros configurables (sin secretos)
├── LICENSE                                  ← MIT (código)
└── DATA_LICENSE.md                          ← origen y licencia de los datos
```

## Cómo ejecutar
**Opción 1 · Google Colab (sin instalar nada).** Abra `notebooks/01_proyecto_final_bodytech.ipynb` en Colab
(*Archivo → Abrir cuaderno → GitHub* y pegue la URL de este repositorio) y use *Entorno de ejecución → Ejecutar todo*. La primera
celda clona el repositorio e instala lo que falta. Tarda unos 20 minutos en CPU (16 minutos en nuestro equipo de 2 núcleos).

**Opción 2 · Equipo local (Python 3.11 o superior).**
```bash
git clone https://github.com/DavidVallejoV/proyecto-final-pln
cd proyecto-final-pln
python -m venv .venv && source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/01_proyecto_final_bodytech.ipynb --output 01_ejecutado.ipynb
pytest -q                                                # pruebas
streamlit run app/demo.py                                # demo en el navegador
```

**BETO (notebook 02).** Ábralo en Colab con GPU T4 y ejecútelo completo (unos 20 minutos). Descarga `beto_resumen.json`,
`beto_probabilidades.csv` y `beto_contrafactuales.csv`; súbalos a `resultados/beto/` y vuelva a ejecutar el notebook 01, cuya
sección 15 los agrega a la comparación.

## Reproducibilidad
- Semilla 42 en todo el flujo; las redes se repiten con 5 semillas para medir la variación.
- El test se abre una sola vez, después de elegir modelos y umbrales con validación.
- Todo el preprocesamiento vive en `src/` y lo usan por igual el entrenamiento, la evaluación y la demo.
- Rutas relativas a la raíz del repositorio; los parámetros se cambian con variables de entorno (`.env.example`).
- El hardware, las versiones y los tiempos quedan en `resultados/entorno.json` y `resultados/registro_experimentos.csv`.

## Documentos
- Informe técnico: `docs/Informe_Tecnico_Proyecto_Final_PLN.docx`
- Presentación (18 diapositivas, con notas del orador): `docs/Presentacion_Proyecto_Final_PLN.pptx`
- Matriz de riesgos: `docs/Matriz_de_Riesgos.xlsx`
- Caso de valor con fórmulas y sensibilidad: `docs/Caso_de_Valor.xlsx` (misma lógica en `docs/caso_de_valor.py`)
- Figuras del informe, generadas desde `resultados/`: `docs/figuras_documentos.py`
- Model card: `docs/model_card.md` · Data card: `datos/README.md`
- Declaración de uso de IA: `docs/declaracion_uso_IA.md`

## Licencias
Código: MIT (`LICENSE`). Datos: ver `DATA_LICENSE.md` (CC BY-NC 4.0 según la ficha de Kaggle; uso académico).
