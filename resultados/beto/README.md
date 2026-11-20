# Resultados de BETO

Esta carpeta recibe los archivos que genera `notebooks/02_beto_finetuning_colab.ipynb` al ejecutarse en Google Colab con GPU:

| Archivo | Contenido |
|---|---|
| `beto_resumen.json` | Configuración elegida en validación, tiempos, latencia en CPU, tamaño, F1 por semilla y robustez |
| `beto_probabilidades.csv` | Probabilidades de cada clase para validación, test y la prueba externa (alineadas por `id` con el corpus) |
| `beto_contrafactuales.csv` | Predicciones de BETO en los mismos casos escritos a mano del notebook principal |

Para subirlos desde GitHub: abra esta carpeta → *Add file* → *Upload files*. Después vuelva a ejecutar `notebooks/01_proyecto_final_bodytech.ipynb`: su sección 15 agrega BETO al tablero, a los intervalos de confianza, a los contrafactuales y al criterio de adopción.

Los pesos ajustados (unos 440 MB, en `modelos/beto/`) no se suben: superan el límite de GitHub y no hacen falta para reproducir las cifras.
