# Licencia y origen de los datos

## Fuente
- **Dataset:** *Andalusian Hotels' Reviews* (archivo `Big_AHR.csv`), publicado en Kaggle por el usuario `chizhikchi`:
  https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced
- **Contenido:** 18.172 filas de reseñas en español, publicadas en TripAdvisor, sobre hoteles de Andalucía (España), con título,
  texto, calificación de 1 a 5 estrellas, hotel y ubicación.
- **Subconjunto COAH:** las 1.816 filas sin hotel ni ubicación coinciden en número con el corpus COAH del grupo SINAI de la
  Universidad de Jaén (https://sinai.ujaen.es/en/research/resources/coah):
  Molina-González, M. D., Martínez-Cámara, E., Martín-Valdivia, M. T. y Ureña-López, L. A. (2014). *Cross-domain sentiment
  analysis using Spanish opinionated words*. NLDB 2014, LNCS 8455, pp. 214-219. Springer. https://doi.org/10.1007/978-3-319-07983-7_28

## Licencia
- La ficha del dataset en Kaggle lo publica con licencia **CC BY-NC 4.0** (atribución, uso no comercial).
  **Antes de cualquier uso distinto del académico, verifique la licencia vigente en la página de Kaggle.**
- Este repositorio usa los datos con fines **académicos y no comerciales**, con la atribución de esta página.
- El código del repositorio tiene licencia MIT (ver `LICENSE`); esa licencia **no** se extiende a los datos.

## Qué datos contiene el repositorio
| Archivo | Tipo de dato | Qué contiene |
|---|---|---|
| `datos/raw/Big_AHR.csv` | Real, público | Copia sin modificar del archivo de Kaggle, para que el notebook pueda reproducirse de inicio a fin |
| `datos/procesado/corpus_bodytech_anonimizado.csv` | Real, anonimizado | Reseñas sin duplicados, con nombres de personas (los que detecta el anonimizador; ver el riesgo residual en `datos/README.md`), sedes, correos, teléfonos y enlaces enmascarados; la sede como código seudónimo y la ubicación como provincia |
| `datos/procesado/diccionario_datos.csv` | Documentación | Diccionario de columnas del corpus anonimizado |

## Uso simulado
El caso de negocio (Bodytech) es **simulado**: las reseñas no son de usuarios de Bodytech. El hotel hace las veces de sede y la
provincia de ciudad. Ningún resultado de este repositorio describe el desempeño del modelo con reseñas reales de gimnasios.
