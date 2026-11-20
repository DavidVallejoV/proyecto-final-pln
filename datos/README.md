# Data card · Corpus de reseñas por sede (proxy del caso Bodytech)

| | |
|---|---|
| **Archivo que se entrega** | `datos/procesado/corpus_bodytech_anonimizado.csv` (13.331 filas, 13,8 MB, UTF-8) |
| **Diccionario** | `datos/procesado/diccionario_datos.csv` (resumido abajo) |
| **Tipo de datos** | **Reales y públicos, anonimizados.** No hay datos sintéticos ni datos de clientes de Bodytech: el caso Bodytech es **simulado**. |
| **Tarea** | Clasificar el sentimiento de una reseña en 3 clases: negativa, neutra o positiva. Clase crítica: **negativa** |
| **Versión** | 1.0 · noviembre de 2026 · generado por `notebooks/01_proyecto_final_bodytech.ipynb` (secciones 3 a 6) |
| **Responsable** | David Vallejo Villegas (Ciencia de Datos) |

## 1. Origen, periodo y licencia
- **Fuente:** `Big_AHR.csv`, conjunto *Andalusian Hotels' Reviews* publicado en Kaggle (usuario *chizhikchi*): reseñas de TripAdvisor, en español, de hoteles de Andalucía. Copia original sin cambios en `datos/raw/Big_AHR.csv` (MD5 `9fbdcce75618ac267d517aa38b79b126`).
- **Dos fuentes dentro del archivo.** Las últimas 1.816 filas no tienen hotel ni ubicación y se comportan distinto (el doble de negativas, reseñas casi el doble de largas, años mencionados 2006-2013). Coinciden con el corpus **COAH** del grupo SINAI de la Universidad de Jaén (Molina-González et al., 2014). En el corpus se marcan como `fuente = B` y se usan solo como **prueba externa**.
- **Periodo:** no hay fecha por reseña. Por los años que mencionan los textos: fuente A, 2009-2021 (más frecuente 2019); fuente B, 2006-2013.
- **Licencia y permisos:** datos públicos; según la ficha de Kaggle, licencia CC BY-NC 4.0 (uso no comercial, con atribución). **Verificar en la ficha antes de cualquier uso fuera del ámbito académico.** Ver `DATA_LICENSE.md`.

## 2. Población, cobertura y uso como *proxy*
- Huéspedes que escribieron en TripAdvisor sobre **715 hoteles** de 8 provincias de Andalucía (y 20 reseñas de Faro, Portugal), con un máximo de 20 reseñas únicas por hotel.
- **Proxy del caso Bodytech:** el hotel hace de **sede** y la provincia de **ciudad**. No existe un corpus público de reseñas de gimnasios en español; los hoteles comparten la estructura del problema (servicio presencial, personal, limpieza, instalaciones, precio).
- **Lo que no cubre:** vocabulario propio de un gimnasio (máquinas, clases, cupos, entrenadores), español de Colombia, canales como la app o las encuestas de Bodytech, y autores que son afiliados con contrato (no turistas).

## 3. Esquema de etiquetas
| Estrellas | Etiqueta (`sentimiento`) | % del archivo original |
|---|---|---|
| 1 y 2 | negativa | 14,7 % |
| 3 | neutra | 12,5 % |
| 4 y 5 | positiva | 72,8 % |

La etiqueta es la calificación que puso el autor, no una lectura del texto: una reseña de 3★ puede tener un texto claramente positivo o negativo. Es la principal fuente de ruido (ver sección 12 del notebook).

## 4. Procesamiento aplicado (en este orden)
1. **Duplicados:** se quitan 4.841 copias exactas (26,6 % del archivo); ninguna tenía etiquetas en conflicto. Quedan 13.331 reseñas únicas.
2. **Idioma** (`langdetect`, semilla fija): 130 reseñas no están en español → `parte = otro_idioma`, fuera del entrenamiento.
3. **Limpieza** (`src/texto.py::normalizar`): normalización NFC, enlaces, correos y teléfonos → `[URL]`, `[EMAIL]`, `[TEL]`; se quita la marca de corte «…». Se conservan tildes, mayúsculas, signos y emojis.
4. **Sede:** si la reseña menciona el nombre completo de su sede, se reemplaza por `[SEDE]` (349 reseñas). La sede se identifica con un código seudónimo (`S001`…).
5. **Partición por sede** (ver sección 6 de esta ficha).
6. **Nombres de personas → `[NOMBRE]`** con spaCy `es_core_news_sm` y una regla de precisión (una palabra con mayúscula se enmascara solo si nunca aparece en minúscula en train). El léxico se aprende **solo con train**. Resultado: 1.631 reseñas (12,4 %) con al menos un nombre enmascarado.

## 5. Diccionario de datos
| Columna | Tipo | Descripción |
|---|---|---|
| `id` | entero | Fila en el archivo original `Big_AHR.csv` (trazabilidad) |
| `fuente` | texto | A = TripAdvisor con sede identificada · B = sin sede (COAH), prueba externa |
| `sede_id` | texto | Código seudónimo de la sede (`S001`…); `sin_sede` en la fuente B |
| `provincia` | texto | Provincia de la sede (en Bodytech: la ciudad) |
| `rating` | entero | Estrellas de 1 a 5 (no es entrada del modelo) |
| `sentimiento` | texto | negativa (1-2★) · neutra (3★) · positiva (4-5★) |
| `titulo`, `resena` | texto | Título y reseña, limpios y anonimizados |
| `texto` | texto | Entrada de los modelos: título + ". " + reseña |
| `idioma` | texto | Idioma detectado |
| `cortada_en_origen` | booleano | La reseña original terminaba en «…» (texto incompleto) |
| `n_nombres_enmascarados` | entero | Nombres reemplazados por `[NOMBRE]` |
| `parte` | texto | `train`, `val`, `test`, `externo` u `otro_idioma` |

Nulos: ninguno en ninguna columna.

## 6. Partición
| Parte | Reseñas | Sedes | % negativas | Uso |
|---|---|---|---|---|
| train | 6.873 | 421 | 13,4 | Entrenamiento (y léxico del anonimizador, vocabulario, Word2Vec) |
| val | 1.710 | 106 | 13,7 | Selección de modelos y umbrales |
| test | 2.827 | 176 | 13,4 | Evaluación final, se abre una sola vez |
| externo | 1.791 | — | 27,9 | Prueba externa (fuente B): otra época y otro recolector |
| otro_idioma | 130 | — | — | Prueba de entradas fuera de alcance |

Cada sede está entera en una sola parte (`GroupShuffleSplit`, semilla 42): ninguna sede de test aparece en train ni en validación.

## 7. Calidad (auditoría)
- Nulos: 0 en el corpus final. Duplicados: eliminados.
- El **12,5 %** de las reseñas de la fuente A está cortado en origen (marcado en `cortada_en_origen`).
- Longitud mediana: 61 palabras (negativas 98, positivas 54); percentil 99: 300 palabras.
- Datos de contacto en el texto original: 0 correos, 2 teléfonos y 1 enlace (enmascarados).
- El 10 % de las sedes con más quejas concentra el 32 % de las reseñas negativas.

## 8. Privacidad
- **Qué se protege:** nombres de empleados y huéspedes (`[NOMBRE]`), correos, teléfonos y enlaces. Los autores de las reseñas no están identificados en el archivo original.
- **Riesgo residual:** el anonimizador puede dejar pasar nombres que el modelo pequeño de spaCy etiqueta como lugar u organización (en las pruebas de la demo, «Laura» y «Carolina» no se enmascararon) o que están escritos en minúscula, y enmascara de más algunas palabras al comienzo de una frase («Rechazamos», «Desearía») y algunos lugares o marcas. No hay un conjunto de referencia etiquetado a mano para medir su recall. Control: revisión humana de una muestra antes de publicar una nueva versión del corpus y, en producción, un reconocedor más grande con una lista de nombres de pila.
- El código de sede es un seudónimo, no anonimización completa: el archivo original es público y permite volver al hotel. No es un dato personal.

## 9. Limitaciones conocidas
- Dominio hotelero, no de gimnasios; español de España, no de Colombia.
- Etiqueta derivada de las estrellas (ruido en 2★-4★).
- Máximo 20 reseñas por sede y sin fechas: no se puede validar con una partición temporal.
- Reseñas cortadas en origen (12,5 % de la fuente A).
- Solo 8 provincias; algunas con pocas negativas en test (Jaén 17, Huelva 26).

## 10. Usos no previstos (no usar para)
- Evaluar, sancionar o perfilar a empleados identificados en las reseñas.
- Tomar decisiones automáticas sobre clientes sin revisión humana.
- Intentar reidentificar a autores o personas mencionadas.
- Uso comercial (la licencia es no comercial).
- Presentar las cifras como opiniones reales de afiliados de Bodytech.

## 11. Cómo se regenera
Ejecutar `notebooks/01_proyecto_final_bodytech.ipynb` de inicio a fin: las secciones 3 a 6 parten de `datos/raw/Big_AHR.csv` y escriben el corpus, el diccionario y el léxico del anonimizador (`modelos/lexico_minusculas.txt`).

**Referencia:** Molina-González, M. D., Martínez-Cámara, E., Martín-Valdivia, M. T. y Ureña-López, L. A. (2014). Cross-domain sentiment analysis using Spanish opinionated words. En *Natural Language Processing and Information Systems (NLDB 2014)*, LNCS 8455, pp. 214-219. Springer. https://doi.org/10.1007/978-3-319-07983-7_28
