# Model card · Clasificador de reseñas por sede (caso Bodytech)

| | |
|---|---|
| **Modelo desplegado en la demo** | Baseline TF-IDF (palabras y pares de palabras) + regresión logística balanceada · `modelos/baseline_tfidf_lr.joblib` |
| **Versión** | huella SHA-256 `a35d91355926` (la demo la muestra y la registra en cada clasificación) |
| **Política de decisión** | `config/politica.json`: umbral de confianza 0,50 · alerta si P(negativa) ≥ τ = 0,323 · revisión si hay más de 30 % de palabras desconocidas, otro idioma, texto vacío o predicción neutra |
| **Alternativas evaluadas** | BiLSTM con embedding al azar y con Word2Vec (`modelos/bilstm/`); BETO (notebook 02): el más preciso, pero no cumple la latencia en CPU y falla con el texto en mayúsculas; corre en modo sombra durante el piloto |
| **Dueño del modelo** | Fabián David Sánchez (Machine Learning) · dueño del dato: David Vallejo Villegas (Ciencia de Datos) · dueño del proceso y de la decisión de suspender: Jesús Joseph Suescun (Gerencia de IA) |
| **Fecha** | noviembre de 2026 |

## Uso previsto
- **Usuarios:** coordinadores de sede y equipo de servicio al cliente.
- **Tarea:** clasificar cada reseña en negativa, neutra o positiva y decidir si va como **alerta al coordinador**, a **revisión humana** o a **registro automático**.
- **Entrada:** título y texto de una reseña en español. **Salida:** clase, probabilidades, confianza, decisión, motivo y los términos que más pesaron (evidencia).

## Usos fuera de alcance
- Evaluar o sancionar a empleados mencionados en las reseñas.
- Responder automáticamente a clientes o cerrar quejas sin que una persona las lea.
- Textos en otros idiomas, textos que no son reseñas, o reseñas de sectores distintos sin reentrenar y volver a validar.
- Uso con reseñas reales de Bodytech sin reentrenar con datos propios: este modelo se entrenó con reseñas de hoteles (*proxy*).

## Datos
Corpus anonimizado de 13.331 reseñas únicas de hoteles de Andalucía (ver `datos/README.md`). Entrenamiento con train (6.873 reseñas de 421 sedes); umbrales con validación (1.710 de 106 sedes); evaluación con test (2.827 de 176 sedes nunca vistas) y con una prueba externa de otra fuente y otra época (1.791).

## Desempeño (test, sedes nunca vistas)
| Métrica | Baseline (desplegado) | BiLSTM + Word2Vec | BETO |
|---|---|---|---|
| F1 macro (IC 95 %) | 0,723 (0,703 a 0,743) | 0,727 (0,708 a 0,747) | 0,786 (diferencia +0,064; IC +0,042 a +0,085) |
| Recall de negativas (clase más probable) | 0,818 | 0,853 | 0,834 |
| F1 de la clase neutra | 0,472 | 0,480 | 0,568 |
| F1 macro en la prueba externa | 0,744 | 0,746 | 0,818 |
| Error de calibración (ECE) | 0,119 | 0,044 | 0,075 |
| Latencia p95 por reseña (CPU) | 1,2 ms | 14,3 ms | 380 ms |
| Tamaño en disco · memoria del proceso | 2,5 MB · 160 MB | 8,6 MB · 662 MB | 440 MB · no medida |

**Con la política de decisión (baseline, test):** de cada 1.000 reseñas, 646 se registran solas (aciertan el 96 %), 177 van a revisión y 178 generan alerta. Las alertas cubren el 87,9 % de las quejas reales y una persona ve el 99,5 %.

**Por segmento:** el F1 macro por provincia va de 0,661 a 0,755; las cifras de provincias con pocas quejas son inestables. Las reseñas de 101 a 160 palabras son las más difíciles.

## Limitaciones conocidas
- La etiqueta viene de las estrellas: el 84 % de los errores son reseñas de 2★ a 4★, muchas con opiniones mixtas.
- No entiende el orden de las palabras («No es malo, es bueno») ni la ironía.
- Asigna confianza alta a textos en otro idioma: por eso existe la regla de palabras desconocidas, que detiene el 99 % de esos textos.
- No detecta textos fuera de tema («La fotosíntesis depende de la luz» genera una alerta falsa).
- El anonimizador depende del reconocedor de entidades de spaCy, que a veces etiqueta un nombre como lugar u organización («Laura», «Carolina») y no lo enmascara.
- El umbral τ se eligió en validación para alertar el 90 % de las quejas; en sedes nuevas alertó el 87,9 %. Debe recalcularse con datos reales durante el piloto.

## Consideraciones éticas y de privacidad
- Los nombres de personas, correos, teléfonos y enlaces se enmascaran antes de entrenar y antes de clasificar.
- La demo registra cada decisión con la huella de la entrada, nunca con el texto (`resultados/registro_demo.csv`).
- Ninguna decisión sobre un cliente o un empleado se toma solo con el modelo: las alertas y las revisiones las atiende una persona.

## Mantenimiento
- **Monitoreo semanal sin etiquetas:** PSI de longitud y de confianza, % de palabras desconocidas, % de predichas negativas y % enviado a revisión o alerta. **Mensual con etiquetas:** 200 reseñas etiquetadas a mano para medir el F1 real y el recall de quejas.
- **Reentrenar** si el F1 real cae más de 0,03 o el recall de quejas con alerta baja de 0,85.
- **Suspender (volver al proceso manual)** si una muestra mensual muestra que más del 2 % de las quejas reales se registraron automáticamente, si se detecta un dato personal sin enmascarar en los registros o si la carga de revisión supera la capacidad del equipo dos semanas seguidas.
- **Volver a la versión anterior (rollback):** cada versión del modelo y de la política queda identificada por su huella; la anterior se conserva para restaurarla.
