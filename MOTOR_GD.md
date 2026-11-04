# Motor del Gemelo Digital de Inventario — SOAZ

Documento técnico del motor (`modulo_simulacion/`). No es una lista de archivos:
es el registro de las decisiones de modelado y de por qué cada una es como es.
Lo que hay aquí es lo que hay que poder defender.

---

## 1. Qué se gemela

El activo es el **proceso de abastecimiento y almacenamiento** de la
Comercializadora SOAZ (Calle Progreso 760, Otuzco – La Libertad). Un gemelo
digital no requiere que el activo sea una máquina: puede ser un proceso,
siempre que tenga estado observable, dinámica propia y decisiones que lo
alteren. El inventario cumple las tres.

**Alcance:** los 24 SKU de clase A del análisis ABC — el 40% del catálogo que
concentra el 80% del movimiento. Un solo almacén. Horizonte de 90 días.

**Fuera de alcance, por escrito:** 3D, IoT, sensores, ruteo, multi-almacén,
precios dinámicos, personal. El alcance se congela en `parametros.py` y se cita
cuando aparezca la tentación de expandirlo — que aparece siempre.

---

## 2. Por qué es un gemelo y no un dashboard

Cuatro piezas, y la tercera es la que casi todos se saltan:

| Pieza | Dónde vive |
|---|---|
| 1. Un activo real que emite datos | `datos_historicos.py` (hoy sintético; mañana, la tabla del sistema web) |
| 2. Una réplica virtual del estado | `estimacion.construir_estado_gemelo()` |
| 3. **Un modelo que simula el futuro** | `motor_des.py` — eventos discretos con SimPy |
| 4. Sincronización y **lazo de retorno** | `escenarios.recomendaciones_reposicion()` |

El lazo de retorno es **asistido**: el gemelo deposita la orden sugerida con su
razón, y el encargado aprueba o rechaza. Con humano en el medio no es un lazo
totalmente automático, y así está declarado. La tasa de aceptación es la
métrica que dice si el gemelo sirve.

---

## 3. Por qué eventos discretos y no un bucle por día

Un bucle diario obliga a que todo ocurra en la misma grilla. El inventario no
funciona así: la demanda llega a lo largo del día, la orden se emite en el
instante en que la posición cruza el punto de reorden, y el camión llega cuando
llega. SimPy deja que el reloj salte de evento a evento.

**Cuatro procesos por SKU:** demanda, revisión, llegada, cierre de día
(vencimiento + traza).

**La variable que dispara la compra es la posición, no el stock:**

```
posicion = stock_a_mano + en_transito − pendientes_por_atender
```

Confundirlas produce sobrecompra —el sistema vuelve a pedir lo que ya viene en
camino— y es el error más difícil de ver en los resultados, porque el servicio
mejora y solo el costo delata el problema.

---

## 4. La fórmula que casi todos simplifican

```
SS = z · √( L·σ²_d  +  d²·σ²_L )
```

El segundo término —la variabilidad del proveedor— es el que se omite al asumir
tiempo de entrega constante. **Medido en este catálogo: omitirlo subestima el
stock de seguridad en un 52% en promedio.** Para un proveedor que sube a Otuzco
por carretera de sierra, la irregularidad pesa más que la demanda.

`politicas.stock_seguridad_simplificado()` existe solo para poder mostrar esa
diferencia en el informe.

---

## 5. El problema que decide la calidad del gemelo: demanda censurada

**Las ventas no son la demanda.** Los días en que el producto se agotó, el
cliente que no fue atendido no dejó registro. En el histórico de este proyecto,
**el 11.5% de la demanda nunca llegó al sistema**.

Ajustar sobre ventas crudas subestima la demanda, el gemelo calcula un stock de
seguridad bajo y termina recomendando comprar de menos — reproduciendo el
problema que vino a resolver. Es el error que más silenciosamente arruina un
proyecto de inventarios, porque el modelo se ve perfectamente sano.

Tres estimadores implementados y comparados contra la verdad conocida:

| Estimador | Sesgo medio | Peor SKU |
|---|---|---|
| Promedio de ventas (ingenuo) | **−7.2%** | **−22.2%** |
| Excluyendo días con ruptura | −0.9% | −3.2% |
| **EM sobre normal censurada** | **+0.2%** | −2.7% |

El paso E reemplaza cada día censurado por su esperanza condicional
`E[D | D ≥ s] = μ + σ·φ(z)/(1−Φ(z))`; el paso M reestima. Sin bibliotecas
externas y sin optimización iterativa frágil.

---

## 6. Validación de fidelidad — el diferencial del proyecto

Un gemelo cuyo parecido con el original nunca se midió es una animación.

### El protocolo
1. Se corta el histórico en D₀ = día 360. El gemelo estima sus parámetros
   **solo** con lo anterior.
2. Se le carga el stock real de D₀ y corre 90 días hacia adelante **con la
   política que SOAZ realmente aplicó**, no con la optimizada. Validar con la
   política mejorada no prueba fidelidad, prueba otra cosa.
3. Se compara contra el registro.
4. Los umbrales se fijaron **antes**. Fijarlos después convierte una validación
   en una justificación.

### El hallazgo metodológico más importante del proyecto

Los umbrales originales eran **20% de MAPE del stock diario** y **2 días de
diferencia en rupturas**. Ambos resultaron **imposibles de cumplir para
cualquier modelo, incluido el verdadero**.

La razón: el stock es un diente de sierra y el día exacto del salto depende del
tiempo de entrega, que es aleatorio. Al promediar réplicas los saltos se
desalinean y el promedio se suaviza; compararlo punto a punto contra un diente
real castiga un desfase que ningún modelo puede eliminar.

**La prueba:** se corre el gemelo contra **sí mismo** con otra semilla.

| Métrica | Piso irreducible | Umbral original |
|---|---|---|
| MAPE del stock diario | **59%** | 20% ❌ imposible |
| Δ días de ruptura | **4.5 días** | 2 días ❌ imposible |
| Cobertura de banda 95% | **~82%** (no 95%, por autocorrelación) | 95% ❌ imposible |

Un umbral que ni el modelo verdadero puede alcanzar no es exigente: está mal
formulado. Por eso las métricas que comparan trayectorias estocásticas se
juzgan **relativas a su piso**, y `verificar_motor.py` incluye una prueba
(`t_umbrales_alcanzables`) que impide volver a fijar umbrales imposibles.

### Resultado

```
[CUMPLE] Nivel medio de stock                 13.7 %   (umbral ≤ 20%)
[CUMPLE] Cobertura, relativa al techo          0.96 ×  (umbral ≥ 0.90)
[CUMPLE] MAPE relativo al piso                 0.82 ×  (umbral ≤ 1.20)
[CUMPLE] Días de ruptura, relativo al piso     0.98 ×  (umbral ≤ 1.20)
[CUMPLE] Órdenes emitidas                      8.5 %   (umbral ≤ 15%)
VEREDICTO: PASA (5/5)
```

### Los tres huecos que hubo que cerrar para llegar ahí

La retro-prueba falló al principio, y el diagnóstico fue tan valioso como el
resultado:

1. **Estacionalidad.** El motor usaba media constante y el negocio es
   estacional. Se ajusta un armónico anual por mínimos cuadrados, y la
   simulación arranca en la fase del ciclo que corresponde al día de corte.
2. **Cumplimiento del proveedor.** El estado del gemelo no llevaba la
   probabilidad de que una orden llegue incompleta. Se estima conciliando
   órdenes contra llegadas.
3. **El decisor, no solo la física.** El dueño no compraba con la demanda
   corregida: compraba con su promedio de las últimas semanas, sesgado a la
   baja por la censura — y lo **reestimaba cada mes**. Congelar esa percepción
   hacía que el gemelo no siguiera la campaña estacional. Reproducir la
   fidelidad de un gemelo exige modelar **cómo aprende quien decide**.

---

## 7. Interpretabilidad y explicabilidad en cada figura y tabla

Ninguna salida se dibuja sola. Todo pasa por `narrativa.mostrar_figura()` o
`narrativa.mostrar_tabla()`, que adjuntan un bloque de tres partes **generado a
partir de los números de esa corrida** — no es texto fijo:

- **Qué muestra** — interpretabilidad: cómo se lee, en qué unidades, qué
  significan los ejes y los umbrales.
- **Por qué sale así** — explicabilidad: qué mecanismo del modelo produjo ese
  resultado y no otro.
- **Qué NO se puede concluir** — el límite honesto de la evidencia.

La tercera parte es la que suele faltar y la que más protege en una
sustentación: adelantarse a la objeción vale más que esperarla.

### Tres niveles de explicación

| Nivel | Pregunta | Método |
|---|---|---|
| **Local** | ¿Por qué *este* SKU cuesta lo que cuesta? | Valores de Shapley **exactos** |
| **Global** | ¿Qué factores mueven el costo del almacén? | Tornado / sensibilidad OAT |
| **Escenario** | ¿Por qué este what-if cambia el resultado? | Palanca a palanca |

### Por qué Shapley exacto y no SHAP

SHAP *aproxima* los valores de Shapley porque los modelos que explica son cajas
negras costosas de evaluar. Aquí el costo total tiene forma cerrada en nueve
factores: evaluarlo 2⁹ = 512 veces cuesta microsegundos, así que los valores
salen **exactos, sin muestreo**.

Eso habilita algo que con SHAP no se puede hacer: **verificar el axioma de
eficiencia en pantalla**. Error medido: `5.46e-12` — redondeo de punto flotante
y nada más. También se verifica el axioma nulo: un factor en su valor de
referencia recibe contribución exactamente cero.

### La honestidad que esto exige

La función analítica **no es el simulador**: es un sustituto cerrado del
simulador. Explicar el sustituto y presentarlo como si se explicara el motor
sería un fraude silencioso. Por eso `fidelidad_sustituto()` lo mide, y ese
número acompaña a toda explicación.

La primera medición dio **33.6% de error y correlación 0.25** — el sustituto no
servía. La causa: no modelaba la **merma**, que es el 35% del costo total y
domina por completo en lácteos y perecibles. Con el término de merma añadido:
**18.1% de error, correlación 0.969**. Un sustituto que omite el costo dominante
produce explicaciones bien formadas y equivocadas.

---

## 8. Toda cifra viene con incertidumbre

El motor es estocástico: la demanda y el tiempo de entrega se sortean. Una sola
corrida es una muestra de tamaño 1. Cada escenario se ejecuta con N réplicas de
semilla distinta y se reporta media e intervalo de confianza.

De ahí sale la columna que decide la comparación entre escenarios: no la
diferencia de costo, sino **«¿Distinguible del Base?»** — si el intervalo del
escenario se separa del intervalo del Base. Sin esa columna, cualquier política
parece funcionar.

---

## 9. Lo que el motor NO afirma

Está escrito en la interfaz y debe estar en el informe:

1. **El histórico es sintético.** Los resultados describen el **modelo**, no a
   SOAZ. Comparar escenarios entre sí es legítimo; leer estos soles como el
   costo real del negocio no lo es.
2. **La penalidad de ruptura es el supuesto más delicado.** Nadie registra al
   cliente que se fue, y ese número mueve el resultado más que ningún otro.
   Debe acordarse con la representante del negocio (Sania Gómez Rojas).
3. **La retro-prueba valida bajo la política que ya se usaba.** No prueba que el
   gemelo acierte bajo una política nueva: extrapolar a un régimen no observado
   es un salto que la validación no cubre.
4. **`parametros.py` marca la procedencia de cada coeficiente** — TEORIA,
   ESTIMADO o SUPUESTO — para que «¿de dónde sale ese número?» tenga respuesta
   sin improvisar.

---

## 10. Archivos

| Archivo | Rol |
|---|---|
| `parametros.py` | Todo coeficiente, con unidad y **procedencia**. Ningún número mágico en la lógica. |
| `catalogo.py` | SKU y clasificación ABC. Separa parámetros **verdaderos** de **estimados**. |
| `datos_historicos.py` | Genera el histórico de movimientos **censurado**, como el real |
| `estimacion.py` | Demanda (EM sobre censura), lead time, estacionalidad, cumplimiento |
| `politicas.py` | SS/ROP/EOQ, políticas (s,Q), (R,S) y (R,s,Q), modelo analítico de costo |
| `motor_des.py` | Simulación de eventos discretos con SimPy |
| `escenarios.py` | Réplicas, IC, comparación, curva servicio-costo, recomendaciones |
| `validacion.py` | Retro-prueba de fidelidad y **pisos irreducibles** |
| `explicabilidad.py` | Shapley exacto, tornado, descomposición de escenario, fidelidad del sustituto |
| `narrativa.py` | Genera las interpretaciones y **obliga** a que toda figura las lleve |
| `modelo_gemelo.py` | Interfaz Streamlit en 10 pestañas |
| `crisp_dm.py` | Protocolo CRISP-DM del pronóstico sobre Favorita, tienda 44 |
| `vista_crisp.py` | Pestaña que muestra ese protocolo, con pie en cada figura y tabla |
| `verificar_motor.py` | 23 pruebas numéricas del comportamiento del modelo |
| `app_motor.py` | Arranque del motor, sin base de datos ni login |

El copiloto de escenarios vive aparte, en `modulo_copiloto/`, y es **opcional**: sin clave de API la pestaña se desactiva sola y el motor funciona igual.

| Archivo | Rol |
|---|---|
| `esquemas.py` | Las 8 palancas que el modelo de lenguaje puede pedir, tipadas y validadas. Toda su superficie de contacto con el motor cabe aquí. |
| `proveedor.py` | Acceso al LLM de redacción y la decisión de si el copiloto puede correr. |
| `langflow_cliente.py` | Llama al flujo de Langflow que traduce la pregunta a palancas. Si no hay flujo configurado, traducir sigue en el modelo directo. |
| `guardia.py` | Cerco numérico: verifica que toda cifra del texto generado aparezca entre las que se le entregaron al modelo |
| `grafo.py` | El grafo LangGraph: traducir → aprobar → simular → redactar → cercar, con ciclo de corrección en la traducción y en la redacción |

---

## 11. Cómo correrlo

```bash
pip install -r requirements.txt

# Verificar que el modelo se comporta como dice
python verificar_motor.py

# Levantar el motor
streamlit run app_motor.py
```

Cuando el sistema web de la tesis esté operativo, se cambia `FUENTE_DATOS` en
`config.py` de `"sintetico"` a `"postgres"` y se implementa la lectura de
`movimiento_inventario` en `datos_historicos.py`. El resto del motor no se toca:
esa fue la razón de aislar la fuente de datos detrás de una sola función.

---

## 12. Salida de la verificación

```
[PASA] El alcance ABC concentra el movimiento declarado
       24 SKU (40.0% del catálogo) concentran 79.6% del movimiento
[PASA] El histórico está censurado, como el real
       11.48% de la demanda nunca llegó al registro
[PASA] La corrección por censura reduce el sesgo de la demanda
       promedio de ventas: −7.17% (peor SKU −22.2%) → EM: +0.18%
[PASA] El stock de seguridad completo supera al simplificado
       omitir σ_L subestima el stock de seguridad en 52%
[PASA] Axioma de eficiencia de Shapley — peor error 5.46e-12 sobre 512 coaliciones
[PASA] Axioma nulo (dummy) de Shapley
[PASA] El sustituto analítico es fiel a la simulación — correlación 0.969
[PASA] Las políticas se ordenan como la teoría predice
[PASA] Reproducibilidad con la misma semilla
[PASA] Semillas distintas producen resultados distintos
[PASA] Cobertura del intervalo de confianza
[PASA] Los umbrales de fidelidad están por ENCIMA del piso irreducible
[PASA] El gemelo pasa la retro-prueba de fidelidad — 5/5 criterios
[PASA] La descomposición de escenario suma coherentemente
[PASA] El tornado ordena por efecto real
[PASA] Un escenario ad-hoc no cae en el Base en silencio
       ad-hoc S/. 21,008 ≠ Base S/. 19,150, y reproduce S1 exactamente
[PASA] El cerco numérico rechaza una cifra que el motor nunca produjo
       4 cifras permitidas; texto fiel aceptado y la invención señalada
[PASA] El grafo corrige un escenario inválido y nunca simula uno malo
       rechazó nivel_servicio=0.97 y esperó la confirmación del humano
[PASA] Sin clave de API el copiloto se apaga y el motor no se entera
[PASA] El copiloto no manda parámetros que el modelo ya rechaza
       payload limpio: sin temperature/top_p/top_k ni en extra_body
[PASA] Ante una pregunta que no es un escenario, el copiloto NO inventa uno
       rechaza sin simular y ofrece las 8 palancas reales en su lugar

  Las 21 pruebas pasaron.
```

---

## 13. Referencias del marco

- Grieves, M., & Vickers, J. (2017). *Digital Twin: Mitigating Unpredictable,
  Undesirable Emergent Behavior in Complex Systems.* Springer.
- Kritzinger, W., Karner, M., Traar, G., Henjes, J., & Sihn, W. (2018).
  *Digital Twin in manufacturing: A categorical literature review and
  classification.* IFAC-PapersOnLine, 51(11), 1016–1022.
- ISO 23247-1:2021 a 23247-4:2021. *Digital twin framework for manufacturing.*
- Silver, E. A., Pyke, D. F., & Thomas, D. J. *Inventory and Production
  Management in Supply Chains* — políticas (s,Q) y (R,S), stock de seguridad con
  tiempo de entrega estocástico, y la función de pérdida normal G(z).

La estructura del motor viene de ahí. Los coeficientes concretos no: están
marcados como SUPUESTO en `parametros.py` y deben calibrarse con el negocio.
