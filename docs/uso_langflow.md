# Uso de Langflow en el gemelo digital de inventario — Comercializadora SOAZ

Documento de referencia para redactar el informe o el artículo. Describe lo que el código hace hoy. Las cifras de simulación las produce el motor Python; Langflow no las calcula.

Universidad Nacional de Trujillo — Ingeniería de Sistemas. Proyecto: gemelo digital de inventario de la Comercializadora SOAZ (Otuzco, La Libertad).

---

## 1. Cómo usar este documento

Sirve como fuente para un apartado de arquitectura, de interfaz en lenguaje natural o de límites del modelo de lenguaje. Cada afirmación de las secciones 2 a 12 está anclada en un archivo del repositorio. La sección 13 separa lo que el informe puede sostener de lo que no debe atribuirle a Langflow.

No incluye claves, identificadores de sesión ni el contenido de `.env`.

---

## 2. Qué papel tiene Langflow

Langflow es el traductor de la pestaña **Copiloto**. Recibe una pregunta de abastecimiento en español y devuelve un JSON plano con las palancas del gemelo. El flujo se llama **Traductor de escenarios SOAZ**. El endpoint publicado es `traductor_escenarios_soaz`.

Ese JSON no es un resultado. Es una propuesta de escenario. Python la valida, una persona la aprueba y, solo entonces, el motor de simulación la corre.

Langflow corre en local, en `http://localhost:7860`. La aplicación lo llama por HTTP. No está embebido dentro de Streamlit y no sustituye a SimPy, a Prophet ni al protocolo CRISP-DM.

Si Langflow no está configurado, el copiloto sigue existiendo: el mismo nodo de traducción llama al modelo directo declarado en `COPILOTO_MODELO`. Langflow es un reemplazo de ese nodo, no un requisito del gemelo.

---

## 3. Qué problema resuelve

El encargado no escribe factores ni nombres de política. Escribe una pregunta del tipo «¿y si el proveedor se demora el triple y se vuelve impredecible?». El motor, en cambio, solo acepta un conjunto cerrado de palancas numéricas.

Langflow ocupa el hueco entre esas dos formas de hablar. Su salida permitida es una de dos:

- **Escenario.** La pregunta es un *what-if* que las palancas pueden expresar. El JSON nombra cuáles se mueven y cuánto.
- **Rechazo.** La pregunta es un saludo, pregunta por la herramienta, o pide algo que el motor no modela (precios, rutas, un proveedor nuevo, otro almacén, contratar personal). En ese caso el motivo explica por qué no hay escenario. Inventar un escenario de consuelo está prohibido.

El rechazo es una salida válida. Existe porque, cuando el único canal de salida era un escenario bien formado, una pregunta que no se podía simular terminaba convertida en un escenario que el usuario no había pedido. Si el sistema inventa la pregunta, también queda en duda la respuesta.

---

## 4. Qué Langflow no hace

| Tarea | Quién la hace |
|---|---|
| Traducir la pregunta a palancas | Langflow, si está configurado |
| Validar rangos, políticas y coherencia | `modulo_copiloto/esquemas.py` |
| Pedir la aprobación humana | El grafo, nodo `confirmar` |
| Simular demanda, reposición, llegadas y vencimiento | Motor de eventos discretos (SimPy) |
| Calcular costo, servicio e intervalos | El mismo motor, con réplicas |
| Redactar el párrafo final | Modelo directo de `proveedor.py` (no el flujo de Langflow) |
| Rechazar una cifra que la simulación no produjo | `modulo_copiloto/guardia.py` |
| Pronosticar la serie, comparar modelos, CRISP-DM | `modulo_simulacion/`, sin Langflow |

El modelo de lenguaje, esté en Langflow o en el nodo de redacción, no estima resultados ni opina sobre lo que «pasaría». Elige palancas, o escribe sobre números que el motor ya entregó.

---

## 5. Dónde aparece en la aplicación

La pestaña **Copiloto** de `modulo_simulacion/modelo_gemelo.py` construye el grafo con un traductor opcional:

- Si `LANGFLOW_URL`, `LANGFLOW_FLOW_ID` y `LANGFLOW_API_KEY` están definidas, la pantalla indica que la traducción la hace el flujo **Traductor de escenarios SOAZ**.
- La simulación, el cerco numérico y la redacción del párrafo final permanecen en este proceso.

El resto de pestañas (alcance, Prophet, CRISP-DM, corrida, escenarios, fidelidad, explicación por SKU, tornado y recomendaciones) no llama a Langflow. El copiloto es opcional: sin clave, sin red o con `COPILOTO_ACTIVO=0`, esas pestañas siguen funcionando.

---

## 6. Arquitectura

```text
Pregunta en español
        │
        ▼
Pestaña Copiloto (Streamlit)
        │
        ▼
Grafo LangGraph
        │
        ├─ traducir ──► Langflow local :7860
        │                 Chat Input → Agente → Chat Output
        │                 JSON plano de palancas
        │                      │
        │                      ▼
        │               esquemas.py valida
        │               (hasta 3 intentos si el campo es corregible)
        │
        ├─ confirmar ──► la persona aprueba o descarta las palancas
        │
        ├─ simular ──► motor SimPy: caso Base y caso propuesto
        │
        ├─ redactar ──► modelo directo (Anthropic u otro COPILOTO_MODELO)
        │
        └─ cercar ──► el texto solo puede citar cifras ya entregadas
```

El estado del inventario no viaja a Langflow ni al checkpoint del grafo. El grafo lleva la conversación. El mundo simulado entra por cierre (*closure*) en los nodos `simular` y siguientes.

Hay dos modelos de lenguaje y no hacen el mismo trabajo:

1. **Agente del flujo Langflow.** Modelo `anthropic/claude-sonnet-4.5` vía OpenRouter. Solo traduce. La clave de OpenRouter vive en las variables de Langflow, no en este repositorio.
2. **Modelo directo del copiloto.** Por defecto `anthropic:claude-sonnet-5`, configurable con `COPILOTO_MODELO`. Redacta el párrafo. También traduce si Langflow no está configurado.

El flujo «Tutor de idiomas» que puede existir en la misma instancia de Langflow no forma parte de la aplicación. Sirvió solo como plantilla para copiar los componentes de entrada, agente y salida al crear el traductor.

---

## 7. El flujo dentro de Langflow

Lo crea o actualiza `crear_flujo_traductor.py`. Es idempotente: si el flujo ya existe, lo actualiza.

Cadena:

1. **Chat Input.** Recibe el texto que arma `langflow_cliente.armar_entrada`.
2. **Agente.** Lleva la instrucción de traducción (`INSTRUCCION_TRADUCCION` en `grafo.py`) y el contrato de campos (`INSTRUCCION_CAMPOS`). La salida estructurada es una tabla plana. Langflow 1.12 no admite ahí un objeto anidado del tipo `EscenarioPropuesto`, así que cada palanca viaja como texto. Una cadena vacía significa «esta palanca no se pidió».
3. **Chat Output.** Devuelve ese JSON al cliente HTTP.

La aplicación llama a:

```text
POST {LANGFLOW_URL}/api/v1/run/{LANGFLOW_FLOW_ID}
```

Cuerpo: `input_type` y `output_type` en `chat`, más un `session_id` nuevo en cada intento (`soaz-` y un identificador único). El rechazo del validador ya va dentro del texto. Reutilizar la sesión de chat duplicaría la pregunta.

Tiempo de espera del cliente: 120 segundos. Si la respuesta llega comprimida con gzip, el cliente la descomprime antes de leer el JSON.

---

## 8. Contrato de salida

Campos que el agente debe devolver. Los define `ESQUEMA_LANGFLOW` en `modulo_copiloto/langflow_cliente.py`.

| Campo | Tipo en el flujo | Significado |
|---|---|---|
| `es_simulable` | bool | `true` solo si las palancas alcanzan para la pregunta |
| `motivo` | str | Si no es simulable, explicación en español. Si es simulable, vacío |
| `nombre` | str | Nombre corto del escenario, 3 a 80 caracteres |
| `pregunta` | str | Una oración, máximo 200 caracteres, con la pregunta del negocio |
| `mecanismo` | str | Dos oraciones: qué parte del modelo se activa y por qué. Sin cifras de resultados |
| Ocho palancas | str | Número en texto, o cadena vacía si la pregunta no las mueve |

Reglas que viajan en la instrucción:

- Los números van con punto decimal, sin unidades y sin texto alrededor: `1.30`, `0.99`, `7`.
- Lo que no se mueve se queda en cadena vacía. No se rellena con `0` ni con `1`, porque esos valores sí cambiarían el escenario.
- Una palanca de más responde otra pregunta. Solo se mueven las que la pregunta pide.
- `pregunta` es la pregunta del negocio. No se copia en ella el aviso del validador.

`interpretar_plano` convierte ese plano en un `Traduccion` de Pydantic:

- Si `es_simulable` es falso, devuelve `Traduccion` solo con `motivo`. El grafo se detiene y no reintenta: insistir empujaría al modelo a inventar un escenario.
- Si es verdadero, construye un `EscenarioPropuesto`. Las palancas vacías se omiten. El resto pasa por los validadores de la sección 10.

Recortes locales, sin volver a llamar a Langflow:

- `nombre` a 80 caracteres, `pregunta` a 200, `mecanismo` a 400, `motivo` a 600. El corte cae en el último punto o espacio que cabe.
- Si el texto de `pregunta` contiene «RECHAZADA» o «validador del motor», se muestra la pregunta original del usuario.

Esos recortes existen porque un `mecanismo` demasiado largo disparaba un reintento, el modelo copiaba el aviso del validador dentro de `pregunta`, y la pantalla le volvía a preguntar al usuario algo que ya había respondido. Un campo fuera de rango, una política inexistente o un escenario sin palancas sí vuelven a Langflow: eso sí es un error de traducción.

---

## 9. Las ocho palancas

Son las únicas que `motor_des.simular` lee. Están declaradas en `PALANCAS`, dentro de `modulo_copiloto/esquemas.py`. Un campo extra está prohibido (`extra="forbid"`). Si el modelo inventara, por ejemplo, un factor de merma, el motor lo ignoraría en silencio y el resultado parecería haberlo aplicado. Prohibirlo convierte ese silencio en un error antes de simular.

| Palanca | Qué multiplica o fija | Rango que acepta el esquema | Ejemplo de lectura |
|---|---|---|---|
| `factor_demanda` | Demanda media | 0.1 a 5 | 1.20 = la demanda sube 20 % |
| `factor_lead_time` | Tiempo de entrega | 0.1 a 10 | 3.0 = se demora el triple |
| `factor_desv_lead_time` | Variabilidad del tiempo de entrega | 0.1 a 10 | Se usa cuando el proveedor además se vuelve impredecible |
| `nivel_servicio` | Nivel de servicio objetivo | Solo 0.90, 0.95, 0.975 y 0.99 | 0.99 = 99 % |
| `tipo_politica` | Política de reposición | `(s,Q)`, `(R,S)` o `(R,s,Q)` | `(R,S)` = revisión periódica hasta un nivel objetivo |
| `periodo_revision_dias` | Cada cuántos días se revisa | Entero de 1 a 90 | 7 = una vez por semana |
| `fraccion_presupuesto` | Fracción financiable de lo que la reposición pide | Mayor que 0 y hasta 2 | 0.70 = alcanza para el 70 % |
| `presupuesto_mensual` | Tope de compra en soles | Mayor que 0 | Solo si la pregunta da un monto |

Reglas de traducción que están en la instrucción del sistema, no improvisadas en pantalla:

- «Sube o baja un X %» la demanda → `factor_demanda = 1 + X/100`.
- «Se demora el doble» → `factor_lead_time = 2`. Si además se vuelve impredecible, también se mueve `factor_desv_lead_time`.
- «Nivel de servicio del 99 %» → `nivel_servicio = 0.99`. Si el porcentaje pedido no está en la tabla, se elige el tabulado más cercano a la intención, y el validador rechaza cualquier otro número.
- «Comprar una vez por semana» → `tipo_politica = "(R,S)"` y `periodo_revision_dias = 7`.
- «Solo me alcanza para el 85 %» → `fraccion_presupuesto = 0.85`.

`nivel_servicio` solo acepta esos cuatro valores porque son los únicos con *z* tabulado. Un 0.97 no haría fallar el motor: caería al *z* que el SKU ya traía (el de 95 %) mientras el escenario diría que se pidió 97 %. Limitar el campo hace que lo pedido y lo simulado sean el mismo número.

---

## 10. Validaciones de coherencia

Además del rango, `EscenarioPropuesto` rechaza combinaciones que el motor ejecutaría de una forma distinta a la declarada:

- El escenario tiene que mover al menos una palanca. Sin eso sería idéntico al caso Base.
- `periodo_revision_dias` no se combina con `(s,Q)`, porque esa política revisa de forma continua.
- `fraccion_presupuesto` y `presupuesto_mensual` no van juntos. El motor daría precedencia al monto absoluto e ignoraría la fracción.

Si alguna de estas reglas falla, el grafo guarda el mensaje y vuelve a llamar al traductor. El tope es tres intentos (el primero más dos correcciones). Al tercero, el grafo se detiene y lo dice: la pregunta no se pudo expresar con las palancas disponibles. Un fallo de red, de autenticación o de cuota no se reintenta.

La persona no aprueba JSON. Aprueba una tabla con el nombre de la palanca, el valor y la clave del motor (`filas_confirmacion`). Si descarta, no hay simulación.

---

## 11. Recorrido de una pregunta

1. La persona escribe la pregunta, o elige uno de los ejemplos de la pestaña, y pulsa **Traducir a escenario**.
2. El grafo entra al nodo `traducir`. Con Langflow configurado, el cliente envía la pregunta al flujo.
3. `interpretar_plano` arma un `Traduccion`. Pydantic aplica las reglas de la sección 10.
4. Si el escenario es válido, el grafo se interrumpe en `confirmar`. Streamlit muestra nombre, pregunta, mecanismo y palancas.
5. Si la persona aprueba, `simular` corre el caso Base y el caso propuesto con las mismas réplicas y la misma ventana de días. La comparación marca si la diferencia es distinguible del ruido de las réplicas.
6. `redactar` pide un párrafo al modelo directo, sobre un contexto ya calculado.
7. `cercar` extrae los números del párrafo y los contrasta con los números que ese contexto contenía, más las palancas aprobadas. Una cifra que no está ahí rechaza el párrafo. Hay un solo reintento de redacción. Si el segundo también falla, se descarta el párrafo del modelo y queda la interpretación determinista del motor.

El estado del grafo vive en la sesión de Streamlit (`InMemorySaver`). Por eso la pausa de aprobación sobrevive a que Streamlit vuelva a ejecutar el script al pulsar un botón.

---

## 12. Ejemplo de traducción

Pregunta:

> ¿Y si el proveedor se demora el triple y se vuelve impredecible?

Lectura que pide la instrucción:

- `factor_lead_time = 3`
- `factor_desv_lead_time = 3`
- el resto de palancas vacías

En una llamada real al flujo, Langflow devolvió exactamente esos dos factores. El `mecanismo` ocupó 237 caracteres, dentro del máximo de 400, y bastó una sola llamada. La simulación posterior, si la persona aprueba, la hace el motor. Langflow no devolvió costo ni nivel de servicio resultante.

Otros ejemplos que la pestaña ofrece, y que siguen el mismo contrato:

- «¿Qué pasa si la demanda sube 30 % en campaña navideña?» → `factor_demanda = 1.30`.
- «¿Cuánto cuesta prometer 99 % de nivel de servicio?» → `nivel_servicio = 0.99`.
- «¿Y si solo puedo financiar el 70 % de lo que la reposición pide?» → `fraccion_presupuesto = 0.70`.

Esos tres están en la interfaz como puntos de partida, no como un menú cerrado. Las ocho palancas se combinan. Si la pregunta no cabe en ellas, la respuesta correcta es el motivo de rechazo.

---

## 13. Qué puede decir el informe, y qué no

Puede afirmar:

- Langflow integra el gemelo con una interfaz en lenguaje natural, acotada a un flujo local de traducción.
- La integración es del tipo opción A: Langflow traduce, Python valida, la persona aprueba, SimPy simula y un cerco numérico revisa el texto.
- La superficie por la que el lenguaje toca la simulación son ocho palancas tipadas. Nada fuera de ese esquema llega al motor.
- El humano permanece en el lazo: sin aprobación no hay corrida del escenario propuesto.
- El copiloto es opcional. El gemelo de inventario, el pronóstico y CRISP-DM no dependen de Langflow.
- Hay una prueba automática (`t_langflow_traduce_en_el_grafo` en `verificar_motor.py`) que inyecta un traductor falso, rechaza un `nivel_servicio` de 0.97, exige la corrección y comprueba que el modelo directo no es quien tradujo ni quien calculó.

No debe afirmar:

- Que Langflow pronostica la demanda, elige el modelo de CRISP-DM o calcula el costo de reposición.
- Que el párrafo final sale del flujo de Langflow. Sale del modelo directo y puede ser descartado por el cerco.
- Que cualquier pregunta en español se puede simular. El rechazo es parte del diseño.
- Que el flujo «Tutor de idiomas» participa en el gemelo.
- Cifras de ahorro, fill rate o costo atribuibles a Langflow. Esas cifras, cuando existen, salen de una corrida concreta del motor y hay que citar esa corrida.

---

## 14. Configuración

Variables en el entorno o en `.env`. `config.py` las lee sin pisar las que ya estén definidas en el proceso. Los nombres y el papel; no los valores secretos:

| Variable | Papel |
|---|---|
| `LANGFLOW_URL` | Base del servidor. Si falta, el cliente usa `http://localhost:7860` |
| `LANGFLOW_FLOW_ID` | Identificador del flujo **Traductor de escenarios SOAZ** |
| `LANGFLOW_API_KEY` | Clave con la que la app llama a la API de Langflow |
| `COPILOTO_MODELO` | Modelo directo. Redacta siempre. Traduce solo si Langflow no está |
| `COPILOTO_ACTIVO` | `0` apaga el copiloto entero. El motor no se entera |
| `ANTHROPIC_API_KEY` | Clave del modelo directo cuando el proveedor es Anthropic |

Para crear o actualizar el flujo, con Langflow encendido:

```powershell
.\venv\Scripts\python.exe crear_flujo_traductor.py
```

El script escribe en `.env` la URL, el identificador del flujo y la clave de la API de Langflow. Ese archivo no se versiona.

---

## 15. Archivos

| Archivo | Papel respecto de Langflow |
|---|---|
| `crear_flujo_traductor.py` | Crea o actualiza el flujo en el Langflow local |
| `config.py` | Lee `LANGFLOW_URL`, `LANGFLOW_FLOW_ID` y `LANGFLOW_API_KEY` |
| `modulo_copiloto/langflow_cliente.py` | Cliente HTTP, esquema plano y conversión a `Traduccion` |
| `modulo_copiloto/grafo.py` | Nodos traducir → confirmar → simular → redactar → cercar. Langflow solo entra en traducir |
| `modulo_copiloto/esquemas.py` | Las ocho palancas, los rangos y las reglas de coherencia |
| `modulo_copiloto/guardia.py` | Cerco numérico del párrafo redactado |
| `modulo_copiloto/proveedor.py` | Modelo directo y decisión de si el copiloto puede arrancar |
| `modulo_simulacion/modelo_gemelo.py` | Pestaña Copiloto: inyecta el traductor si está configurado |
| `verificar_motor.py` | Prueba del ciclo de corrección con un traductor inyectado, sin red |

---

## 16. Formulación breve para el artículo

Langflow se usa como traductor de escenarios del copiloto del gemelo. Un flujo local convierte la pregunta de abastecimiento en un JSON de palancas de inventario. El motor Python valida ese JSON, la persona lo aprueba y la simulación de eventos discretos produce las cifras. Un segundo modelo redacta el párrafo sobre esos resultados, y un cerco numérico lo rechaza si cita un número que la corrida no entregó. El pronóstico, la comparación de modelos y el resto de la interfaz no pasan por Langflow.
