# 📦 Gemelo Digital de Inventario — Comercializadora SOAZ

Réplica ejecutable del proceso de abastecimiento de la Comercializadora SOAZ
(Otuzco – La Libertad), para **probar políticas de reposición antes de
aplicarlas al almacén real**.

Universidad Nacional de Trujillo — Ingeniería de Sistemas · 2026

---

## Qué hace

Un gemelo digital del inventario: mantiene una réplica del estado de cada SKU
sincronizada con los movimientos del negocio, **simula hacia adelante** con
eventos discretos, y devuelve órdenes de reposición al proceso real.

- **Simulación de eventos discretos** con SimPy — demanda, revisión, llegada
  del proveedor y vencimiento como procesos independientes
- **Corrección de demanda censurada** por Esperanza-Maximización: las ventas no
  son la demanda, y el sesgo medido es de −7% (hasta −22% en el peor SKU)
- **Validación de fidelidad** con retro-prueba contra el histórico no visto, con
  umbrales fijados a priori y benchmarkeados contra el **piso irreducible** del
  propio motor
- **Interpretabilidad y explicabilidad en cada figura y tabla**, generadas de
  los números de la corrida — nunca texto fijo
- **Valores de Shapley exactos** (512 coaliciones, sin muestreo) con
  verificación del axioma de eficiencia en pantalla
- **Réplicas con intervalos de confianza** en toda métrica, y la columna
  «¿Distinguible del Base?» que dice si una diferencia sobrevive al ruido
- **Copiloto de escenarios**: se le pregunta en español, traduce a palancas del
  motor, **el humano aprueba** y recién entonces simula — y todo número del texto
  generado se verifica contra los resultados antes de mostrarse
- **23 pruebas numéricas** del comportamiento del modelo en `verificar_motor.py`

Ver **[`MOTOR_GD.md`](MOTOR_GD.md)** para el documento técnico completo: las
decisiones de modelado, los hallazgos y lo que el motor no afirma.

---

## Inicio rápido

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python verificar_motor.py       # 23 pruebas del modelo
streamlit run app_motor.py      # la interfaz
python -m modulo_simulacion.crisp_dm   # protocolo CRISP-DM (una vez; la app lee el caché)
```

No requiere PostgreSQL ni autenticación. La demanda observada sale del corte
cacheado de Favorita (tienda 44). El stock, el precio y el tiempo de entrega
no vienen en ese dataset: el motor los reconstruye con supuestos declarados.

---

## Ejecutable de Windows

Para llevar el gemelo a una máquina sin Python —la de la empresa, la de la
sustentación— hay un ejecutable:

```powershell
powershell -ExecutionPolicy Bypass -File construir_exe.ps1
```

Produce `dist\GemeloInventarioSOAZ\`. Se copia **la carpeta completa** a la
máquina destino y se abre `GemeloInventarioSOAZ.exe`: levanta el motor y abre
el navegador solo. Detalles, advertencias de Windows y diagnóstico en
**[`EJECUTABLE.md`](EJECUTABLE.md)**.

---

## Estructura

```
gemelo_inventario_soaz/
├── app_motor.py                 # Aplicación Streamlit
├── config.py                    # Configuración de entorno (no del modelo)
├── verificar_motor.py           # 23 pruebas numéricas del modelo
├── MOTOR_GD.md                  # Documento técnico del motor
│
├── lanzador.py                  # Entrada del .exe: levanta Streamlit y el navegador
├── crear_flujo_traductor.py     # Crea en Langflow el flujo que traduce preguntas
├── gemelo_soaz.spec             # Receta de PyInstaller
├── construir_exe.ps1            # Compilación del ejecutable, de un comando
├── EJECUTABLE.md                # Cómo se compila y se distribuye el .exe
│
├── modulo_simulacion/
│   ├── parametros.py            # Todo coeficiente, con su PROCEDENCIA
│   ├── catalogo.py              # SKU y clasificación ABC
│   ├── datos_historicos.py      # Histórico de movimientos (censurado)
│   ├── estimacion.py            # Demanda con corrección EM, lead time, estacionalidad
│   ├── politicas.py             # SS/ROP/EOQ, (s,Q), (R,S), (R,s,Q), costo analítico
│   ├── motor_des.py             # Simulación de eventos discretos — SimPy
│   ├── escenarios.py            # Réplicas, IC, comparación, recomendaciones
│   ├── validacion.py            # Retro-prueba de fidelidad y pisos irreducibles
│   ├── explicabilidad.py        # Shapley exacto, tornado, descomposición
│   ├── narrativa.py             # Interpretación obligatoria de cada figura y tabla
│   ├── favorita.py              # Corte real de Favorita, tienda 44
│   ├── prophet_demanda.py       # Holdout de Prophet sobre la clase A
│   ├── crisp_dm.py              # EDA, grilla, validación cruzada y pruebas
│   ├── vista_crisp.py           # Pestaña CRISP-DM
│   └── modelo_gemelo.py         # Interfaz en 10 pestañas
│
└── modulo_copiloto/             # OPCIONAL: sin clave de API se desactiva solo
    ├── esquemas.py              # Las 8 palancas que el modelo puede pedir, tipadas
    ├── proveedor.py             # Acceso al LLM y si el copiloto puede correr
    ├── guardia.py               # Cerco numérico sobre el texto generado
    ├── grafo.py                 # El grafo: traducir → aprobar → simular → cercar
    └── langflow_cliente.py      # Traducción vía el flujo de Langflow, si está configurado
```

---

## Las 10 pestañas

| Pestaña | Qué responde |
|---|---|
| 📦 **Alcance y datos** | Qué SKU entran, cuánta demanda no se registró y de dónde sale cada parámetro |
| 📈 **Prophet** | Holdout de 28 días del pronóstico local sobre la clase A |
| 📋 **CRISP-DM** | EDA, entrenamiento, hiperparámetros, validación cruzada, selección, pruebas y reporte |
| ▶️ **Corrida** | Qué pasa bajo un escenario: costo, servicio, diente de sierra por SKU |
| 📊 **Escenarios** | Comparación con IC, y cuánto cuesta prometer más nivel de servicio |
| ✅ **Fidelidad** | ¿El gemelo reproduce el pasado que sí ocurrió? |
| 🔬 **Por qué este SKU** | Descomposición exacta de Shapley del costo de un producto |
| 🌐 **Qué mueve el costo** | Tornado global y descomposición de un escenario palanca a palanca |
| 🔁 **Recomendaciones** | El lazo de retorno: qué comprar, cuánto y por qué |
| 🧭 **Copiloto** | Una pregunta en español convertida en escenario, aprobada por una persona y simulada por el motor |

---

## Advertencias que deben acompañar a toda salida

1. **Las unidades son reales; el almacén reconstruido no.** Las ventas salen de
   Corporación Favorita (tienda 44, Quito). El dataset no trae stock, precios ni
   tiempos de entrega.
2. **La penalidad de ruptura es el supuesto más delicado** y debe acordarse con
   la representante del negocio.
3. **La retro-prueba valida bajo la política que ya se usaba**, no bajo una nueva.
4. **Las recomendaciones son sugerencias**: el humano se queda en el lazo.
5. **El copiloto traduce y redacta, no calcula.** Propone palancas que una persona
   aprueba antes de simular, y las cifras siempre las produce el motor. Todo número
   de su texto se verifica contra los resultados; si uno no aparece, el texto entero
   se descarta y queda la interpretación determinista.
