# modulo_simulacion/parametros.py
"""
Parámetros del motor del Gemelo Digital de Inventario — Comercializadora SOAZ.

Principio de diseño: ningún número mágico vive dentro de la lógica. Todo
coeficiente está aquí, con su unidad, su rango admisible y su PROCEDENCIA.
La procedencia es lo que permite defender el modelo ante un jurado: distingue
lo que viene de teoría de inventarios, lo que se estima de datos y lo que es
un supuesto del autor todavía sin calibrar.

Códigos de procedencia:
    TEORIA      — se deriva de la teoría clásica de inventarios (Silver, Pyke & Thomas)
    ESTIMADO    — se calcula del histórico de movimientos de SOAZ
    SUPUESTO    — supuesto del autor; DEBE validarse con la representante del negocio
"""

# ==========================================================================
# 1. IDENTIDAD DEL CASO
# ==========================================================================
NEGOCIO = {
    "nombre": "Comercializadora SOAZ",
    "direccion": "Calle Progreso 760, Otuzco — La Libertad",
    "moneda": "S/.",
    "dias_habiles_semana": 6,     # SUPUESTO — abre de lunes a sábado
    "dias_por_mes": 26,
}

# ==========================================================================
# 2. CATÁLOGO — qué SKU entran al gemelo
# ==========================================================================
# El alcance declarado del proyecto es la clase A del análisis ABC: los SKU
# que concentran ~80% del movimiento. Modelar el catálogo completo multiplica
# el trabajo sin cambiar ninguna conclusión.
CATALOGO = {
    "n_sku_total": 60,
    "umbral_clase_a": 0.80,       # participación acumulada que define la clase A
    "solo_clase_a": True,
}

# Familias del catálogo: gobiernan demanda, precio y perecibilidad.
# Una comercializadora de abarrotes en sierra no tiene un solo tipo de producto,
# y mezclar arroz con lácteos en un mismo modelo esconde el problema de merma.
FAMILIAS = {
    "Abarrote seco": {
        "demanda_diaria_media": (2.5, 13.0),      # rango uniforme entre SKU
        "coef_variacion": (0.25, 0.55),       # ESTIMADO del histórico por SKU
        "costo_unitario": (2.5, 18.0),        # S/.
        "vida_util_dias": 540,                # SUPUESTO
        "estacionalidad": 0.10,               # amplitud relativa
    },
    "Bebida": {
        "demanda_diaria_media": (3.5, 17.0),
        "coef_variacion": (0.30, 0.70),
        "costo_unitario": (1.2, 9.0),
        "vida_util_dias": 300,
        "estacionalidad": 0.28,               # SUPUESTO — sube en fiestas y en calor
    },
    "Lácteo y perecible": {
        "demanda_diaria_media": (2.0, 9.0),
        "coef_variacion": (0.35, 0.80),
        "costo_unitario": (3.0, 14.0),
        "vida_util_dias": 21,                 # SUPUESTO — aquí la merma sí muerde
        "estacionalidad": 0.12,
    },
    "Limpieza y hogar": {
        "demanda_diaria_media": (1.2, 6.5),
        "coef_variacion": (0.40, 0.90),
        "costo_unitario": (4.0, 26.0),
        "vida_util_dias": 720,
        "estacionalidad": 0.08,
    },
}

# ==========================================================================
# 3. PROVEEDORES — el tiempo de entrega es una VARIABLE, no una constante
# ==========================================================================
# Otuzco está a ~2.5 h de Trujillo por carretera de sierra. El supuesto de
# lead time constante, que es el que usa la mayoría de los trabajos de curso,
# es justamente el que no aplica aquí.
PROVEEDORES = {
    "Mayorista Trujillo": {
        "lead_time_medio_dias": 4.0,     # ESTIMADO
        "lead_time_desv_dias": 1.6,      # ESTIMADO — la variabilidad es el problema
        "prob_incumplimiento": 0.06,     # SUPUESTO — pedido que llega incompleto
        "fraccion_si_incumple": 0.65,    # SUPUESTO — cuánto llega cuando falla
        "lote_minimo": 1,
    },
    "Distribuidor regional": {
        "lead_time_medio_dias": 7.0,
        "lead_time_desv_dias": 3.2,
        "prob_incumplimiento": 0.10,
        "fraccion_si_incumple": 0.50,
        "lote_minimo": 6,
    },
    "Productor local": {
        "lead_time_medio_dias": 2.0,
        "lead_time_desv_dias": 0.8,
        "prob_incumplimiento": 0.14,
        "fraccion_si_incumple": 0.70,
        "lote_minimo": 1,
    },
}

# ==========================================================================
# 4. COSTOS
# ==========================================================================
COSTOS = {
    "costo_pedido": 35.0,               # SUPUESTO — S/. por orden emitida (transporte + gestión)
    "tasa_mantenimiento_anual": 0.28,   # SUPUESTO — 28% del valor inmovilizado al año
    "penalidad_ruptura_factor": 1.40,   # SUPUESTO — margen perdido + costo de perder al cliente
    "valor_merma_factor": 1.00,         # una unidad vencida se pierde a costo
}

# ==========================================================================
# 5. POLÍTICAS DE REPOSICIÓN
# ==========================================================================
# z corresponde al nivel de servicio de ciclo objetivo.
NIVELES_SERVICIO = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}

POLITICA_POR_DEFECTO = {
    "tipo": "(s,Q)",              # revisión continua
    "nivel_servicio": 0.95,       # SUPUESTO — es una DECISIÓN, y su costo se mide
    "periodo_revision_dias": 7,   # solo aplica a (R,S)
}

# ==========================================================================
# 6. ESCENARIOS WHAT-IF
# ==========================================================================
# Cada escenario declara qué palanca mueve y qué pregunta del negocio responde.
# Sin la pregunta, un escenario es un número sin destinatario.
ESCENARIOS = {
    "Base": {
        "nombre": "Política actual",
        "pregunta": "¿Qué pasa si seguimos comprando como hasta ahora?",
        "palancas": {},
        "mecanismo": "Trayectoria contrafactual: el punto de comparación de todo lo demás.",
    },
    "S1": {
        "nombre": "Campaña de fiestas (+20% demanda)",
        "pregunta": "¿Aguantamos diciembre sin quebrar stock?",
        "palancas": {"factor_demanda": 1.20},
        "mecanismo": "Sube la demanda media sin cambiar la política: expone si el "
                     "stock de seguridad actual absorbe el pico o no.",
    },
    "S2": {
        "nombre": "Proveedor se demora el doble",
        "pregunta": "¿Y si el proveedor duplica su tiempo de entrega?",
        "palancas": {"factor_lead_time": 2.0, "factor_desv_lead_time": 1.5},
        "mecanismo": "El tiempo de entrega entra al cuadrado en el stock de seguridad: "
                     "es la palanca que más lo mueve, y la que el negocio no controla.",
    },
    "S3": {
        "nombre": "Servicio objetivo 99%",
        "pregunta": "¿Cuánto cuesta pasar de 95% a 99% de nivel de servicio?",
        "palancas": {"nivel_servicio": 0.99},
        "mecanismo": "Sube z y con él el stock de seguridad. Compra servicio con "
                     "capital inmovilizado; el gemelo pone precio a esa compra.",
    },
    "S4": {
        "nombre": "Revisión semanal en vez de continua",
        "pregunta": "¿Podemos comprar una vez por semana en lugar de estar revisando?",
        "palancas": {"tipo_politica": "(R,S)", "periodo_revision_dias": 7},
        "mecanismo": "Menos órdenes y menos gestión, pero hay que cubrir el riesgo de "
                     "L + R días en vez de L. Es un intercambio, no una mejora.",
    },
    "S5": {
        "nombre": "Presupuesto de compra al 85%",
        "pregunta": "¿Y si solo puedo financiar el 85% de lo que la reposición pide?",
        "palancas": {"fraccion_presupuesto": 0.85},
        "mecanismo": "Racionamiento: cuando el presupuesto no alcanza, se prioriza por "
                     "criticidad. Revela qué SKU se sacrifican y a qué costo.",
    },
}

# ==========================================================================
# 7. EJECUCIÓN
# ==========================================================================
EJECUCION = {
    "dias_historico": 540,        # ~18 meses de registro para estimar parámetros
    "dias_simulacion": 90,        # horizonte del escenario, según la propuesta
    "n_replicas": 30,             # nunca 1: el modelo es estocástico
    "semilla_base": 2026,
    "nivel_confianza": 0.95,
}

# ==========================================================================
# 8. VALIDACIÓN DE FIDELIDAD — umbrales fijados A PRIORI
# ==========================================================================
# Se declaran antes de correr la prueba. Fijarlos después de ver el resultado
# es lo que convierte una validación en una justificación.
# NOTA METODOLÓGICA que hay que poder defender:
# el primer diseño de esta prueba usaba el MAPE punto a punto del stock diario,
# con umbral de 20%. Ese criterio es INALCANZABLE por construcción y conviene
# entender por qué antes de fijar cualquier umbral sobre una simulación.
#
# El stock sigue un diente de sierra: baja con la venta y salta cuando llega el
# pedido. El día exacto del salto depende del tiempo de entrega, que es
# aleatorio. Al promediar réplicas, los saltos se desalinean y el promedio
# suaviza el diente. Comparar ese promedio suave contra UN diente real produce
# un error grande aunque el modelo sea perfecto.
#
# La prueba está en el propio motor: el gemelo comparado contra SÍ MISMO con
# otra semilla da ~56% de MAPE. Ningún modelo puede bajar de ahí. Por eso el
# MAPE se reporta SIEMPRE relativo a ese piso irreducible, y los criterios de
# aceptación se fijan sobre estadísticos que sí son estables: el nivel medio,
# la cobertura de la banda y los conteos de eventos.
FIDELIDAD = {
    "dias_ajuste": 360,             # con qué parte del histórico se estiman parámetros
    "dias_prueba": 90,              # contra qué parte se valida, sin haberla visto
    "umbral_error_nivel": 20.0,     # % — error del nivel MEDIO de stock, clase A
    "umbral_cobertura_relativa": 0.90,# cobertura del gemelo / cobertura alcanzable
    "umbral_mape_relativo": 1.20,   # MAPE del gemelo / MAPE irreducible del propio motor
    "umbral_ruptura_relativa": 1.20,# Δ días del gemelo / Δ días irreducible
    "umbral_error_pedidos": 15.0,   # % de diferencia en número de órdenes
}

# ==========================================================================
# 9. ETIQUETAS Y FACTORES DE EXPLICABILIDAD
# ==========================================================================
# Los 8 factores sobre los que se calcula la descomposición de Shapley del
# costo total de un SKU. Son los que el negocio puede reconocer y, algunos,
# los que puede mover.
ETIQUETAS_FACTORES = {
    "demanda_media":      "Demanda media diaria",
    "demanda_desv":       "Variabilidad de la demanda",
    "lead_time_medio":    "Tiempo de entrega medio",
    "lead_time_desv":     "Variabilidad del tiempo de entrega",
    "costo_unitario":     "Costo unitario del producto",
    "costo_pedido":       "Costo de emitir una orden",
    "nivel_servicio_z":   "Nivel de servicio exigido (z)",
    "cantidad_pedido":    "Tamaño del lote de compra",
    "vida_util":          "Vida útil del producto",
}
FACTORES_SHAPLEY = list(ETIQUETAS_FACTORES.keys())

PROCEDENCIA_RESUMEN = {
    "TEORIA": "Deriva de la teoría clásica de inventarios con demanda y tiempo de "
              "entrega estocásticos.",
    "ESTIMADO": "Se calcula del histórico de movimientos de SOAZ, corrigiendo "
                "por demanda censurada.",
    "SUPUESTO": "Supuesto del autor. Debe validarse con la representante del negocio "
                "antes de usar el modelo para decidir compras.",
}
