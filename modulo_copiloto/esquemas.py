# modulo_copiloto/esquemas.py
"""
Contrato tipado entre la pregunta del negocio y las palancas del motor.

ESTE ARCHIVO ES TODA LA SUPERFICIE POR LA QUE EL LLM TOCA LA SIMULACIÓN
------------------------------------------------------------------------
El modelo de lenguaje no calcula: traduce. Su única salida permitida es un
EscenarioPropuesto, y uno que no valida no llega al motor. Todo lo que el
modelo puede pedirle al gemelo cabe aquí, y son ocho números.

Si en la sustentación preguntan «¿y si el modelo se inventa un escenario?»,
la respuesta no es una promesa sino este esquema: un factor fuera de rango,
una política que el motor no implementa, una palanca que no existe o un nivel
de servicio que no está tabulado se rechazan ANTES de correr nada, y el error
vuelve al modelo para que corrija.

POR QUÉ nivel_servicio SOLO ACEPTA LOS VALORES TABULADOS
---------------------------------------------------------
politicas.construir_politica resuelve z así (politicas.py:98):

    z = NIVELES_SERVICIO.get(round(nivel, 3), sku_estado.get("z", 1.6449))

Un nivel fuera de la tabla NO falla: cae al z que el SKU ya traía. La política
quedaría declarando «nivel_servicio: 0.97» mientras simula con el z de 95%, y
el escenario reportaría un nivel de servicio que no es el que corrió. Limitar
el campo a las claves tabuladas es la única forma de que lo pedido y lo
simulado sean lo mismo.

POR QUÉ extra="forbid"
-----------------------
motor_des.simular lee las palancas con escenario.get(...): una clave que no
reconoce la ignora sin decir nada. Si el modelo inventara «factor_merma», el
escenario correría sin esa palanca y el resultado se presentaría como si la
hubiera aplicado. Prohibir campos extra convierte ese silencio en un error.

SOBRE LAS ANOTACIONES DE TIPO
------------------------------
modulo_simulacion no las usa. Aquí sí, y no es inconsistencia de estilo: son
el mecanismo. Pydantic construye el validador desde las anotaciones, así que
el tipo ES la regla.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from modulo_simulacion.parametros import NIVELES_SERVICIO

# Los únicos niveles con z tabulado. Ver el encabezado: fuera de aquí, lo
# pedido y lo simulado dejan de coincidir en silencio.
NIVELES_VALIDOS = tuple(sorted(NIVELES_SERVICIO))

# Las ocho palancas que motor_des.simular() efectivamente lee
# (motor_des.py:320-331). No es documentación: es el límite de lo pedible.
PALANCAS = (
    "factor_demanda",
    "factor_lead_time",
    "factor_desv_lead_time",
    "nivel_servicio",
    "tipo_politica",
    "periodo_revision_dias",
    "fraccion_presupuesto",
    "presupuesto_mensual",
)

# Etiquetas para la tabla de confirmación. El humano aprueba palancas, no JSON.
ETIQUETAS = {
    "factor_demanda": "Factor sobre la demanda media",
    "factor_lead_time": "Factor sobre el tiempo de entrega",
    "factor_desv_lead_time": "Factor sobre la variabilidad del tiempo de entrega",
    "nivel_servicio": "Nivel de servicio objetivo",
    "tipo_politica": "Política de reposición",
    "periodo_revision_dias": "Periodo de revisión (días)",
    "fraccion_presupuesto": "Fracción del presupuesto de compra",
    "presupuesto_mensual": "Presupuesto mensual (S/.)",
}


class EscenarioPropuesto(BaseModel):
    """Un escenario what-if traducido desde una pregunta en lenguaje natural."""

    model_config = ConfigDict(extra="forbid")

    nombre: str = Field(
        ..., min_length=3, max_length=80,
        description="Nombre corto del escenario, como aparecería en una tabla. "
                    "Ejemplo: 'Campaña de fiestas (+20% demanda)'.")
    pregunta: str = Field(
        ..., min_length=5, max_length=200,
        description="La pregunta del negocio que este escenario responde, "
                    "reformulada en una oración.")
    mecanismo: str = Field(
        ..., min_length=20, max_length=400,
        description="Qué mecanismo del modelo se activa y por qué eso responde "
                    "la pregunta. No incluyas cifras de resultados: el escenario "
                    "todavía no se ha simulado.")

    factor_demanda: Optional[float] = Field(
        None, ge=0.1, le=5.0,
        description="Multiplica la demanda media. 1.20 = sube 20%; 0.85 = baja 15%. "
                    "Omitir si la pregunta no habla de la demanda.")
    factor_lead_time: Optional[float] = Field(
        None, ge=0.1, le=10.0,
        description="Multiplica el tiempo de entrega del proveedor. 2.0 = el doble.")
    factor_desv_lead_time: Optional[float] = Field(
        None, ge=0.1, le=10.0,
        description="Multiplica la variabilidad del tiempo de entrega. Úsalo junto "
                    "con factor_lead_time cuando la pregunta sugiera que el "
                    "proveedor además se volvió errático.")
    nivel_servicio: Optional[float] = Field(
        None,
        description="Nivel de servicio objetivo. SOLO se admiten 0.90, 0.95, 0.975 "
                    "y 0.99: son los únicos con z tabulado en el motor.")
    tipo_politica: Optional[Literal["(s,Q)", "(R,S)", "(R,s,Q)"]] = Field(
        None,
        description="Política de reposición. '(s,Q)' revisión continua; '(R,S)' "
                    "revisión periódica hasta un nivel objetivo; '(R,s,Q)' la "
                    "práctica actual de la empresa.")
    periodo_revision_dias: Optional[int] = Field(
        None, ge=1, le=90,
        description="Cada cuántos días se revisa el inventario. Solo tiene sentido "
                    "con '(R,S)' o '(R,s,Q)'.")
    fraccion_presupuesto: Optional[float] = Field(
        None, gt=0.0, le=2.0,
        description="Fracción de lo que la reposición pide que sí se puede "
                    "financiar. 0.85 = alcanza para el 85%.")
    presupuesto_mensual: Optional[float] = Field(
        None, gt=0.0,
        description="Presupuesto mensual de compra en soles, en absoluto. Prefiere "
                    "fraccion_presupuesto salvo que la pregunta dé un monto exacto.")

    # ------------------------------------------------------------------
    # Validaciones que no son de rango sino de coherencia. Cada una existe
    # porque el motor, sin ella, haría algo distinto de lo pedido sin avisar.
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _nivel_servicio_tabulado(self):
        if self.nivel_servicio is not None:
            if round(self.nivel_servicio, 3) not in NIVELES_VALIDOS:
                admitidos = ", ".join(str(n) for n in NIVELES_VALIDOS)
                raise ValueError(
                    f"nivel_servicio={self.nivel_servicio} no está tabulado. "
                    f"El motor solo tiene z para: {admitidos}. Elige el más "
                    f"cercano a la intención de la pregunta.")
        return self

    @model_validator(mode="after")
    def _al_menos_una_palanca(self):
        if not any(getattr(self, c) is not None for c in PALANCAS):
            raise ValueError(
                "El escenario no mueve ninguna palanca, así que sería idéntico al "
                "caso Base. Si la pregunta no se puede expresar con las palancas "
                "disponibles, dilo en vez de devolver un escenario vacío.")
        return self

    @model_validator(mode="after")
    def _revision_coherente_con_politica(self):
        if self.periodo_revision_dias is not None and self.tipo_politica == "(s,Q)":
            raise ValueError(
                "periodo_revision_dias no tiene sentido con '(s,Q)': esa política "
                "revisa de forma continua. Usa '(R,S)' o '(R,s,Q)', o quita el "
                "periodo de revisión.")
        return self

    @model_validator(mode="after")
    def _presupuesto_no_duplicado(self):
        if self.fraccion_presupuesto is not None and self.presupuesto_mensual is not None:
            raise ValueError(
                "No declares fraccion_presupuesto y presupuesto_mensual a la vez: "
                "el motor da precedencia al monto absoluto e ignora la fracción sin "
                "avisar. Elige uno.")
        return self

    # ------------------------------------------------------------------
    # Salidas hacia el motor y hacia la pantalla
    # ------------------------------------------------------------------
    def a_palancas(self) -> dict:
        """Solo las palancas seteadas, con los nombres exactos que lee el motor."""
        return {c: getattr(self, c) for c in PALANCAS if getattr(self, c) is not None}

    def a_cfg(self, codigo: str = "COPILOTO") -> dict:
        """El cfg que consume escenarios.ejecutar_escenario(..., cfg=...)."""
        return {
            "codigo": codigo,
            "nombre": self.nombre,
            "pregunta": self.pregunta,
            "palancas": self.a_palancas(),
            "mecanismo": self.mecanismo,
        }

    def filas_confirmacion(self) -> list:
        """Filas legibles para que el humano apruebe palancas, no JSON."""
        filas = []
        for clave, valor in self.a_palancas().items():
            filas.append({
                "Palanca": ETIQUETAS.get(clave, clave),
                "Valor": valor,
                "Clave del motor": clave,
            })
        return filas


class Traduccion(BaseModel):
    """
    La respuesta del traductor: un escenario, o la razón por la que no hay uno.

    POR QUÉ ESTE ENVOLTORIO EXISTE
    -------------------------------
    Antes se pedía directamente un EscenarioPropuesto con with_structured_output.
    Eso OBLIGA al modelo a devolver un escenario válido: es su único canal de
    salida. Y el esquema exige al menos una palanca.

    El resultado era que a un «hola, ¿qué se puede hacer aquí?» el copiloto
    respondía inventando «Aumento de demanda (+20%)» junto con una pregunta de
    negocio que el usuario nunca hizo. No era el modelo portándose mal: era lo
    único que se le había dejado hacer. La instrucción le pedía fallar y el
    esquema no le daba forma de fallar.

    El daño no es cosmético. Si el copiloto inventa la PREGUNTA, quien lo mire
    va a suponer que también inventa la RESPUESTA — y con razón. Un escenario
    fabricado destruye exactamente la confianza que el cerco numérico construye.

    Con este envoltorio, «no se puede traducir» es una salida legítima y
    tipada. Rechazar es tan válido como proponer.
    """

    model_config = ConfigDict(extra="forbid")

    escenario: Optional[EscenarioPropuesto] = Field(
        None,
        description="El escenario, SOLO si la pregunta describe un what-if que las "
                    "palancas disponibles pueden expresar.")
    motivo: Optional[str] = Field(
        None, min_length=15, max_length=600,
        description="SOLO si no hay escenario posible. Explica en lenguaje del negocio "
                    "por qué esta pregunta no se puede convertir en una simulación: un "
                    "saludo, una pregunta sobre la herramienta misma, o algo que el "
                    "motor no modela (precios, rutas, proveedores nuevos, más "
                    "almacenes, personal). No propongas un escenario de consuelo.")

    @model_validator(mode="after")
    def _exactamente_una_salida(self):
        if (self.escenario is None) == (self.motivo is None):
            raise ValueError(
                "Devuelve exactamente uno: `escenario` si la pregunta es un what-if "
                "simulable, o `motivo` si no lo es. Nunca ambos ni ninguno.")
        return self


def catalogo_palancas() -> list:
    """
    Las ocho palancas en lenguaje de negocio, con un ejemplo de pregunta.

    Se genera desde PALANCAS y ETIQUETAS, no de una lista escrita aparte: si
    mañana el motor gana una palanca, esta tabla la muestra sola. Es lo que se
    le enseña a quien pregunta «¿qué se puede hacer aquí?», y sirve para dejar
    claro que el copiloto no tiene un menú de preguntas preparadas sino un
    espacio de ocho perillas que se combinan libremente.
    """
    ejemplos = {
        "factor_demanda": "¿Y si la demanda sube 30% en campaña?",
        "factor_lead_time": "¿Y si el proveedor se demora el doble?",
        "factor_desv_lead_time": "¿Y si el proveedor se vuelve impredecible?",
        "nivel_servicio": "¿Cuánto cuesta prometer 99% de servicio?",
        "tipo_politica": "¿Y si compro una vez por semana en vez de revisar siempre?",
        "periodo_revision_dias": "¿Y si reviso cada 15 días en lugar de cada 7?",
        "fraccion_presupuesto": "¿Y si solo puedo financiar el 70% de la reposición?",
        "presupuesto_mensual": "¿Y si mi presupuesto de compra es S/. 8,000 al mes?",
    }
    return [{"Palanca": ETIQUETAS[c], "Una pregunta que la mueve": ejemplos[c],
             "Clave del motor": c} for c in PALANCAS]
