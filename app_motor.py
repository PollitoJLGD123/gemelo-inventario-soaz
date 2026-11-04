# app_motor.py
"""
Punto de entrada del motor del Gemelo Digital de Inventario — SOAZ.

    streamlit run app_motor.py

No requiere PostgreSQL ni autenticación: mientras el sistema web de la tesis
no esté operativo, el motor genera su propio histórico de movimientos y trabaja
sobre él. Cuando el sistema exista, se cambia FUENTE_DATOS en config.py y el
resto del motor no se toca.
"""
import streamlit as st

from config import NOMBRE_SISTEMA, INSTITUCION

st.set_page_config(page_title="Gemelo Digital de Inventario — SOAZ",
                   page_icon="📦", layout="wide", initial_sidebar_state="expanded")

st.title("📦 Gemelo Digital de Inventario — Comercializadora SOAZ")
st.caption(f"{INSTITUCION} · Otuzco, La Libertad · Simulación de eventos discretos "
           f"con SimPy. Toda figura y toda tabla incluyen su bloque de interpretación "
           f"y explicabilidad.")

with st.expander("📌 Antes de leer cualquier resultado", expanded=False):
    st.markdown("""
Cuatro advertencias que deben acompañar a toda salida de este motor:

1. **Las unidades son reales; el almacén reconstruido no.** Las ventas salen
   de Corporación Favorita (tienda 44, Quito). El dataset no trae stock,
   precios ni tiempos de entrega: el gemelo reconstruye las compras con la
   práctica semanal declarada y un costo supuesto. Sirve para comparar
   políticas sobre esa demanda, no para afirmar el costo contable de SOAZ.

2. **Las ventas no son la demanda.** Los días con ruptura no registran al
   cliente que se fue. El motor corrige esa censura por Esperanza-Maximización;
   la pestaña *Alcance y datos* muestra cuánto sesgo elimina.

3. **La penalidad de ruptura es el supuesto más delicado.** Nadie registra al
   cliente que se fue, y ese número mueve el resultado más que ningún otro.
   Debe acordarse con la representante del negocio antes de decidir compras.

4. **Las recomendaciones son sugerencias.** El encargado aprueba o rechaza: el
   humano se queda en el lazo, y su tasa de aceptación es lo que mide si el
   gemelo sirve.
""")

st.markdown("---")

from modulo_simulacion.modelo_gemelo import ejecutar_simulacion_interfaz

ejecutar_simulacion_interfaz()
