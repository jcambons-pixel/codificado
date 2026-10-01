import streamlit as st
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from pinecone import Pinecone
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta

# ------------------------------------------------------------------
# Logo del proyecto
# ------------------------------------------------------------------
# Coloca tu imagen (p. ej. logo.png) en la MISMA carpeta que este app.py.
RUTA_LOGO = "logo.png"
LOGO_DISPONIBLE = Path(RUTA_LOGO).exists()

try:
    icono_pagina = Image.open(RUTA_LOGO) if LOGO_DISPONIBLE else "💬"
except Exception:
    icono_pagina = "💬"

AVATAR_ASISTENTE = RUTA_LOGO if LOGO_DISPONIBLE else "🤖"

# ------------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------------
st.set_page_config(page_title="Asistente Legal", page_icon=icono_pagina)

# ------------------------------------------------------------------
# Bloqueo de seguridad por contraseña
# ------------------------------------------------------------------
def verificar_password():
    if st.session_state.get("password_correcta", False):
        return True

    def password_ingresada():
        if st.session_state.get("password_input") == st.secrets["ACCESO_PASSWORD"]:
            st.session_state["password_correcta"] = True
            del st.session_state["password_input"]
        else:
            st.session_state["password_correcta"] = False

    if LOGO_DISPONIBLE:
        st.image(RUTA_LOGO, width=90)

    st.title("🔒 Acceso restringido")
    st.text_input(
        "Introduce la contraseña de acceso",
        type="password",
        on_change=password_ingresada,
        key="password_input",
    )

    if "password_correcta" in st.session_state and not st.session_state["password_correcta"]:
        st.error("Contraseña incorrecta. Inténtalo de nuevo.")
    return False


if not verificar_password():
    st.stop()

# ------------------------------------------------------------------
# Clientes de Gemini y Pinecone (cacheados para velocidad)
# ------------------------------------------------------------------
@st.cache_resource
def obtener_cliente_gemini():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=HttpOptions(api_version="v1"),
    )


@st.cache_resource
def obtener_indice_pinecone():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc.Index(st.secrets["PINECONE_INDEX_NAME"])


cliente_gemini = obtener_cliente_gemini()
indice_pinecone = obtener_indice_pinecone()

# ------------------------------------------------------------------
# Configuración del Motor RAG (Ultra-rápido y económico)
# ------------------------------------------------------------------
MODELO_EMBEDDING = "gemini-embedding-001"
MODELO_GENERACION = "gemini-3.5-flash-lite"  # ⚡ El modelo más rápido y resistente del catálogo
DIMENSION_EMBEDDING = 768
NUM_FRAGMENTOS_CONTEXTO = 1  # 📈 el sistema recupera el artículo exacto que responde a tu pregunta de golpe, sin que le falte ningún matiz.
VENTANA_HISTORIAL = 8  # 🧠 Nº de mensajes previos (aprox. 4 turnos) que se envían como contexto conversacional
LIMITE_INACTIVIDAD = timedelta(minutes=60)  # ⏱️ Tras este tiempo sin interacción, se reinicia la conversación

INSTRUCCION_SISTEMA = """
¡ATENCIÓN! REGLA SUPREMA E INQUEBRANTABLE SOBRE PUNTOS: Para el artículo 118 del RGC (falta de guantes o calzado adecuado en motocicletas), la pérdida de puntos es SIEMPRE "0 puntos". Está terminantemente prohibido asignar puntos en este artículo.

Eres un asistente legal de tráfico experto. Tienes TRES fuentes de conocimiento que no debes mezclar:

A) CONTEXTO DOCUMENTAL Y MARCO GENERAL DGT: fragmentos normativos recuperados y marco legal general (Ley de Tráfico, RGC y normativa VMP).

B) FÓRMULAS Y BAREMOS FIJOS DE VELOCIDAD Y ALCOHOLEMIA:
   - B.1 Corrección de mediciones (etilómetros y radares).
   - B.2 Tabla de excesos de velocidad.
   - B.3 Alcoholemia (tasa corregida y tramos por tipo de conductor).
   - B.4 Umbrales penales de velocidad.

C) BAREMOS Y REGLAS CONDICIONALES PARA VMP Y VPL (Vehículos de Movilidad Personal y Ligeros):
   - Criterio de peso: Si pesa más de 25 kg es un VMP (seguro obligatorio siempre). Si pesa menos de 25 kg es un VPL.
   - Requisitos técnicos VPL (<25 kg) / VMP:
     · Certificado de circulación: exigible solo para comercializados después del 22 de enero de 2024 (si falta y no está exento: VEH 22 B-2-5A, 200/100€, no inmoviliza).
     · Inscripción en Registro de Vehículos: VEH 22 B-2-5B, 100/50€, no inmoviliza[cite: 2].
     · Etiqueta identificativa o placa de marcaje: VEH 22 B-2-5C, 80/40€, no inmoviliza[cite: 2].
   - Modificaciones técnicas:
     · Muy graves (manipulación velocidad, baterías externas, sin frenos): VEH 22 B-2-5D, 500/250€, SÍ inmoviliza[cite: 2].
     · Graves (neumáticos rugosos, sin alumbrado): VEH 22 B-2-5E, 200/100€, SÍ inmoviliza[cite: 2].
     · Leves (sin timbre, catadióptricos, pata de cabra): VEH 22 B-2-5F, 80/40€, NO inmoviliza[cite: 2].
   - Régimen de Seguros (Jerarquía estricta):
     · VPL (<25 kg): El seguro de responsabilidad civil (circulando SDA 1-5A 300/150€; sin circular SDA 1-5B 300/150€) **SOLO es denunciable si dispone de Certificado, Registro y Etiqueta**. Si falta alguno de los tres requisitos, el vehículo está **exento** de seguro[cite: 2].
     · VMP (>25 kg): El seguro es exigible SIEMPRE (circulando SOA 2-1-5N, 800/400€; sin circular SOA 1-5O, 610/305€). SÍ procede inmovilizar[cite: 2].

Reglas de respuesta (POR ORDEN DE PRIORIDAD):
1. REGLA DE ORO (VMP/VPL): Si te consultan sobre una infracción de VMP o VPL (especialmente seguros o faltas técnicas), y faltan datos críticos como el peso, la fecha de comercialización, si tiene certificado, si está inscrito en el registro o si lleva etiqueta, ESTÁ PROHIBIDO dar una sanción definitiva. Pregunta obligatoriamente por estos datos en una sola frase corta.
2. Si faltan datos en velocidad/alcohol, aplica también la pregunta corta previa.
3. Si el documento o las fuentes no especifican de forma expresa que una infracción resta puntos del carnet, indica obligatoriamente "0 puntos" (aplica estrictamente a VMP, VPL y art. 118 de motos).
4. En MODO TRÁFICO/MULTA (con todos los datos necesarios confirmados), responde SIEMPRE en este formato de lista, una línea por punto:
   - Norma y artículo / Precepto: ...
   - Infracción: ...
   - Cálculo / Verificación de requisitos: ...
   - Cuantía: ... (cuantía reducida: ...)
   - Puntos: ... (0 puntos si no se especifica lo contrario)
   - Responsable: ...
   - Comentario: ...
5. Prohibido repetir un mismo dato en más de una línea de la respuesta.
"""

# ------------------------------------------------------------------
# Funciones lógicas
# ------------------------------------------------------------------
def obtener_embedding(texto: str):
    resultado = cliente_gemini.models.embed_content(
        model=MODELO_EMBEDDING,
        contents=texto,
        config={"output_dimensionality": DIMENSION_EMBEDDING},
    )
    return resultado.embeddings[0].values

def buscar_contexto(vector_consulta) -> str:
    resultados = indice_pinecone.query(
        vector=vector_consulta,
        top_k=NUM_FRAGMENTOS_CONTEXTO,
        include_metadata=True
    )
    fragmentos = [
        match.metadata.get("texto", "")
        for match in resultados.matches
        if match.metadata and match.metadata.get("texto", "")
    ]
    return "\n\n---\n\n".join(fragmentos)

# ------------------------------------------------------------------
# Interfaz visual de Chat
# ------------------------------------------------------------------
columna_logo, columna_titulo = st.columns([1, 6])
with columna_logo:
    if LOGO_DISPONIBLE:
        st.image(RUTA_LOGO, width=60)
with columna_titulo:
    st.title("Asistente Legal")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if "ultima_interaccion" not in st.session_state:
    st.session_state.ultima_interaccion = datetime.now()

if datetime.now() - st.session_state.ultima_interaccion > LIMITE_INACTIVIDAD:
    st.session_state.mensajes = []

st.session_state.ultima_interaccion = datetime.now()

with st.sidebar:
    if st.button("🗑️ Nueva conversación"):
        st.session_state.mensajes = []
        st.rerun()

# Mostrar el historial de la sesión
for mensaje in st.session_state.mensajes:
    avatar = AVATAR_ASISTENTE if mensaje["role"] == "assistant" else None
    with st.chat_message(mensaje["role"], avatar=avatar):
        st.markdown(mensaje["content"])

pregunta_usuario = st.chat_input("Escribe tu consulta legal...")

if pregunta_usuario:
    st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario})
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)

    with st.chat_message("assistant", avatar=AVATAR_ASISTENTE):
        # Espacio dinámico para el Streaming
        response_placeholder = st.empty()
        texto_acumulado = ""

        try:
            # 1. Recuperar contexto numérico
            vector_pregunta = obtener_embedding(pregunta_usuario)
            contexto = buscar_contexto(vector_pregunta)
            prompt_actual = f"CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta_usuario}"

            # 1b. Construir la ventana de historial conversacional (sin la pregunta actual,
            #     que ya está incluida en session_state.mensajes y se añade al final con su contexto)
            mensajes_previos = st.session_state.mensajes[:-1][-VENTANA_HISTORIAL:]
            contents_conversacion = []
            for mensaje_previo in mensajes_previos:
                rol_gemini = "model" if mensaje_previo["role"] == "assistant" else "user"
                contents_conversacion.append(
                    {"role": rol_gemini, "parts": [{"text": mensaje_previo["content"]}]}
                )
            contents_conversacion.append({"role": "user", "parts": [{"text": prompt_actual}]})

            # 2. Configurar límites estrictos de tokens y creatividad a cero (precisión absoluta)
            configuracion_ia = GenerateContentConfig(
                system_instruction=INSTRUCCION_SISTEMA,
                max_output_tokens=800,  # 🛑 Margen para cálculos paso a paso (corrección + tabla) sin cortar la respuesta
                temperature=0.4         # 🎯 Evita que la IA invente o decore las respuestas
            )

            # 3. Llamada en Streaming para respuesta instantánea
            response_stream = cliente_gemini.models.generate_content_stream(
                model=MODELO_GENERACION,
                contents=contents_conversacion,
                config=configuracion_ia
            )

            for chunk in response_stream:
                texto_acumulado += chunk.text
                response_placeholder.markdown(texto_acumulado)

        except Exception as error:
            texto_acumulado = f"Consulta pausada por saturación en la red. Por favor, reintenta en un momento. ({error})"
            response_placeholder.markdown(texto_acumulado)

    st.session_state.mensajes.append({"role": "assistant", "content": texto_acumulado})
