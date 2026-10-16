import re
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
import streamlit as st
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from pinecone import Pinecone

try:
    from buscador import construir_indices, buscar_lexico, fusionar
    BUSCADOR_OK = True
except ImportError:
    BUSCADOR_OK = False

try:
    from calculos import cargar_datos, calcular, opciones_dimension_para_prompt
    CALCULOS_OK = True
except ImportError:
    CALCULOS_OK = False

    def cargar_datos():
        return None

    def calcular(*args, **kwargs):
        return {"estado": "sin_calculo", "texto": "", "etiqueta": ""}

    def opciones_dimension_para_prompt():
        return ""

# ------------------------------------------------------------------
# Logo del proyecto
# ------------------------------------------------------------------
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
# Clientes de Gemini y Pinecone
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
# Configuración general y ahorro de datos
# ------------------------------------------------------------------
MODELO_EMBEDDING = "gemini-embedding-001"
MODELO_GENERACION = "gemini-3.8-flash"  # O el modelo estándar que utilices
DIMENSION_EMBEDDING = 768
VENTANA_HISTORIAL = 4  # Reducido para ahorrar tokens y datos por petición
LIMITE_INACTIVIDAD = timedelta(minutes=60)

USAR_FILTROS_METADATOS = False
MODO_DEBUG = True

# Decorador para reintentar automáticamente si hay errores 503 (Alta demanda)
def con_reintento_api(func):
    def wrapper(*args, **kwargs):
        intentos = 3
        espera = 2
        for intento in range(intentos):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e)
                if "503" in error_str or "UNAVAILABLE" in error_str or "high demand" in error_str:
                    if intento < intentos - 1:
                        time.sleep(espera)
                        espera *= 2
                        continue
                raise e
    return wrapper

# ------------------------------------------------------------------
# 1. ROUTER: clasifica la consulta
# ------------------------------------------------------------------
INSTRUCCION_ROUTER = """
Clasifica la ÚLTIMA consulta del usuario en UNA de estas categorías y responde SOLO con la etiqueta, sin nada más:

MULTA_TRAFICO: sanción, cuantía, puntos, precepto o infracción de TRÁFICO (velocidad, alcoholemia, drogas, móvil, cinturón, casco, VMP, seguro, ITV, semáforos...).
MULTA_TRANSPORTE: sanción o tipificación de TRANSPORTE (tacógrafo, tiempos, autorización, tarjeta, exceso de peso, LOTT/ROTT, taxi, VTC, autobús, códigos tipo SG01.01...).
CONSULTA_GENERAL: pregunta conceptual o de conocimiento sobre una norma que NO pide rellenar una denuncia ni calcular una sanción concreta.
"""

MODOS_VALIDOS = ("MULTA_TRAFICO", "MULTA_TRANSPORTE", "CONSULTA_GENERAL")


@con_reintento_api
def clasificar_consulta(pregunta: str, historial: list) -> str:
    contexto_previo = "\n".join(
        f"{m['role']}: {m['content'][:200]}" for m in historial[-2:]
    )
    prompt = f"HISTORIAL RECIENTE:\n{contexto_previo}\n\nÚLTIMA CONSULTA:\n{pregunta}"
    try:
        r = cliente_gemini.models.generate_content(
            model=MODELO_GENERACION,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=INSTRUCCION_ROUTER,
                max_output_tokens=15,
                temperature=0,
            ),
        )
        etiqueta = (r.text or "").strip().upper()
        for modo in MODOS_VALIDOS:
            if modo in etiqueta:
                return modo
    except Exception:
        pass
    return "CONSULTA_GENERAL"


# ------------------------------------------------------------------
# 2. PROMPTS POR MODO
# ------------------------------------------------------------------
REGLAS_COMUNES = """
Responde siempre en español, basándote en el CONTEXTO recuperado. Si el contexto no contiene la respuesta, dilo claramente. No repitas un mismo dato.
"""

INSTRUCCION_MULTA_TRAFICO = REGLAS_COMUNES + """
MODO MULTA DE TRÁFICO. Eres un asistente de denuncias de tráfico (Baremo DGT).
Estructura de respuesta obligatoria (una línea por punto si aplica):
- Norma y artículo / Precepto: ...
- Infracción: ...
- Calificación: ...
- Cuantía: ... € (cuantía reducida: ... €)
- Puntos: ...
- Responsable: ...
"""

INSTRUCCION_MULTA_TRANSPORTE = REGLAS_COMUNES + """
MODO MULTA DE TRANSPORTE (LOTT/ROTT).
Estructura de respuesta obligatoria:
- Código: ...
- Concepto: ...
- Norma infringida: ...
- Norma sancionadora: ...
- Cuantía: ...
- Inmovilización: ...
- Responsable: ...
"""

INSTRUCCION_CONSULTA_GENERAL = REGLAS_COMUNES + """
MODO CONSULTA GENERAL. Explica de forma clara y concisa. Cita la norma y artículos relevantes sin usar formato de boletín de denuncia.
"""

CONFIG_MODOS = {
    "MULTA_TRAFICO": {
        "prompt": INSTRUCCION_MULTA_TRAFICO, "top_k": 2, "max_tokens": 700,
        "temp": 0.2, "etiqueta": "🚦 Multa de tráfico",
    },
    "MULTA_TRANSPORTE": {
        "prompt": INSTRUCCION_MULTA_TRANSPORTE, "top_k": 3, "max_tokens": 700,
        "temp": 0.2, "etiqueta": "🚛 Multa de transporte",
    },
    "CONSULTA_GENERAL": {
        "prompt": INSTRUCCION_CONSULTA_GENERAL, "top_k": 3, "max_tokens": 800,
        "temp": 0.4, "etiqueta": "📖 Consulta general",
    },
}

FUENTE_POR_MODO = {
    "MULTA_TRAFICO": "trafico",
    "MULTA_TRANSPORTE": "transporte",
    "CONSULTA_GENERAL": None,
}

# ------------------------------------------------------------------
# 3. Detección y Filtros
# ------------------------------------------------------------------
def detectar_codigo(texto: str):
    m = re.search(r"\b[A-Z]{2}\d{2}\.\d{2}\b", texto.upper())
    return m.group(0) if m else None

PATRON_CODIGO_FICHA = re.compile(r"\b[A-Z]{2}\d{2}\.\d{2}\b")

def es_ficha_transporte(texto: str) -> bool:
    return (
        "Baremo Sancionador de Transportes" in texto
        or "LOTT" in texto
        or "ROTT" in texto
        or bool(PATRON_CODIGO_FICHA.search(texto))
    )

def detectar_articulo(texto: str):
    m = re.search(r"\bart(?:[íi]culo|\.)?\s*(\d{1,3})", texto.lower())
    return int(m.group(1)) if m else None

def construir_filtro(modo: str, pregunta: str):
    if not USAR_FILTROS_METADATOS:
        return None
    filtro = {}
    fuente = FUENTE_POR_MODO.get(modo)
    if fuente:
        filtro["fuente"] = {"$eq": fuente}
    return filtro or None

# ------------------------------------------------------------------
# 4. Funciones de búsqueda y Embeddings
# ------------------------------------------------------------------
@con_reintento_api
def obtener_embedding(texto: str):
    resultado = cliente_gemini.models.embed_content(
        model=MODELO_EMBEDDING,
        contents=texto,
        config={"output_dimensionality": DIMENSION_EMBEDDING},
    )
    return resultado.embeddings[0].values


def _consultar(vector, top_k, filtro):
    kwargs = dict(vector=vector, top_k=top_k, include_metadata=True)
    if filtro:
        kwargs["filter"] = filtro
    return indice_pinecone.query(**kwargs)


SOBREMUESTREO = 4


def buscar_contexto(vector_consulta, top_k: int, filtro: dict | None = None, modo: str | None = None):
    resultados = _consultar(vector_consulta, top_k, filtro) if filtro else None
    descripcion = filtro

    if resultados is not None and resultados.matches:
        fragmentos = [
            m.metadata.get("texto", "")
            for m in resultados.matches
            if m.metadata and m.metadata.get("texto", "")
        ]
    else:
        pedir = top_k * SOBREMUESTREO if modo in ("MULTA_TRAFICO", "MULTA_TRANSPORTE") else top_k
        resultados = _consultar(vector_consulta, pedir, None)
        fragmentos = [
            m.metadata.get("texto", "")
            for m in resultados.matches
            if m.metadata and m.metadata.get("texto", "")
        ]
        descripcion = None
        if modo == "MULTA_TRANSPORTE":
            fragmentos = [f for f in fragmentos if es_ficha_transporte(f)][:top_k]
        elif modo == "MULTA_TRAFICO":
            fragmentos = [f for f in fragmentos if not es_ficha_transporte(f)][:top_k]
        else:
            fragmentos = fragmentos[:top_k]

    return "\n\n---\n\n".join(fragmentos), fragmentos, descripcion


def buscar_contexto_multi(vectores: list, top_k: int, filtro: dict | None = None, modo: str | None = None, limite_total: int | None = None):
    listas, descripcion = [], None
    for v in vectores:
        _, frs, descripcion = buscar_contexto(v, top_k, filtro, modo)
        listas.append(frs)
    vistos, fragmentos = set(), []
    for i in range(max((len(l) for l in listas), default=0)):
        for l in listas:
            if i < len(l) and l[i] not in vistos:
                vistos.add(l[i])
                fragmentos.append(l[i])
    fragmentos = fragmentos[: (limite_total or top_k * 2)]
    return "\n\n---\n\n".join(fragmentos), fragmentos, descripcion

# ------------------------------------------------------------------
# 4b. Descomposición de hechos
# ------------------------------------------------------------------
INSTRUCCION_DESCOMPOSITOR = """
Descompón el caso del ÚLTIMO MENSAJE en los HECHOS INFRACTORES de tráfico. Responde SOLO con JSON válido, sin markdown: {"hechos": ["...", "..."]} (máximo 2). Si es pregunta general, responde {"hechos": []}.
"""

@con_reintento_api
def descomponer_hechos(pregunta: str, historial: list) -> list:
    contexto = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in historial[-2:]
    )
    prompt = f"HISTORIAL:\n{contexto}\n\nÚLTIMO MENSAJE:\n{pregunta}"
    try:
        r = cliente_gemini.models.generate_content(
            model=MODELO_GENERACION,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=INSTRUCCION_DESCOMPOSITOR,
                max_output_tokens=200,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        datos = json.loads(re.sub(r"```(?:json)?", "", r.text or "").strip())
        return [h.strip() for h in datos.get("hechos", []) if isinstance(h, str) and h.strip()][:2]
    except Exception:
        return []

# ------------------------------------------------------------------
# 5. Cálculos y Extractor
# ------------------------------------------------------------------
@st.cache_resource
def obtener_datos_calculo():
    return cargar_datos()

DATOS_CALCULO = obtener_datos_calculo()

INSTRUCCION_EXTRACTOR = """
Extractor de datos para calculadoras de transporte y tráfico. Devuelve SOLO un JSON válido:
{
 "calculo": "tiempos" | "masas" | "dimensiones" | "cargas" | null,
 "tiempos": null, "masas": null, "dimensiones": null, "cargas": null
}
"""

@con_reintento_api
def extraer_calculo(pregunta: str, historial: list):
    contexto = "\n".join(
        f"{m['role']}: {m['content'][:400]}" for m in historial[-4:]
    )
    prompt = f"HISTORIAL:\n{contexto}\n\nÚLTIMO MENSAJE:\n{pregunta}"
    try:
        r = cliente_gemini.models.generate_content(
            model=MODELO_GENERACION,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=INSTRUCCION_EXTRACTOR,
                max_output_tokens=600,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        bruto = re.sub(r"```(?:json)?", "", r.text or "").strip()
        return json.loads(bruto)
    except Exception:
        return None

INSTRUCCION_CALCULO = REGLAS_COMUNES + """
MODO CÁLCULO. Usa el bloque "RESULTADO DEL CÁLCULO" verificado por código. No inventes datos.
"""
CONFIG_MODOS["CALCULO"] = {
    "prompt": INSTRUCCION_CALCULO, "top_k": 0, "max_tokens": 1000,
    "temp": 0.1, "etiqueta": "🧮 Cálculo",
}

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
    st.session_state.calculo_reciente = False

st.session_state.ultima_interaccion = datetime.now()

with st.sidebar:
    if st.button("🗑️ Nueva conversación"):
        st.session_state.mensajes = []
        st.session_state.calculo_reciente = False
        st.rerun()
    st.caption(f"Filtros metadatos: {'ON' if USAR_FILTROS_METADATOS else 'OFF'}")
    st.caption(f"Debug: {'ON' if MODO_DEBUG else 'OFF'}")
    if not CALCULOS_OK:
        st.caption("Calculadoras: OFF")
    elif not DATOS_CALCULO:
        st.caption("Calculadoras: OFF (falta carpeta datos/)")
    else:
        st.caption("Calculadoras: ON")

for mensaje in st.session_state.mensajes:
    avatar = AVATAR_ASISTENTE if mensaje["role"] == "assistant" else None
    with st.chat_message(mensaje["role"], avatar=avatar):
        st.markdown(mensaje["content"])

pregunta_usuario = st.chat_input("Escribe tu consulta legal...")

if pregunta_usuario:
    historial_previo = list(st.session_state.mensajes)
    st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario})
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)

    with st.chat_message("assistant", avatar=AVATAR_ASISTENTE):
        contenedor_respuesta = st.empty()
        
        # 1. Router / Clasificación
        modo = clasificar_consulta(pregunta_usuario, historial_previo)
        
        # 2. Lógica de cálculo o búsqueda vectorial
        texto_calculo = ""
        resultado_calculo = None
        if CALCULOS_OK and DATOS_CALCULO and ('calculo_reciente' in st.session_state and st.session_state.calculo_reciente):
            datos_extraccion = extraer_calculo(pregunta_usuario, historial_previo)
            if datos_extraccion and datos_extraccion.get("calculo"):
                resultado_calculo = calcular(datos_extraccion, DATOS_CALCULO)
                if resultado_calculo["estado"] in ("ok", "faltan_datos", "multiples_opciones"):
                    modo = "CALCULO"
                    texto_calculo = f"\n\nRESULTADO DEL CÁLCULO:\n{resultado_calculo['texto']}"

        cfg = CONFIG_MODOS.get(modo, CONFIG_MODOS["CONSULTA_GENERAL"])
        
        # 3. Búsqueda en Pinecone si aplica
        contexto_recuperado = ""
        fragmentos_usados = []
        filtro_usado = None
        
        if modo != "CALCULO" and cfg["top_k"] > 0:
            filtro = construir_filtro(modo, pregunta_usuario)
            if modo == "MULTA_TRAFICO":
                hechos = descomponer_hechos(pregunta_usuario, historial_previo)
                if len(hechos) > 1:
                    vectores = [obtener_embedding(h) for h in hechos]
                    contexto_recuperado, fragmentos_usados, filtro_usado = buscar_contexto_multi(
                        vectores, cfg["top_k"], filtro, modo
                    )
                else:
                    vec = obtener_embedding(hechos[0] if hechos else pregunta_usuario)
                    contexto_recuperado, fragmentos_usados, filtro_usado = buscar_contexto(
                        vec, cfg["top_k"], filtro, modo
                    )
            else:
                vec = obtener_embedding(pregunta_usuario)
                contexto_recuperado, fragmentos_usados, filtro_usado = buscar_contexto(
                    vec, cfg["top_k"], filtro, modo
                )

        # 4. Generación de respuesta con Gemini
        historial_recortado = historial_previo[-VENTANA_HISTORIAL:]
        contenido_chat = "\n".join([f"{m['role']}: {m['content']}" for m in historial_recortado])
        
        prompt_final = f"""
{cfg['prompt']}

CONTEXTO RECUPERADO:
{contexto_recuperado}
{texto_calculo}

HISTORIAL DE CONVERSACIÓN:
{contenido_chat}

ÚLTIMA CONSULTA DEL USUARIO:
{pregunta_usuario}
"""
        
        try:
            @con_reintento_api
            def generar_respuesta_gemini():
                return cliente_gemini.models.generate_content(
                    model=MODELO_GENERACION,
                    contents=prompt_final,
                    config=GenerateContentConfig(
                        max_output_tokens=cfg["max_tokens"],
                        temperature=cfg["temp"],
                    ),
                )

            respuesta = generar_respuesta_gemini()
            texto_respuesta = respuesta.text or "No se ha podido generar una respuesta."
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                texto_respuesta = "⚠️ El servicio está experimentando alta demanda en este momento (error 503). Por favor, reintenta enviar tu mensaje en unos segundos."
            else:
                texto_respuesta = f"⚠️ Ocurrió un error al procesar tu solicitud: {e}"

        contenedor_respuesta.markdown(texto_respuesta)
        st.session_state.mensajes.append({"role": "assistant", "content": texto_respuesta})
