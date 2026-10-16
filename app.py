import re
import json
import time
import threading
import streamlit as st
from collections import deque
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from pinecone import Pinecone
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta
try:
    from buscador import construir_indices, buscar_lexico, fusionar
    BUSCADOR_OK = True
except ImportError:  # sin buscador.py la app sigue con la búsqueda semántica sola
    BUSCADOR_OK = False
try:
    from calculos import cargar_datos, calcular, opciones_dimension_para_prompt
    CALCULOS_OK = True
except ImportError:  # falta calculos.py junto a app.py: la app sigue funcionando sin calculadoras
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
# Configuración general
# ------------------------------------------------------------------
MODELO_EMBEDDING = "gemini-embedding-001"
MODELO_GENERACION = "gemini-3.5-flash-lite"
DIMENSION_EMBEDDING = 768
VENTANA_HISTORIAL = 8
LIMITE_INACTIVIDAD = timedelta(minutes=60)
MAX_RPM = 12  # peticiones a Gemini por minuto que se permite la app (la cuota real es 15)

# Filtros por metadatos en Pinecone.
# Déjalo en False hasta que hayas añadido los campos "fuente" (trafico/transporte),
# "codigo" (transporte) y "articulo" (tráfico, como número) a tus vectores.
# Si está en True y el filtro no devuelve nada, se reintenta sin filtro automáticamente.
USAR_FILTROS_METADATOS = False
MODO_DEBUG = True  # muestra modo detectado, filtro usado y fragmentos recuperados

# ------------------------------------------------------------------
# 0. LIMITADOR DE PETICIONES Y REINTENTOS (cuota de Gemini: 15 RPM)
#    Todas las llamadas a Gemini (generación y embeddings) pasan por aquí.
#    El contador es global: se comparte entre todas las sesiones de la app.
# ------------------------------------------------------------------
@st.cache_resource
def _estado_limitador():
    return {"lock": threading.Lock(), "llamadas": deque()}


def esperar_turno():
    est = _estado_limitador()
    while True:
        with est["lock"]:
            ahora = time.time()
            while est["llamadas"] and ahora - est["llamadas"][0] > 60:
                est["llamadas"].popleft()
            if len(est["llamadas"]) < MAX_RPM:
                est["llamadas"].append(ahora)
                return
            espera = 60 - (ahora - est["llamadas"][0]) + 0.2
        time.sleep(max(espera, 0.2))


def es_error_cuota(error) -> bool:
    texto = str(error)
    return "429" in texto or "RESOURCE_EXHAUSTED" in texto


def con_reintentos(funcion, intentos: int = 3):
    for i in range(intentos):
        esperar_turno()
        try:
            return funcion()
        except Exception as e:
            if es_error_cuota(e) and i < intentos - 1:
                time.sleep(5 * (i + 1))
                continue
            raise


def llamar_gemini(**kwargs):
    return con_reintentos(lambda: cliente_gemini.models.generate_content(**kwargs))


def generar_stream(**kwargs):
    """Streaming con limitador; reintenta solo si falla antes de emitir el primer fragmento."""
    for intento in range(3):
        esperar_turno()
        emitido = False
        try:
            for chunk in cliente_gemini.models.generate_content_stream(**kwargs):
                emitido = True
                yield chunk
            return
        except Exception as e:
            if not emitido and es_error_cuota(e) and intento < 2:
                time.sleep(5 * (intento + 1))
                continue
            raise


# ------------------------------------------------------------------
# Datos de cálculo y detección de códigos de transporte
# (se cargan aquí porque los usa la detección de fichas de transporte)
# ------------------------------------------------------------------
@st.cache_resource
def obtener_datos_calculo():
    return cargar_datos()  # None si falta la carpeta datos/ (ver preparar_datos.py)


DATOS_CALCULO = obtener_datos_calculo()

CLAVES_CODIGO = ("codigo", "código", "code", "cod")


def _buscar_codigos(nodo, dentro_de_transporte: bool, salida: set):
    if isinstance(nodo, dict):
        for k, v in nodo.items():
            k_low = str(k).lower()
            if dentro_de_transporte and k_low in CLAVES_CODIGO and isinstance(v, str) and v.strip():
                salida.add(v.strip())
            _buscar_codigos(v, dentro_de_transporte or "transporte" in k_low, salida)
    elif isinstance(nodo, list):
        for x in nodo:
            _buscar_codigos(x, dentro_de_transporte, salida)


def extraer_codigos_transporte(datos) -> list:
    """Recoge los códigos reales del baremo de transporte, sea cual sea su formato.
    Asume que, en el JSON de datos/, las fichas de transporte cuelgan de una clave que contiene
    'transporte' y que cada ficha tiene un campo 'codigo'. Ajusta CLAVES_CODIGO si se llama distinto."""
    salida = set()
    if datos:
        try:
            _buscar_codigos(datos, False, salida)
        except Exception:
            pass
    return sorted(salida)


def construir_regex_codigos(codigos: list):
    """Una sola regex con todos los códigos reales (tolera espacios variables dentro del código)."""
    ordenados = sorted({c.strip().upper() for c in codigos if c and c.strip()}, key=len, reverse=True)
    if not ordenados:
        return None
    alternativas = "|".join(r"\s+".join(re.escape(parte) for parte in c.split()) for c in ordenados)
    return re.compile(rf"(?<![A-Z0-9])(?:{alternativas})(?![A-Z0-9])", re.I)


PATRON_CODIGO_GENERICO = re.compile(r"\b[A-Z]{2}\d{2}\.\d{2}\b")  # respaldo si no hay códigos reales
CODIGOS_TRANSPORTE = extraer_codigos_transporte(DATOS_CALCULO)
RE_CODIGO_TRANSPORTE = construir_regex_codigos(CODIGOS_TRANSPORTE)

# ------------------------------------------------------------------
# 1. PLANIFICADOR: clasifica la consulta y, si es multa de tráfico, la descompone en hechos
#    (una sola llamada a Gemini en lugar de dos)
# ------------------------------------------------------------------
INSTRUCCION_PLANIFICADOR = """
Clasifica la ÚLTIMA consulta del usuario en UNA de estas categorías:

MULTA_TRAFICO: el usuario quiere saber la sanción, cuantía, puntos, precepto o infracción concreta de un hecho de TRÁFICO
  (velocidad/radar, alcoholemia, drogas, móvil, cinturón, casco, VMP/patinetes, seguro, ITV, semáforos, estacionamiento,
  conducción temeraria...). Alcohol, drogas, VMP y radar son siempre tráfico.
MULTA_TRANSPORTE: el usuario quiere la sanción o tipificación de un hecho de TRANSPORTE de mercancías/viajeros
  (tacógrafo, tiempos de conducción y descanso, autorización de transporte, ámbito territorial, tarjeta, exceso de peso
  en transporte, LOTT/ROTT, taxi, VTC, autobús, arrendador, códigos de infracción del baremo de transportes...).
CONSULTA_GENERAL: pregunta conceptual o de conocimiento sobre una norma (qué dice, cómo funciona, requisitos, plazos,
  definiciones, procedimiento, diferencias...) que NO pide rellenar una denuncia ni calcular una sanción concreta.

Usa el historial para entender mensajes cortos de seguimiento (ej. "¿y si es reincidente?" hereda el modo anterior).
Si dudas entre multa y general, elige CONSULTA_GENERAL salvo que pidan expresamente cuantía, puntos, artículo o "qué multa pone".

Además, si la categoría es MULTA_TRAFICO, descompón el caso en los HECHOS INFRACTORES que podrían denunciarse, cada uno
redactado como lo escribiría el baremo de la DGT en su campo "Hecho infringido" (lenguaje formal:
"NO SEÑALIZAR REGLAMENTARIAMENTE LA CARGA QUE SOBRESALE...", "CIRCULAR CON UN VEHÍCULO CUYA CARGA...").
- Máximo 3 hechos; el más importante primero.
- Incluye solo hechos que se deduzcan de los datos que ha dado el usuario. No añadas infracciones hipotéticas
  (p. ej. matrícula tapada, circulación nocturna) si no se han mencionado.
- Si el mensaje es una pregunta de conocimiento y no un caso concreto, "hechos": [].
En cualquier otra categoría, "hechos": [].

Responde SOLO con JSON válido, sin markdown: {"modo": "<ETIQUETA>", "hechos": ["...", "..."]}
"""

MODOS_VALIDOS = ("MULTA_TRAFICO", "MULTA_TRANSPORTE", "CONSULTA_GENERAL")


def planificar(pregunta: str, historial: list):
    """Devuelve (modo, hechos). Una sola llamada a Gemini."""
    contexto_previo = "\n".join(
        f"{m['role']}: {m['content'][:600] if m['role'] == 'user' else m['content'][:300]}"
        for m in historial[-4:]
    )
    prompt = f"HISTORIAL RECIENTE:\n{contexto_previo}\n\nÚLTIMA CONSULTA:\n{pregunta}"
    try:
        r = llamar_gemini(
            model=MODELO_GENERACION,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=INSTRUCCION_PLANIFICADOR,
                max_output_tokens=400,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        datos = json.loads(re.sub(r"```(?:json)?", "", r.text or "").strip())
        modo = str(datos.get("modo", "")).strip().upper()
        if modo not in MODOS_VALIDOS:
            modo = "CONSULTA_GENERAL"
        hechos = [h.strip() for h in datos.get("hechos", []) if isinstance(h, str) and h.strip()][:3]
        return modo, (hechos if modo == "MULTA_TRAFICO" else [])
    except Exception:
        return "CONSULTA_GENERAL", []


# ------------------------------------------------------------------
# 2. PROMPTS POR MODO
# ------------------------------------------------------------------
REGLAS_COMUNES = """
Responde siempre en español, basándote en el CONTEXTO recuperado. Si el contexto no contiene la respuesta,
dilo claramente en vez de inventar. No repitas un mismo dato en más de una línea.
"""

INSTRUCCION_MULTA_TRAFICO = REGLAS_COMUNES + """
MODO MULTA DE TRÁFICO. Eres un asistente de denuncias de tráfico.
Fuente: Baremo de la DGT (Ley de Tráfico, RGC, normativa VMP). Alcohol, drogas, VMP y radar usan las mismas fichas.
NO apliques el baremo de transportes (LOTT, ROTT, tacógrafo) en este modo.

ESTRUCTURA DE LAS FICHAS DEL CONTEXTO. Cada ficha contiene:
  Norma | Artículo | Apartado | Opción | Puntos | Calificación | Hecho infringido | Multa | Responsable | Comentario
- "Multa" trae DOS importes: el primero es la cuantía completa y el segundo la cuantía reducida (pronto pago).
- "Puntos" es el dato oficial de la ficha: cópialo tal cual (0 es un valor válido). Nunca lo modifiques ni lo deduzcas.
- "Calificación" puede incluir el artículo de la LSV que la fundamenta (ej. "Muy grave 077.e").
- Los ceros a la izquierda del artículo (002, 003) se escriben sin ellos: art. 2, art. 3.
- Si hay conflicto entre las reglas fijas de este prompt (bloques B y C) y la ficha del contexto, prevalece la ficha.

Reglas:
1. Elige la ficha cuyo "Hecho infringido" describa exactamente el hecho consultado. Si varias fichas son plausibles
   (mismo artículo con distintas opciones), NO elijas al azar: pregunta en UNA frase corta el dato que las distingue.
2. Si ninguna ficha encaja claramente, dilo y no inventes artículo, opción, importe ni puntos.
3. VMP/VPL: si faltan datos críticos (peso, fecha de comercialización, certificado, registro, etiqueta),
   PROHIBIDO dar sanción definitiva; pregunta por ellos en una sola frase corta.
4. Velocidad/radar y alcohol/drogas: si falta algún dato (valor medido, tipo de vía, límite, tipo de conductor...),
   pregunta antes de calcular.
5. El campo Comentario úsalo solo si es relevante para el caso (posible delito, compatibilidad con otros
   expedientes, aclaraciones de aplicación). Resúmelo; no lo copies entero.
6. Con la ficha identificada, responde SIEMPRE en esta lista, una línea por punto:
   - Norma y artículo / Precepto: ... (norma, art., apartado, opción)
   - Infracción: ... (hecho infringido, resumido)
   - Calificación: ...
   - Cálculo / Verificación de requisitos: ... (solo si hay cálculo o requisitos; si no, omite la línea)
   - Cuantía: ... € (cuantía reducida: ... €)
   - Puntos: ...
   - Responsable: ...
   - Comentario: ... (solo si aporta algo; si no, omite)
7. Si el caso contiene VARIAS infracciones distintas (p. ej. carga que sobresale + señalización no reglamentaria),
   presenta una lista por cada una, con una ficha del CONTEXTO que encaje exactamente con ese hecho. No añadas
   infracciones hipotéticas que el usuario no haya mencionado. Si para alguna falta un dato (p. ej. si circulaba de noche),
   pregúntalo al final en una frase corta. Si una infracción del caso no tiene ficha en el CONTEXTO, dilo en vez de inventarla.
8. Si la ficha más cercana es GENÉRICA (p. ej. "en los términos reglamentariamente previstos", "en los casos reglamentarios"),
   aplícala si el hecho encaja en su sentido general, y avisa en una frase de que el límite o requisito concreto depende del
   reglamento, que no consta en el CONTEXTO: no lo inventes ni des cifras de memoria. No digas que "no hay ficha" cuando existe
   una ficha genérica que cubre el hecho.

B) FÓRMULAS Y BAREMOS FIJOS DE VELOCIDAD Y ALCOHOLEMIA:
   - B.1 Corrección de mediciones (etilómetros y radares).
   - B.2 Tabla de excesos de velocidad.
   - B.3 Alcoholemia (tasa corregida y tramos por tipo de conductor).
   - B.4 Umbrales penales de velocidad.
   [... pega aquí el detalle de tus fórmulas y tablas B.1-B.4, si lo tienes fuera de Pinecone ...]

C) VMP Y VPL (Vehículos de Movilidad Personal y Ligeros):
   - Criterio de peso: más de 25 kg = VMP (seguro obligatorio siempre). Menos de 25 kg = VPL.
   - Requisitos técnicos VPL (<25 kg) / VMP:
     · Certificado de circulación: exigible solo para comercializados después del 22 de enero de 2024
       (si falta y no está exento: VEH 22 B-2-5A, 200/100€, no inmoviliza).
     · Inscripción en Registro de Vehículos: VEH 22 B-2-5B, 100/50€, no inmoviliza.
     · Etiqueta identificativa o placa de marcaje: VEH 22 B-2-5C, 80/40€, no inmoviliza.
   - Modificaciones técnicas:
     · Muy graves (manipulación velocidad, baterías externas, sin frenos): VEH 22 B-2-5D, 500/250€, SÍ inmoviliza.
     · Graves (neumáticos rugosos, sin alumbrado): VEH 22 B-2-5E, 200/100€, SÍ inmoviliza.
     · Leves (sin timbre, catadióptricos, pata de cabra): VEH 22 B-2-5F, 80/40€, NO inmoviliza.
   - Régimen de seguros (jerarquía estricta):
     · VPL (<25 kg): el seguro de responsabilidad civil (circulando SDA 1-5A 300/150€; sin circular SDA 1-5B 300/150€)
       SOLO es denunciable si dispone de Certificado, Registro y Etiqueta. Si falta alguno de los tres, el vehículo
       está exento de seguro.
     · VMP (>25 kg): el seguro es exigible SIEMPRE (circulando SOA 2-1-5N, 800/400€; sin circular SOA 1-5O, 610/305€).
       SÍ procede inmovilizar.
"""

INSTRUCCION_MULTA_TRANSPORTE = REGLAS_COMUNES + """
MODO MULTA DE TRANSPORTE. Eres un asistente de denuncias de transporte terrestre.
Fuente: Baremo Sancionador de Transportes (LOTT, ROTT y normativa asociada).
NO apliques el baremo de tráfico (RGC, puntos del carné, VMP) en este modo.

ESTRUCTURA DE LAS FICHAS DEL CONTEXTO. Cada ficha tiene:
  Código (ej. SG01.01) | Concepto | Infracción (= norma infringida) | Sanción (= norma sancionadora) | Cuantía
El campo Cuantía puede incluir, además del importe: "Inmovilización" (con su artículo entre paréntesis) y el
responsable (ej. "Arrendador"). Extráelos tal cual aparecen; no añadas datos que la ficha no contenga.

Reglas:
1. Elige la ficha cuyo CONCEPTO describa exactamente el hecho consultado. Si varias fichas son plausibles
   (códigos parecidos), NO elijas al azar: pregunta en UNA frase corta el dato que las distingue.
2. Si ninguna ficha del contexto encaja claramente, dilo y no inventes código, artículo ni importe.
3. Si el campo Infracción dice "Según Normativa CCAA" (u otra remisión), indícalo tal cual y avisa de que
   el precepto concreto depende de esa normativa.
4. No menciones puntos del carné ni cuantía reducida salvo que la ficha lo diga.
5. Con la ficha identificada, responde SIEMPRE en esta lista, una línea por punto:
   - Código: ...
   - Concepto: ...
   - Norma infringida: ...
   - Norma sancionadora: ...
   - Cuantía: ...
   - Inmovilización: Sí (art. ...) / No consta
   - Responsable: ... (solo si la ficha lo indica; si no, "No consta")
   - Comentario: ... (solo si aporta algo útil; si no, omítelo)
"""

INSTRUCCION_CONSULTA_GENERAL = REGLAS_COMUNES + """
MODO CONSULTA GENERAL. El usuario quiere ENTENDER una norma, no tramitar una denuncia.
- Explica de forma clara y completa, en prosa o con listas cortas si ayudan. Cita la norma y los artículos relevantes.
- NO uses el formato de boletín de denuncia (Precepto/Infracción/Cuantía/Puntos...) ni inventes "0 puntos".
- Si la pregunta toca sanciones, menciónalas de forma orientativa y ofrece al final que, si quiere,
  puedes calcular la multa concreta con los datos del caso.
- Si la consulta es ambigua entre tráfico y transporte, explica brevemente ambos enfoques.
"""

CONFIG_MODOS = {
    "MULTA_TRAFICO": {
        "prompt": INSTRUCCION_MULTA_TRAFICO, "top_k": 3, "max_tokens": 900,
        "temp": 0.2, "etiqueta": "🚦 Multa de tráfico",
    },
    "MULTA_TRANSPORTE": {
        "prompt": INSTRUCCION_MULTA_TRANSPORTE, "top_k": 4, "max_tokens": 900,
        "temp": 0.2, "etiqueta": "🚛 Multa de transporte",
    },
    "CONSULTA_GENERAL": {
        "prompt": INSTRUCCION_CONSULTA_GENERAL, "top_k": 5, "max_tokens": 1200,
        "temp": 0.4, "etiqueta": "📖 Consulta general",
    },
}

FUENTE_POR_MODO = {
    "MULTA_TRAFICO": "trafico",
    "MULTA_TRANSPORTE": "transporte",
    "CONSULTA_GENERAL": None,
}

# Selector de búsqueda (lo elige el usuario encima del chat)
OPCION_AUTO = "🤖 Automático"
OPCIONES_MODO = {
    OPCION_AUTO: None,
    "🚛 Transporte": "MULTA_TRANSPORTE",
    "🚦 Resto de infracciones": "MULTA_TRAFICO",
    "📖 Consulta general": "CONSULTA_GENERAL",
}

# ------------------------------------------------------------------
# 3. Detección por regex (código de infracción de transporte, nº de artículo)
# ------------------------------------------------------------------
def buscar_codigo_transporte(texto: str):
    """Busca un código del baremo de transporte: primero entre los códigos reales cargados de datos/,
    y como respaldo con el patrón genérico (2 letras + 2 dígitos + punto + 2 dígitos)."""
    if RE_CODIGO_TRANSPORTE:
        m = RE_CODIGO_TRANSPORTE.search(texto)
        if m:
            return re.sub(r"\s+", " ", m.group(0).upper())
    m = PATRON_CODIGO_GENERICO.search(texto.upper())
    return m.group(0) if m else None


def detectar_codigo(texto: str):
    return buscar_codigo_transporte(texto)


CLAVES_TRANSPORTE = (
    "BAREMO SANCIONADOR DE TRANSPORTES", "LOTT", "ROTT", "VTC",
    "ARRENDAMIENTO DE VEHÍCULOS CON CONDUCTOR", "ARRENDAMIENTO DE VEHICULOS CON CONDUCTOR",
    "DECRETO-LEY", "DECRETO LEY",
)


def es_ficha_transporte(texto: str) -> bool:
    t = texto.upper()
    return any(k in t for k in CLAVES_TRANSPORTE) or bool(buscar_codigo_transporte(texto))


def detectar_articulo(texto: str):
    m = re.search(r"\bart(?:[íi]culo|\.)?\s*(\d{1,3})", texto.lower())
    return int(m.group(1)) if m else None


def construir_filtro(modo: str, pregunta: str):
    """Devuelve el filtro de Pinecone para este caso (o None)."""
    if not USAR_FILTROS_METADATOS:
        return None
    filtro = {}
    fuente = FUENTE_POR_MODO.get(modo)
    if fuente:
        filtro["fuente"] = {"$eq": fuente}
    if modo == "MULTA_TRANSPORTE":
        codigo = detectar_codigo(pregunta)
        if codigo:
            filtro["codigo"] = {"$eq": codigo}
    elif modo == "MULTA_TRAFICO":
        articulo = detectar_articulo(pregunta)
        if articulo is not None:
            filtro["articulo"] = {"$eq": articulo}
    return filtro or None


# ------------------------------------------------------------------
# 4. Funciones de búsqueda
# ------------------------------------------------------------------
@st.cache_resource
def _cache_embeddings():
    return {}


def obtener_embeddings(textos: list):
    """Embeddings de varios textos en UNA sola petición. Los ya calculados se reutilizan."""
    cache = _cache_embeddings()
    if len(cache) > 500:
        cache.clear()
    faltan = list(dict.fromkeys(t for t in textos if t not in cache))
    if faltan:
        r = con_reintentos(
            lambda: cliente_gemini.models.embed_content(
                model=MODELO_EMBEDDING,
                contents=faltan,
                config={"output_dimensionality": DIMENSION_EMBEDDING},
            )
        )
        for texto, emb in zip(faltan, r.embeddings):
            cache[texto] = emb.values
    return [cache[t] for t in textos]


def obtener_embedding(texto: str):
    return obtener_embeddings([texto])[0]


def _consultar(vector, top_k, filtro):
    kwargs = dict(vector=vector, top_k=top_k, include_metadata=True)
    if filtro:
        kwargs["filter"] = filtro
    return indice_pinecone.query(**kwargs)


SOBREMUESTREO = 8  # sin metadatos, se piden top_k * 8 resultados y se filtra por texto


def buscar_contexto(vector_consulta, top_k: int, filtro: dict | None = None, modo: str | None = None):
    """Devuelve (contexto_texto, lista_fragmentos, descripcion_filtro_usado)."""
    resultados = _consultar(vector_consulta, top_k, filtro) if filtro else None
    descripcion = filtro

    if resultados is not None and resultados.matches:
        fragmentos = [
            m.metadata.get("texto", "")
            for m in resultados.matches
            if m.metadata and m.metadata.get("texto", "")
        ]
    else:
        # Sin metadatos (o el filtro no devolvió nada): sobremuestrear y filtrar por texto
        pedir = top_k * SOBREMUESTREO if modo in ("MULTA_TRAFICO", "MULTA_TRANSPORTE") else top_k
        resultados = _consultar(vector_consulta, pedir, None)
        fragmentos = [
            m.metadata.get("texto", "")
            for m in resultados.matches
            if m.metadata and m.metadata.get("texto", "")
        ]
        descripcion = None
        if modo == "MULTA_TRANSPORTE":
            # Primero las fichas de transporte; si faltan, se completa con el resto para no dejar el contexto vacío
            filtrados = [f for f in fragmentos if es_ficha_transporte(f)]
            resto = [f for f in fragmentos if f not in filtrados]
            fragmentos = (filtrados + resto)[:top_k]
            descripcion = "postfiltro texto: transporte"
        elif modo == "MULTA_TRAFICO":
            fragmentos = [f for f in fragmentos if not es_ficha_transporte(f)][:top_k]
            descripcion = "postfiltro texto: tráfico"
        else:
            fragmentos = fragmentos[:top_k]

    return "\n\n---\n\n".join(fragmentos), fragmentos, descripcion


def buscar_contexto_multi(vectores: list, top_k: int, filtro: dict | None = None, modo: str | None = None,
                          limite_total: int | None = None):
    """Busca con varios vectores y une los resultados sin duplicados, repartiendo por turnos
    (el 1.º de cada consulta, luego el 2.º de cada una...) para que ningún hecho se quede sin ficha."""
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
# 4b. Descomposición de casos de tráfico en hechos infractores
#     En modo automático lo hace el planificador (una sola llamada).
#     Esta función se usa cuando el usuario fuerza "Resto de infracciones".
# ------------------------------------------------------------------
INSTRUCCION_DESCOMPOSITOR = """
Eres un asistente de denuncias de tráfico en España. Descompón el caso del ÚLTIMO MENSAJE en los HECHOS INFRACTORES
que podrían denunciarse, cada uno redactado como lo escribiría el baremo de la DGT en su campo "Hecho infringido"
(lenguaje formal: "NO SEÑALIZAR REGLAMENTARIAMENTE LA CARGA QUE SOBRESALE...", "CIRCULAR CON UN VEHÍCULO CUYA CARGA...").
Responde SOLO con JSON válido, sin markdown: {"hechos": ["...", "..."]}  (máximo 3; el más importante primero).
- Incluye solo hechos que se deduzcan de los datos que ha dado el usuario. No añadas infracciones hipotéticas
  (p. ej. matrícula tapada, circulación nocturna) si no se han mencionado.
- Un caso puede contener varios hechos distintos (p. ej. carga que sobresale más de lo permitido Y señalización no reglamentaria).
- Si el mensaje es una pregunta de conocimiento y no un caso concreto, responde {"hechos": []}.
"""


def descomponer_hechos(pregunta: str, historial: list) -> list:
    contexto = "\n".join(
        f"{m['role']}: {m['content'][:600] if m['role'] == 'user' else m['content'][:200]}" for m in historial[-4:]
    )
    prompt = f"HISTORIAL:\n{contexto}\n\nÚLTIMO MENSAJE:\n{pregunta}"
    try:
        r = llamar_gemini(
            model=MODELO_GENERACION,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=INSTRUCCION_DESCOMPOSITOR,
                max_output_tokens=400,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        datos = json.loads(re.sub(r"```(?:json)?", "", r.text or "").strip())
        return [h.strip() for h in datos.get("hechos", []) if isinstance(h, str) and h.strip()][:3]
    except Exception:
        return []


# ------------------------------------------------------------------
# 5. CÁLCULOS (tiempos de conducción, masas, dimensiones)
#    La IA solo EXTRAE los datos del relato; Python hace las cuentas y elige la ficha.
# ------------------------------------------------------------------
@st.cache_resource
def obtener_indices_lexicos():
    if not (BUSCADOR_OK and DATOS_CALCULO):
        return None
    return construir_indices(DATOS_CALCULO)


INDICES_LEXICOS = obtener_indices_lexicos()

INSTRUCCION_EXTRACTOR = """
Eres un extractor de datos para calculadoras de denuncias de transporte y tráfico. Lees el HISTORIAL y el ÚLTIMO MENSAJE
y devuelves SOLO un JSON válido (sin markdown ni texto extra), combinando los datos de TODA la conversación:
{
 "calculo": "tiempos" | "masas" | "dimensiones" | null,
 "tiempos": {"eventos": [{"tipo": "conduccion|pausa|descanso", "minutos": entero}], "transporte": "mercancias|viajeros|null", "ampliacion_10h": true|false|null},
 "masas": {"peso_medido_kg": número|null, "mma_kg": número|null, "margen_bascula": {"valor": número, "unidad": "kg|%"}|null, "peso_ya_neto": true|null, "categoria": "N1|N2|N3|null", "mma_vehiculo_kg": número|null, "regimen": "publico|privado|null", "base": "mma|eje|null"},
 "dimensiones": {"magnitud": "longitud|anchura|altura|null", "medida_m": número|null, "tipo_vehiculo": clave|null, "modificadores": [claves], "causa_exceso": "carga_indivisible|caracteristica_tecnica|otra|null", "autorizacion": "sin_autorizacion|incumple_condiciones|con_autorizacion_cumple|null"}
}
Reglas:
- "calculo" es null si el último mensaje es una pregunta general o no hay un caso concreto con cifras. Si el último mensaje solo
  completa datos que faltaban de un cálculo anterior del historial, mantén ese cálculo.
- Nunca inventes datos: lo que no se haya dicho va a null (o [] en eventos).
- TIEMPOS: convierte todo a minutos (4,5 h = 270; 13 h 30 = 810). Respeta el orden del relato. "descanso" = 9 horas o más; "pausa" = interrupción más corta.
  "transporte": mercancias (camión, carga) o viajeros (autobús, autocar). "ampliacion_10h": true solo si se dice que puede ampliar o ha ampliado a 10 h; false si se dice que rige el límite de 9 h.
- MASAS: pesos en kg (1 t = 1000 kg). "mma_kg" es la MMA de la ficha técnica o de la autorización especial (la del eje si el exceso es por eje). "regimen": publico (transporte público, con autorización de transporte) o privado (privado complementario).
  "base": "eje" si el exceso es de un eje, "mma" si es de la masa total. "peso_ya_neto": true si dicen que ya está descontado el margen o tolerancia de la báscula.
- DIMENSIONES: medidas en metros. "tipo_vehiculo" debe ser UNA de estas claves (o null si no está claro):
{{OPCIONES}}
  "causa_exceso": carga_indivisible, caracteristica_tecnica u otra (p. ej. carga divisible). "autorizacion": estado de la autorización especial o complementaria.
"""
INSTRUCCION_EXTRACTOR = INSTRUCCION_EXTRACTOR.replace("{{OPCIONES}}", opciones_dimension_para_prompt())

RE_CALCULO = re.compile(
    r"conduc|descans|paus|horas|minutos|tac[óo]graf|peso|pesa|kg|kilo|tonelad|\bt\b|mma|\beje|altur|\balto|anch|larg|longitud|metros|\bm\b",
    re.I,
)


def parece_calculo(pregunta: str, calculo_reciente: bool) -> bool:
    tiene_cifras = bool(re.search(r"\d", pregunta))
    if calculo_reciente:
        # el mensaje puede completar o variar el cálculo anterior ("público", "sin autorización"...),
        # pero un mensaje largo y sin cifras es casi seguro otra consulta
        return tiene_cifras or len(pregunta.split()) <= 12
    return tiene_cifras and bool(RE_CALCULO.search(pregunta))


def extraer_calculo(pregunta: str, historial: list):
    contexto = "\n".join(
        f"{m['role']}: {m['content'][:1000] if m['role'] == 'user' else m['content'][:300]}" for m in historial[-6:]
    )
    prompt = f"HISTORIAL:\n{contexto}\n\nÚLTIMO MENSAJE:\n{pregunta}"
    try:
        r = llamar_gemini(
            model=MODELO_GENERACION,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=INSTRUCCION_EXTRACTOR,
                max_output_tokens=1500,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        bruto = re.sub(r"```(?:json)?", "", r.text or "").strip()
        return json.loads(bruto)
    except Exception:
        return None


INSTRUCCION_CALCULO = REGLAS_COMUNES + """
MODO CÁLCULO. Recibes un bloque "RESULTADO DEL CÁLCULO" ya verificado por código.
Reglas:
1. NO recalcules ni cambies cifras. No inventes fichas, artículos, importes, puntos ni inmovilizaciones: usa solo lo que está en el bloque.
2. Si el bloque contiene "FALTAN DATOS": pregunta SOLO por esos datos, en una o dos frases cortas, y NO des ninguna sanción.
3. Empieza con una línea "Cálculo:" que resuma las cifras clave del bloque.
4. Fichas de TRANSPORTE: preséntalas con esta lista, una línea por punto:
   - Código: ...
   - Concepto: ...
   - Norma infringida: ...
   - Norma sancionadora: ...
   - Cuantía: ... (solo el importe)
   - Inmovilización: Sí (art. ...) / No consta   (sácalo de "Cuantía y medidas")
   - Responsable: ... (solo si la ficha lo indica; si no, "No consta")
   No menciones puntos del carné en transporte.
5. Fichas de TRÁFICO: preséntalas tal cual vienen (Norma y artículo / Infracción resumida / Calificación / Cuantía / Puntos / Responsable / Comentario resumido).
6. Si hay varias alternativas porque falta un dato, preséntalas todas de forma breve y termina preguntando SOLO lo que indique "PREGUNTAS PENDIENTES".
7. Si hay AVISOS que afecten a la decisión (porcentaje deducido, campos cortados, cercanía al umbral, criterio dudoso), inclúyelos al final en una línea.
"""
CONFIG_MODOS["CALCULO"] = {
    "prompt": INSTRUCCION_CALCULO, "top_k": 0, "max_tokens": 1800,
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
        st.session_state.selector_modo = OPCION_AUTO  # se asigna antes de crear el radio, por eso es válido
        st.rerun()
    st.caption(f"Filtros por metadatos: {'ON' if USAR_FILTROS_METADATOS else 'OFF'}")
    st.caption(f"Debug: {'ON' if MODO_DEBUG else 'OFF'}")
    st.caption(f"Límite de peticiones a Gemini: {MAX_RPM}/min")
    if not CALCULOS_OK:
        st.caption("Calculadoras: OFF (falta calculos.py junto a app.py)")
    elif not DATOS_CALCULO:
        st.caption("Calculadoras: OFF (falta la carpeta datos/ con los 2 JSON)")
    else:
        st.caption("Calculadoras: ON")
    if MODO_DEBUG:
        st.caption(f"Códigos de transporte cargados: {len(CODIGOS_TRANSPORTE)}")

# Selector de dónde buscar
seleccion = st.radio(
    "Buscar en",
    list(OPCIONES_MODO),
    horizontal=True,
    key="selector_modo",
)
modo_forzado = OPCIONES_MODO[seleccion]

if modo_forzado:
    st.info(
        f"Búsqueda limitada a **{seleccion}**. "
        "Cambia a «Automático» para que el asistente decida."
    )

for mensaje in st.session_state.mensajes:
    avatar = AVATAR_ASISTENTE if mensaje["role"] == "assistant" else None
    with st.chat_message(mensaje["role"], avatar=avatar):
        st.markdown(mensaje["content"])

pregunta_usuario = st.chat_input("Escribe tu consulta legal...")

if pregunta_usuario:
    historial_previo = list(st.session_state.mensajes)  # antes de añadir la pregunta actual
    st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario})
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)

    with st.chat_message("assistant", avatar=AVATAR_ASISTENTE):
        response_placeholder = st.empty()
        texto_acumulado = ""

        try:
            # 0. ¿Es un cálculo (tiempos, masas, dimensiones)? La IA extrae, Python calcula.
            #    En "Consulta general" forzada no se usan las calculadoras.
            resultado_calculo = {"estado": "sin_calculo", "texto": "", "etiqueta": ""}
            extraccion = None
            if (
                DATOS_CALCULO
                and modo_forzado != "CONSULTA_GENERAL"
                and parece_calculo(pregunta_usuario, st.session_state.get("calculo_reciente", False))
            ):
                extraccion = extraer_calculo(pregunta_usuario, historial_previo)
                if extraccion:
                    resultado_calculo = calcular(extraccion, DATOS_CALCULO)

            if resultado_calculo["estado"] != "sin_calculo":
                modo = "CALCULO"
                cfg = CONFIG_MODOS[modo]
                st.session_state.calculo_reciente = True
                texto_calc = resultado_calculo["texto"]
                st.caption(resultado_calculo["etiqueta"])
                if MODO_DEBUG:
                    with st.expander("🔍 Debug · cálculo"):
                        st.markdown("**Datos extraídos por la IA**")
                        st.json(extraccion)
                        st.markdown("**Resultado del cálculo (lo que recibe el modelo)**")
                        st.text(texto_calc)
                prompt_actual = f"{texto_calc}\n\nPREGUNTA:\n{pregunta_usuario}"
            else:
                st.session_state.calculo_reciente = False

                # 1. Modo: el elegido por el usuario o, en automático, el que decida el planificador
                hechos = []
                if modo_forzado:
                    modo = modo_forzado
                    if modo == "MULTA_TRAFICO":
                        hechos = descomponer_hechos(pregunta_usuario, historial_previo)
                elif detectar_codigo(pregunta_usuario):
                    modo = "MULTA_TRANSPORTE"  # con un código de infracción de transporte es transporte seguro
                else:
                    modo, hechos = planificar(pregunta_usuario, historial_previo)
                cfg = CONFIG_MODOS[modo]
                st.caption(cfg["etiqueta"] + (" · elegido por ti" if modo_forzado else ""))

                # 2. Recuperar contexto según el modo (en multa de tráfico: un hecho infractor por consulta)
                consultas = hechos + [pregunta_usuario] if hechos else [pregunta_usuario]
                vectores = obtener_embeddings(consultas)  # una sola petición para todas las consultas
                filtro = construir_filtro(modo, pregunta_usuario)
                contexto, fragmentos, filtro_usado = buscar_contexto_multi(
                    vectores, top_k=cfg["top_k"], filtro=filtro, modo=modo,
                    limite_total=cfg["top_k"] * 2 if len(consultas) > 1 else None,
                )
                if INDICES_LEXICOS and modo in ("MULTA_TRAFICO", "MULTA_TRANSPORTE"):
                    # Búsqueda por palabras clave (local, no gasta cuota): rescata fichas cortas o genéricas
                    # que el embedding no encuentra
                    listas_lex = buscar_lexico(INDICES_LEXICOS, DATOS_CALCULO, modo, consultas, k=3)
                    fragmentos = fusionar(fragmentos, listas_lex, cfg["top_k"] * 2 + 2)
                    contexto = "\n\n---\n\n".join(fragmentos)
                if MODO_DEBUG:
                    with st.expander(f"🔍 Debug · modo={modo} · filtro={filtro_usado} · {len(fragmentos)} fragmentos"):
                        if hechos:
                            st.markdown("**Hechos infractores detectados:** " + " | ".join(hechos))
                        for i, f in enumerate(fragmentos, 1):
                            st.markdown(f"**Fragmento {i}**")
                            st.text(f[:700])
                prompt_actual = f"CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta_usuario}"

            # 1b. Historial conversacional
            mensajes_previos = historial_previo[-VENTANA_HISTORIAL:]
            contents_conversacion = [
                {
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [{"text": m["content"]}],
                }
                for m in mensajes_previos
            ]
            contents_conversacion.append({"role": "user", "parts": [{"text": prompt_actual}]})

            # 2. Config según el modo
            configuracion_ia = GenerateContentConfig(
                system_instruction=cfg["prompt"],
                max_output_tokens=cfg["max_tokens"],
                temperature=cfg["temp"],
            )

            # 3. Streaming (pasa por el limitador de peticiones)
            for chunk in generar_stream(
                model=MODELO_GENERACION,
                contents=contents_conversacion,
                config=configuracion_ia,
            ):
                if chunk.text:
                    texto_acumulado += chunk.text
                    response_placeholder.markdown(texto_acumulado)

        except Exception as error:
            texto_acumulado = f"Consulta pausada por saturación en la red. Por favor, reintenta en un momento. ({error})"
            response_placeholder.markdown(texto_acumulado)

    st.session_state.mensajes.append({"role": "assistant", "content": texto_acumulado})
