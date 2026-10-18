import re
import json
import time
import threading
import streamlit as st
from collections import deque, Counter
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
MODELO_ANALISIS = MODELO_GENERACION  # clasificar + extraer datos; puede ser otro modelo si quieres repartir cuota
DIMENSION_EMBEDDING = 768
VENTANA_HISTORIAL = 8
LIMITE_INACTIVIDAD = timedelta(minutes=60)

# Cuota de Gemini: los límites son POR MODELO (compruébalos en Google AI Studio > Rate limits).
# Cada modelo tiene su propio contador aquí, así los embeddings ya no gastan el cupo de generación.
MAX_RPM = 12               # peticiones/min a modelos de generación (la cuota real es 15)
MAX_RPM_EMBEDDING = 60     # peticiones/min al modelo de embeddings (cupo propio y mucho mayor)
LIMITE_RPM = {MODELO_GENERACION: MAX_RPM, MODELO_EMBEDDING: MAX_RPM_EMBEDDING}
LIMITE_RPM.setdefault(MODELO_ANALISIS, MAX_RPM)

# En "Resto de infracciones" forzado, separar el caso en hechos infractores cuesta 1 llamada extra.
# En automático NO cuesta nada, porque lo hace el mismo análisis que clasifica.
DESCOMPONER_EN_MODO_FORZADO = False

# Filtros por metadatos en Pinecone.
# Déjalo en False hasta que hayas añadido los campos "fuente" (trafico/transporte),
# "codigo" (transporte) y "articulo" (tráfico, como número) a tus vectores.
USAR_FILTROS_METADATOS = False
MODO_DEBUG = True  # muestra modo detectado, filtro usado, fragmentos recuperados y nº de llamadas

# ------------------------------------------------------------------
# 0. LIMITADOR DE PETICIONES Y REINTENTOS
#    Todas las llamadas a Gemini (generación y embeddings) pasan por aquí.
#    El contador es global (se comparte entre todas las sesiones) y va por modelo.
# ------------------------------------------------------------------
LLAMADAS_CONSULTA = []  # se vacía en cada ejecución del script (una por mensaje): sirve para el debug


@st.cache_resource
def _estado_limitador():
    return {"lock": threading.Lock(), "llamadas": {}}


def esperar_turno(modelo: str):
    est = _estado_limitador()
    limite = LIMITE_RPM.get(modelo, MAX_RPM)
    while True:
        with est["lock"]:
            cola = est["llamadas"].setdefault(modelo, deque())
            ahora = time.time()
            while cola and ahora - cola[0] > 60:
                cola.popleft()
            if len(cola) < limite:
                cola.append(ahora)
                LLAMADAS_CONSULTA.append(modelo)
                return
            espera = 60 - (ahora - cola[0]) + 0.2
        time.sleep(max(espera, 0.2))


def es_error_cuota(error) -> bool:
    texto = str(error)
    return "429" in texto or "RESOURCE_EXHAUSTED" in texto


def con_reintentos(funcion, modelo: str, intentos: int = 3):
    for i in range(intentos):
        esperar_turno(modelo)
        try:
            return funcion()
        except Exception as e:
            if es_error_cuota(e) and i < intentos - 1:
                time.sleep(5 * (i + 1))
                continue
            raise


def llamar_gemini(**kwargs):
    return con_reintentos(lambda: cliente_gemini.models.generate_content(**kwargs), kwargs["model"])


def generar_stream(**kwargs):
    """Streaming con limitador; reintenta solo si falla antes de emitir el primer fragmento."""
    for intento in range(3):
        esperar_turno(kwargs["model"])
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


def _leer_referencia_inmovilizacion():
    try:
        return (Path(__file__).parent / "datos" / "ref_inmovilizacion.txt").read_text(encoding="utf-8")
    except Exception:
        return ""


REFERENCIA_INMOVILIZACION = _leer_referencia_inmovilizacion()
def _leer_referencia_penal():
    try:
        return (Path(__file__).parent / "datos" / "ref_penal.txt").read_text(encoding="utf-8")
    except Exception:
        return ""


REFERENCIA_PENAL = _leer_referencia_penal()
RE_PENAL = re.compile(
    r"km/?h|velocid|radar|cinem[oó]metro|alcohol|tasa|mg/?l|etil[oó]metro|droga|temerari|kamikaze|negativ|soplar|delito|c[oó]digo penal|\b38[0-3]\b|\b379\b",
    re.I,
)
RE_CONSECUENCIA = re.compile(r"inmoviliz|retirad|dep[óo]sito|gr[úu]a|gastos", re.I)

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
    if isinstance(datos, dict) and isinstance(datos.get("baremo"), dict) and datos["baremo"].get("fichas"):
        return sorted(datos["baremo"]["fichas"].keys())  # formato generado por preparar_datos.py
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
# 1. ANÁLISIS: una sola llamada que clasifica la consulta, descompone el caso en hechos
#    infractores y, si hace falta, extrae los datos para las calculadoras.
# ------------------------------------------------------------------
PLAN_BASE = """
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
Las preguntas sobre las CONSECUENCIAS de un caso concreto (¿procede inmovilizar?, ¿cuántos puntos?, ¿retirada del vehículo?)
son de multa, no CONSULTA_GENERAL.

Además, si la categoría es MULTA_TRAFICO, descompón el caso en los HECHOS INFRACTORES que podrían denunciarse, cada uno
redactado como lo escribiría el baremo de la DGT en su campo "Hecho infringido" (lenguaje formal:
"NO SEÑALIZAR REGLAMENTARIAMENTE LA CARGA QUE SOBRESALE...", "CIRCULAR CON UN VEHÍCULO CUYA CARGA...").
- Máximo 3 hechos; el más importante primero.
- Incluye solo hechos que se deduzcan de los datos que ha dado el usuario. No añadas infracciones hipotéticas
  (p. ej. matrícula tapada, circulación nocturna) si no se han mencionado.
- Si el mensaje es una pregunta de conocimiento y no un caso concreto, "hechos": [].
En cualquier otra categoría, "hechos": [].
"""

INSTRUCCION_PLANIFICADOR = PLAN_BASE + """
Responde SOLO con JSON válido, sin markdown: {"modo": "<ETIQUETA>", "hechos": ["...", "..."]}
"""

EXTRACCION_REGLAS = """
ADEMÁS, extrae los datos para las calculadoras (tiempos de conducción, masas, dimensiones, cargas que sobresalen),
combinando los datos de TODA la conversación. Estructura:
{
 "calculo": "tiempos" | "masas" | "dimensiones" | "cargas" | "velocidad" | null,
 "tiempos": {"eventos": [{"tipo": "conduccion|pausa|descanso", "minutos": entero}], "transporte": "mercancias|viajeros|null", "ampliacion_10h": true|false|null},
 "masas": {"peso_medido_kg": número|null, "mma_kg": número|null, "margen_bascula": {"valor": número, "unidad": "kg|%"}|null, "peso_ya_neto": true|null, "categoria": "N1|N2|N3|null", "mma_vehiculo_kg": número|null, "regimen": "publico|privado|null", "base": "mma|eje|null"},
 "dimensiones": {"magnitud": "longitud|anchura|altura|null", "medida_m": número|null, "tipo_vehiculo": clave|null, "modificadores": [claves], "causa_exceso": "carga_indivisible|caracteristica_tecnica|otra|null", "autorizacion": "sin_autorizacion|incumple_condiciones|con_autorizacion_cumple|null"},
 "velocidad": {"velocidad_medida_kmh": número|null, "limite_kmh": número|null, "via": "urbana|interurbana|travesia|null", "velocidad_corregida_kmh": número|null, "cinemometro": "fijo|movil|tramo|null", "radar_nuevo": true|null, "limite_origen": "senalizado|generico|null", "acceso_ciudad": true|null},
 "cargas": {"vehiculo_exclusivo_mercancias": true|false|null, "longitud_vehiculo_m": número|null, "ancho_vehiculo_m": número|null, "ancho_total_m": número|null, "direccion": "detras|delante|lateral|null", "sobresale_m": número|null, "indivisible": true|false|null, "tipo_carga": "longitud_indivisible|otra|null", "senal_v20": true|false|null, "luz": true|false|null, "visibilidad_reducida": true|false|null, "transporte_especial": true|false|null}
}
Reglas de extracción:
- "calculo" es null salvo que el último mensaje describa un EXCESO o un caso que se resuelva con estas calculadoras. Un dato que solo
  describe el vehículo (p. ej. "camión de 7.500 kg", "furgoneta de 3 m de largo") NO es un cálculo. Si el último mensaje solo completa
  datos que faltaban de un cálculo anterior del historial, mantén ese cálculo. En cualquier otro caso, "calculo": null.
- Nunca inventes datos: lo que no se haya dicho va a null (o [] en eventos).
- TIEMPOS: convierte todo a minutos (4,5 h = 270; 13 h 30 = 810). Respeta el orden del relato. "descanso" = 9 horas o más; "pausa" = interrupción más corta.
  "transporte": mercancias (camión, carga) o viajeros (autobús, autocar). "ampliacion_10h": true solo si se dice que puede ampliar o ha ampliado a 10 h; false si se dice que rige el límite de 9 h.
- MASAS: pesos en kg (1 t = 1000 kg). "mma_kg" es la MMA de la ficha técnica o de la autorización especial (la del eje si el exceso es por eje). "regimen": publico (transporte público, con autorización de transporte) o privado (privado complementario).
  "base": "eje" si el exceso es de un eje, "mma" si es de la masa total. "peso_ya_neto": true si dicen que ya está descontado el margen o tolerancia de la báscula.
- DIMENSIONES: medidas en metros. "tipo_vehiculo" debe ser UNA de estas claves (o null si no está claro):
{{OPCIONES}}
  "causa_exceso": carga_indivisible, caracteristica_tecnica u otra (p. ej. carga divisible). "autorizacion": estado de la autorización especial o complementaria.
- VELOCIDAD: úsala solo si se da una velocidad medida Y el límite de la vía. "via": urbana, interurbana o travesia (tramo de carretera que cruza un núcleo urbano); null si no se dice.
  "velocidad_corregida_kmh" solo si dicen que ya se ha restado el margen del cinemómetro.
  "cinemometro": fijo (radar fijo), movil (radar móvil, instalado en vehículo o de mano) o tramo (velocidad media entre dos puntos); null si no se dice.
  "radar_nuevo": true solo si dicen que el radar es nuevo o está en su primer año de uso tras la verificación; en cualquier otro caso null.
  "limite_origen": "senalizado" si dicen que el límite lo fija una señal (vertical, marcas viales, panel) o es una limitación específica de un vehículo o de un conductor;
  "generico" si dicen que es el límite genérico de la vía (sin señal específica); null si no lo dicen.
  "acceso_ciudad": true solo si dicen que es un tramo de autopista o autovía interurbana de acceso a una ciudad o una circunvalación con límite inferior a 100 km/h; en otro caso null.
- CARGAS vs DIMENSIONES: una carga que SOBRESALE del vehículo (tabla de surf, barra, viga, bultos fuera de su proyección) es "cargas", NUNCA "dimensiones".
  "dimensiones" es solo para el largo, ancho o alto TOTAL del vehículo o conjunto (con su carga). Si no sabes si una medida es la longitud del vehículo o lo que sobresale, no la pongas en "dimensiones".
- CARGAS (art. 15 RGC): "vehiculo_exclusivo_mercancias": true para camión o vehículo de carga; false para turismo y vehículos mixtos o de uso particular; null si no queda claro.
  "indivisible": true si la carga no puede dividirse (tabla de surf, barra, viga, tubo), false si son bultos divisibles. "tipo_carga": longitud_indivisible (vigas, postes, tubos) u otra.
  "senal_v20": true solo si lleva el panel V-20 reglamentario; false si no lleva señal o lleva otra cosa (pañuelo, trapo, cinta). "luz": true si lleva luz roja detrás o blanca delante.
  "visibilidad_reducida": true si circula entre la puesta y la salida del sol o con niebla o lluvia intensa; false si es de día con buena visibilidad.
"""
INSTRUCCION_ANALISIS = (
    PLAN_BASE
    + EXTRACCION_REGLAS.replace("{{OPCIONES}}", opciones_dimension_para_prompt())
    + '\nResponde SOLO con JSON válido, sin markdown, con las claves "modo", "hechos", "calculo", "tiempos", "masas", "dimensiones" y "cargas".\n'
)

MODOS_VALIDOS = ("MULTA_TRAFICO", "MULTA_TRANSPORTE", "CONSULTA_GENERAL")


def analizar(pregunta: str, historial: list, con_calculo: bool):
    """Devuelve (modo, hechos, extraccion). UNA sola llamada a Gemini.
    Si no hay posibilidad de cálculo se usa el prompt corto (solo clasificación y hechos)."""
    contexto_previo = "\n".join(
        f"{m['role']}: {m['content'][:1000] if m['role'] == 'user' else m['content'][:300]}"
        for m in historial[-6:]
    )
    prompt = f"HISTORIAL RECIENTE:\n{contexto_previo}\n\nÚLTIMA CONSULTA:\n{pregunta}"
    try:
        r = llamar_gemini(
            model=MODELO_ANALISIS,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=INSTRUCCION_ANALISIS if con_calculo else INSTRUCCION_PLANIFICADOR,
                max_output_tokens=1500 if con_calculo else 400,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        datos = json.loads(re.sub(r"```(?:json)?", "", r.text or "").strip())
        modo = str(datos.get("modo", "")).strip().upper()
        if modo not in MODOS_VALIDOS:
            modo = "CONSULTA_GENERAL"
        hechos = [h.strip() for h in datos.get("hechos", []) if isinstance(h, str) and h.strip()][:3]
        return modo, (hechos if modo == "MULTA_TRAFICO" else []), (datos if con_calculo else None)
    except Exception:
        return "CONSULTA_GENERAL", [], None


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
4. Velocidad/radar y alcohol/drogas: pregunta SOLO por lo que falte de verdad (velocidad o tasa medida, límite de la vía,
   tipo de cinemómetro o etilómetro, tipo de conductor). No preguntes datos que no cambien la ficha: por ejemplo, el tipo de vía
   si ya te dan el límite.
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
9. CONSECUENCIAS (inmovilización, retirada, depósito, gastos): responde solo si preguntan por ellas. Primero identifica la infracción
   y da su ficha; después usa el bloque NORMA DE REFERENCIA del CONTEXTO y cita la letra del art. 104.1 (o del 105.1) que encaje. Si la
   ficha o su comentario ya dicen algo, úsalo. Si ninguna letra encaja, di "no consta" y no la inventes. Si depende de un dato que falta
   (p. ej. la clase de permiso que exige el vehículo o la que posee el conductor), pregúntalo en una frase. Para la letra k aplica los
   criterios del bloque de referencia (C1 y D1 también; no se inmoviliza si posee C1 o D1 cuando se exige C o D, ni si el permiso adecuado está caducado).
10. VELOCIDAD, ALCOHOL, DROGAS: si el CONTEXTO incluye la NORMA DE REFERENCIA PENAL, úsala para decidir si el hecho es delito o infracción
    administrativa ANTES de dar una multa; si es presunto delito, indícalo en lugar de una multa. Las opciones de la ficha del art. 52
    (5A, 5E, 5G, 5I...) dependen del exceso sobre el límite DESPUÉS de restar el margen del cinemómetro: aplica margen y tabla de excesos
    SOLO si figuran en el CONTEXTO o en el bloque B; si no figuran, dilo en una frase y NO elijas una opción al azar ni des cuantía.
    No preguntes por el margen si no consta: indica que debe aplicarse el EMP del certificado del equipo.

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
            ),
            MODELO_EMBEDDING,
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
# 4b. Descomposición de casos de tráfico en hechos infractores (solo si se fuerza
#     "Resto de infracciones" Y DESCOMPONER_EN_MODO_FORZADO está activo)
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
            model=MODELO_ANALISIS,
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
# 5. CÁLCULOS (tiempos de conducción, masas, dimensiones, cargas)
#    La IA solo EXTRAE los datos del relato (en la misma llamada del análisis);
#    Python hace las cuentas y elige la ficha.
# ------------------------------------------------------------------
@st.cache_resource
def obtener_indices_lexicos():
    if not (BUSCADOR_OK and DATOS_CALCULO):
        return None
    return construir_indices(DATOS_CALCULO)


INDICES_LEXICOS = obtener_indices_lexicos()

RE_CALCULO = re.compile(
    r"conduc|descans|paus|horas|minutos|tac[óo]graf|peso|pesa|kg|kilo|tonelad|\bt\b|mma|\beje|altur|\balto|anch|larg|longitud|metros|\bm\b|sobresal|baca|km/?h|velocid|radar|cinem|l[ií]mit|limitad|tramo|urban|v[ií]a|autov|autopista|carretera|travesía|circunvalaci",
    re.I,
)


def parece_calculo(pregunta: str, calculo_reciente: bool) -> bool:
    tiene_cifras = bool(re.search(r"\d", pregunta))
    if calculo_reciente:
        # el mensaje puede completar o variar el cálculo anterior ("público", "sin autorización"...),
        # pero un mensaje largo y sin cifras es casi seguro otra consulta
        return tiene_cifras or len(pregunta.split()) <= 12
    return tiene_cifras and bool(RE_CALCULO.search(pregunta))


def texto_faltan(texto_calc: str) -> str:
    """Pregunta directa (sin llamar al modelo) cuando a la calculadora le faltan datos."""
    m = re.search(r"FALTAN DATOS: (.*)", texto_calc)
    items = [x.strip() for x in (m.group(1).split(";") if m else []) if x.strip()]
    conocidos = re.search(r"Datos ya conocidos: (.*)", texto_calc)
    lineas = [f"Para calcularlo necesito {'este dato' if len(items) == 1 else 'estos datos'}:"]
    lineas += [f"- {x[0].upper() + x[1:]}" for x in items]
    if conocidos:
        lineas.append(f"\n(Ya tengo: {conocidos.group(1)}.)")
    return "\n".join(lineas)


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
8. Si el bloque dice que no hay infracción o "No hay ficha aplicable", NO cites artículos, importes ni fichas que no estén en el propio bloque: da solo el resultado del cálculo.
9. En "Responsable" copia exactamente lo que indique el bloque; no elijas tú entre varios responsables.
10. Si el bloque contiene varias INFRACCIONES, preséntalas por separado, una lista por cada una.
11. En velocidad, la línea "Cálculo:" debe decir SIEMPRE la velocidad medida, el margen del cinemómetro aplicado y la velocidad corregida, tal como figuran en el bloque.
12. Si el bloque contiene "ALTERNATIVA ADMINISTRATIVA", presenta primero la vía penal y después esa alternativa con su ficha completa (cuantía y puntos), aclarando que se rellena por si se desestima la vía penal.
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
    st.caption(f"Límite de peticiones a Gemini: {MAX_RPM}/min (embeddings: {MAX_RPM_EMBEDDING}/min)")
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
            # 0. UN solo análisis por mensaje: modo + hechos infractores + datos para las calculadoras.
            #    Se omite cuando no hace falta (modo forzado sin cálculo posible, o código de transporte).
            codigo_transporte = detectar_codigo(pregunta_usuario)
            posible_calculo = (
                bool(DATOS_CALCULO)
                and modo_forzado != "CONSULTA_GENERAL"
                and parece_calculo(pregunta_usuario, st.session_state.get("calculo_reciente", False))
            )
            necesita_analisis = posible_calculo or (not modo_forzado and not codigo_transporte)
            modo_plan, hechos_plan, extraccion = "CONSULTA_GENERAL", [], None
            if necesita_analisis:
                modo_plan, hechos_plan, extraccion = analizar(pregunta_usuario, historial_previo, posible_calculo)

            resultado_calculo = {"estado": "sin_calculo", "texto": "", "etiqueta": ""}
            if posible_calculo and extraccion:
                resultado_calculo = calcular(extraccion, DATOS_CALCULO)

            respuesta_directa = None  # si no hace falta el modelo para responder, se rellena

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
                if resultado_calculo["estado"] == "faltan":
                    respuesta_directa = texto_faltan(texto_calc)  # pregunta directa: 0 llamadas al modelo
                elif resultado_calculo.get("respuesta"):
                    respuesta_directa = resultado_calculo["respuesta"]  # respuesta ya redactada por el cálculo (velocidad)
                else:
                    prompt_actual = f"{texto_calc}\n\nPREGUNTA:\n{pregunta_usuario}"
            else:
                st.session_state.calculo_reciente = False

                # 1. Modo: el elegido por el usuario, el de un código de transporte o el del análisis
                hechos = []
                if modo_forzado:
                    modo = modo_forzado
                    if modo == "MULTA_TRAFICO":
                        hechos = hechos_plan or (
                            descomponer_hechos(pregunta_usuario, historial_previo) if DESCOMPONER_EN_MODO_FORZADO else []
                        )
                elif codigo_transporte:
                    modo = "MULTA_TRANSPORTE"  # con un código de infracción de transporte es transporte seguro
                else:
                    modo, hechos = modo_plan, hechos_plan
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
                # Consecuencias (inmovilización, retirada, depósito): se añade la norma de referencia
                texto_reciente = " ".join(
                    [pregunta_usuario] + [m["content"] for m in historial_previo[-4:] if m["role"] == "user"]
                )
                if REFERENCIA_INMOVILIZACION and RE_CONSECUENCIA.search(texto_reciente):
                    contexto += (
                        "\n\n---\n\nNORMA DE REFERENCIA (arts. 104 y 105 LSV, instrucción y criterio de aplicación; "
                        "úsala para responder sobre inmovilización, retirada y depósito):\n" + REFERENCIA_INMOVILIZACION
                    )
                # Velocidad, alcohol, drogas, temeridad, negativa: se añade siempre la norma de referencia penal
                if REFERENCIA_PENAL and modo in ("MULTA_TRAFICO", "CONSULTA_GENERAL") and RE_PENAL.search(texto_reciente):
                    contexto += "\n\n---\n\nNORMA DE REFERENCIA PENAL (úsala para decidir si es delito o infracción administrativa):\n" + REFERENCIA_PENAL
                if MODO_DEBUG:
                    with st.expander(f"🔍 Debug · modo={modo} · filtro={filtro_usado} · {len(fragmentos)} fragmentos"):
                        if hechos:
                            st.markdown("**Hechos infractores detectados:** " + " | ".join(hechos))
                        for i, f in enumerate(fragmentos, 1):
                            st.markdown(f"**Fragmento {i}**")
                            st.text(f[:700])
                prompt_actual = f"CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta_usuario}"

            if respuesta_directa is not None:
                texto_acumulado = respuesta_directa
                response_placeholder.markdown(texto_acumulado)
            else:
                # 3. Historial conversacional
                mensajes_previos = historial_previo[-VENTANA_HISTORIAL:]
                contents_conversacion = [
                    {
                        "role": "model" if m["role"] == "assistant" else "user",
                        "parts": [{"text": m["content"]}],
                    }
                    for m in mensajes_previos
                ]
                contents_conversacion.append({"role": "user", "parts": [{"text": prompt_actual}]})

                # 4. Config según el modo
                configuracion_ia = GenerateContentConfig(
                    system_instruction=cfg["prompt"],
                    max_output_tokens=cfg["max_tokens"],
                    temperature=cfg["temp"],
                )

                # 5. Streaming (pasa por el limitador de peticiones)
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

        if MODO_DEBUG:
            resumen = ", ".join(f"{n}× {m}" for m, n in Counter(LLAMADAS_CONSULTA).items()) or "ninguna"
            st.caption(f"📡 Llamadas a Gemini en esta consulta: {len(LLAMADAS_CONSULTA)} ({resumen})")

    st.session_state.mensajes.append({"role": "assistant", "content": texto_acumulado})
