# -*- coding: utf-8 -*-
"""
Calculadoras deterministas del Asistente Legal.

La IA solo EXTRAE datos del relato; aquí se hacen las cuentas y se elige la ficha.
No depende de Streamlit ni de Gemini, así que se puede probar sola.

    datos = cargar_datos()
    r = calcular(extraccion, datos)   # -> {"estado", "tipo", "etiqueta", "texto"}
"""
import json
import re
from pathlib import Path

# ------------------------------------------------------------------
# Carga de datos (generados por preparar_datos.py)
# ------------------------------------------------------------------
def cargar_datos(carpeta: str = "datos"):
    base = Path(__file__).parent / carpeta
    try:
        baremo = json.loads((base / "baremo_transporte.json").read_text(encoding="utf-8"))
        trafico = json.loads((base / "codificado_trafico.json").read_text(encoding="utf-8"))
        try:
            velocidad = json.loads((base / "velocidad.json").read_text(encoding="utf-8"))
        except Exception:
            velocidad = None  # sin cuadro de velocidades, esa calculadora solo comprueba el umbral penal
        try:
            alcohol = json.loads((base / "alcohol.json").read_text(encoding="utf-8"))
        except Exception:
            alcohol = None
        return {"baremo": baremo, "trafico": trafico, "velocidad": velocidad, "alcohol": alcohol}
    except Exception:
        return None


# ------------------------------------------------------------------
# Utilidades de formato
# ------------------------------------------------------------------
def hm(minutos: int) -> str:
    return f"{minutos // 60} h {minutos % 60:02d} min"


def fmt_kg(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".") + " kg"


def fmt_eur(x) -> str:
    if x is None:
        return "?"
    return (f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")).replace(",00", "") + " €"


def _num(s: str) -> str:
    return str(int(s)) if s and s.isdigit() else (s or "")


REGLAMENTO = {"VEH": "RGVeh", "CIR": "RGC", "CON": "RGCon"}


def lineas_ficha_transporte(f: dict, indent: str = "   ") -> str:
    out = [
        f"{indent}- Código: {f['cod']}",
        f"{indent}- Concepto: {f['concepto']}",
        f"{indent}- Norma infringida: {f['norma_infringida']}",
        f"{indent}- Norma sancionadora: {f['norma_sancionadora']}",
        f"{indent}- Cuantía y medidas (tal cual figura en la ficha): {f['cuantia_raw']}",
    ]
    for a in f.get("alertas", []):
        out.append(f"{indent}- AVISO: {a}")
    return "\n".join(out)


def _hecho_limpio(f: dict) -> str:
    """La ficha 48/5A trae '(GRAVE)' en el texto aunque es muy grave: se quita para no contradecir la calificación."""
    h = f["hecho"]
    if re.search(r"\(GRAVE\)\s*$", h) and f["calificacion"].lower() != "grave":
        h = re.sub(r"\s*\(GRAVE\)\s*$", "", h)
    return h


def lineas_ficha_trafico(f: dict, indent: str = "   ") -> str:
    reg = REGLAMENTO.get(f["norma"].split()[0], f["norma"].split()[0])
    art = f"Art. {_num(f['art_reg'])}.{f['apartado_reg']} {reg}" if f["apartado_reg"] else f"Art. {_num(f['art_reg'])} {reg}"
    if f["art_lsv"]:
        art += f" · Art. {_num(f['art_lsv'])}" + (f".{f['apartado_lsv']}" if f["apartado_lsv"] else "") + " LSV"
    cal = f["calificacion"]
    if f["art_sancionador"]:
        cal += f" (art. {re.sub(r'^0+', '', f['art_sancionador'])} LSV)"
    red = f" (cuantía reducida: {fmt_eur(f['multa_reducida'])})" if f["multa_reducida"] is not None else ""
    out = [
        f"{indent}- Norma y artículo / Precepto: {art} · opción {f['opcion']}",
        f"{indent}- Infracción: {_hecho_limpio(f)[:400]}",
        f"{indent}- Calificación: {cal}",
        f"{indent}- Cuantía: {fmt_eur(f['multa'])}{red}",
        f"{indent}- Puntos: {f['puntos'] or '0'}",
        f"{indent}- Responsable: {f['responsable']}",
    ]
    if f["comentario"]:
        out.append(f"{indent}- Comentario: {f['comentario'][:500]}")
    return "\n".join(out)


def _trafico(datos, norma: str, art_reg: str, opcion: str, apartado_reg: str = None):
    for f in datos["trafico"]:
        if f["norma"] == norma and f["art_reg"] == art_reg and f["opcion"] == opcion:
            if apartado_reg is None or f["apartado_reg"] == apartado_reg:
                return f
    return None


def _resultado(estado, tipo, etiqueta, texto, respuesta=None, motivo=None):
    """respuesta: texto ya redactado para el usuario; si existe, la app lo muestra tal cual, sin llamar al modelo.
    motivo: para "sin_calculo", por qué no se usó la calculadora (se muestra en el debug)."""
    return {"estado": estado, "tipo": tipo, "etiqueta": etiqueta, "texto": texto, "respuesta": respuesta, "motivo": motivo}


def _sin_calculo(motivo):
    return _resultado("sin_calculo", None, "", "", None, motivo)


def _faltan(tipo, etiqueta, faltan, datos_conocidos=""):
    t = "RESULTADO DEL CÁLCULO (verificado por código)\n"
    if datos_conocidos:
        t += f"Datos ya conocidos: {datos_conocidos}\n"
    t += "FALTAN DATOS: " + "; ".join(faltan) + "\n"
    t += "No des ninguna sanción todavía: pregunta SOLO por esos datos."
    return _resultado("faltan", tipo, etiqueta, t)


# ==================================================================
# 1. TIEMPOS DE CONDUCCIÓN
# ==================================================================
def _nueva_jornada() -> dict:
    return {"conduccion": 0, "continuo": 0, "max_continuo": 0, "fraccion15": False,
            "pausas_total": 0, "pausa_max": 0}


def calcular_jornadas(eventos: list) -> list:
    """Suma conducción por jornada. Pausa válida: 45 min seguidos, o 15 min + 30 min (en ese orden).
    Un descanso de 9 h o más cierra la jornada."""
    jornadas, j = [], _nueva_jornada()
    for ev in eventos:
        tipo, m = ev["tipo"], int(ev["minutos"])
        if tipo == "conduccion":
            j["conduccion"] += m
            j["continuo"] += m
            j["max_continuo"] = max(j["max_continuo"], j["continuo"])
        elif tipo == "descanso" and m >= 540:
            jornadas.append(j)
            j = _nueva_jornada()
        else:
            if m >= 15:
                j["pausas_total"] += m
                j["pausa_max"] = max(j["pausa_max"], m)
            if m >= 45 or (m >= 30 and j["fraccion15"]):
                j["continuo"], j["fraccion15"] = 0, False
            elif m >= 15 and not j["fraccion15"]:
                j["fraccion15"] = True
    jornadas.append(j)
    return [x for x in jornadas if x["conduccion"] > 0]


def en_tramo(t: dict, v: float) -> bool:
    lo = v >= t["min"] if t["min_incl"] else v > t["min"]
    hi = True if t["max"] is None else v < t["max"]
    return lo and hi


NOMBRE_TRANSPORTE = {"mercancias": "mercancías", "viajeros": "viajeros"}


def resolver_tiempos(p: dict, datos: dict) -> dict:
    etiqueta = "🧮 Cálculo: tiempos de conducción"
    eventos = p.get("eventos") or []
    jornadas = calcular_jornadas(eventos) if eventos else []
    if not jornadas:
        return _faltan("tiempos", etiqueta, ["los periodos de conducción y de pausa o descanso del conductor, en orden"])

    transp = p.get("transporte")
    transportes = [transp] if transp in NOMBRE_TRANSPORTE else ["mercancias", "viajeros"]
    amp = p.get("ampliacion_10h")
    series = [10] if amp is True else [9] if amp is False else [9, 10]

    preguntas = []
    if transp not in NOMBRE_TRANSPORTE:
        preguntas.append("¿Es transporte de mercancías o de viajeros? (los tramos son iguales, cambia el código)")
    if amp not in (True, False):
        preguntas.append("¿El conductor puede ampliar a 10 h ese día (máximo 2 veces por semana) o se aplica el límite de 9 h? "
                         "Decide entre la serie 'sobre 9 h' y la 'sobre 10 h'")

    bar = datos["baremo"]
    lineas = ["RESULTADO DEL CÁLCULO (verificado por código; no recalcules ni cambies estas cifras)",
              "Tipo de cálculo: tiempos de conducción"]
    avisos = []
    for n, j in enumerate(jornadas[:2], 1):
        pref = f"Jornada {n}: " if len(jornadas) > 1 else ""
        T, P, Pm = j["conduccion"], j["pausas_total"], j["pausa_max"]
        lineas.append(f"{pref}conducción diaria total = {hm(T)} ({T} min)")
        lineas.append(f"{pref}pausas/descansos acumulados (≥15 min) = {hm(P)} (la más larga: {hm(Pm)})")
        lineas.append(f"{pref}conducción continua máxima sin pausa válida = {hm(j['max_continuo'])}"
                      + (f" → SUPERA 4 h 30 min en {hm(j['max_continuo'] - 270)}" if j["max_continuo"] > 270
                         else " (no supera 4 h 30 min; el baremo exige superarlas)"))

        lineas.append(f"\n{pref}FICHAS POR CONDUCCIÓN DIARIA:")
        for tr in transportes:
            for serie in series:
                tramos = [t for t in bar["tiempos_diarios"] if t["transporte"] == tr and t["serie"] == serie]
                base = [t for t in tramos if not t["sin_pausa"] and en_tramo(t, T)]
                sinp = [t for t in tramos if t["sin_pausa"] and en_tramo(t, T)]
                aplic, nota = base, None
                if sinp:
                    u = sinp[0]["umbral_pausa_min"]
                    if P < u:
                        aplic = sinp
                        nota = (f"Aplica la variante 'sin pausa o descanso de 4 h 30 min como mínimo': "
                                f"pausas acumuladas {hm(P)} < 4 h 30 min (con cualquier criterio, acumulado o seguido).")
                    elif Pm < u:
                        nota = (f"AVISO: las pausas acumuladas ({hm(P)}) llegan a 4 h 30 min pero ninguna es seguida "
                                f"(la mayor {hm(Pm)}). Si el baremo exige 4 h 30 min SEGUIDOS, aplicaría la variante "
                                f"{sinp[0]['cod']}. El texto del baremo no lo aclara.")
                    else:
                        nota = "Hay pausa o descanso ≥ 4 h 30 min: no aplica la variante 'sin pausa'."
                tit = f"  · {NOMBRE_TRANSPORTE[tr].capitalize()} · serie 'sobre {serie} h'"
                if not aplic:
                    limite = serie * 60
                    lineas.append(f"{tit}: sin ficha (conducción {hm(T)} "
                                  f"{'no supera' if T <= limite else 'sin tramo en el baremo para'} {serie} h)")
                    if nota:
                        lineas.append(f"     {nota}")
                    continue
                for t in aplic:
                    f = bar["fichas"].get(t["cod"])
                    lineas.append(tit + ":")
                    lineas.append(lineas_ficha_transporte(f, "     "))
                    if t.get("nota_30km"):
                        lineas.append("     - Nota del baremo: procede inmovilizar solo si la distancia que aún debe recorrer "
                                      "el vehículo hasta su destino es superior a 30 km")
                if nota:
                    lineas.append(f"     {nota}")

        # conducción continua
        mc = j["max_continuo"]
        if mc > 270:
            lineas.append(f"\n{pref}FICHAS POR CONDUCCIÓN CONTINUA SIN PAUSA ({hm(mc)}):")
            for tr in transportes:
                tr_t = [t for t in bar["tiempos_continua"] if t["transporte"] == tr and en_tramo(t, mc)]
                if not tr_t:
                    lineas.append(f"  · {NOMBRE_TRANSPORTE[tr].capitalize()}: sin tramo en el baremo")
                for t in tr_t:
                    lineas.append(f"  · {NOMBRE_TRANSPORTE[tr].capitalize()}:")
                    lineas.append(lineas_ficha_transporte(bar["fichas"][t["cod"]], "     "))
                    if t.get("nota_30km"):
                        lineas.append("     - Nota del baremo: procede inmovilizar solo si la distancia que aún debe recorrer "
                                      "el vehículo hasta su destino es superior a 30 km")
    if len(jornadas) > 2:
        avisos.append("Hay más de 2 jornadas en el relato: solo se han calculado las 2 primeras.")
    avisos.append("No se calculan aquí los totales semanal/bisemanal ni el descanso diario.")
    if preguntas:
        lineas.append("\nPREGUNTAS PENDIENTES (hazlas al final, en una o dos frases cortas):")
        lineas += [f"  - {q}" for q in preguntas]
    lineas.append("\nAVISOS: " + " ".join(avisos))
    return _resultado("completo", "tiempos", etiqueta, "\n".join(lineas))


# ==================================================================
# 2. MASAS
# ==================================================================
def categoria_de_mma(mma_kg: float) -> str:
    return "N1" if mma_kg <= 3500 else "N2" if mma_kg <= 12000 else "N3"


def resolver_masas(p: dict, datos: dict) -> dict:
    etiqueta = "🧮 Cálculo: exceso de masa"
    faltan = []
    peso, mma = p.get("peso_medido_kg"), p.get("mma_kg")
    base, regimen = p.get("base"), p.get("regimen")
    if not peso:
        faltan.append("peso medido (total o del eje)")
    if not mma:
        faltan.append("MMA autorizada (de la ficha técnica o de la autorización especial; del eje si el exceso es por eje)")
    if base not in ("mma", "eje"):
        faltan.append("si el exceso es sobre la MMA total o sobre un eje")
    if regimen not in ("publico", "privado"):
        faltan.append("si es transporte público o privado complementario")
    margen = p.get("margen_bascula")
    if not p.get("peso_ya_neto") and not (isinstance(margen, dict) and margen.get("valor") is not None):
        faltan.append("margen de la báscula según su certificado (kg o %), o confirmar que el peso ya viene neto")
    cat = p.get("categoria")
    if cat not in ("N1", "N2", "N3"):
        if base == "mma" and mma:
            cat = categoria_de_mma(mma)
        elif p.get("mma_vehiculo_kg"):
            cat = categoria_de_mma(p["mma_vehiculo_kg"])
        elif base == "eje":
            faltan.append("categoría del vehículo (N1, N2 o N3) o su MMA total")
    if peso and mma and (peso < 100 or mma < 100):
        faltan.append("confirmar unidades: los pesos parecen estar en toneladas")
    conocidos = ", ".join(x for x in (
        f"peso medido {fmt_kg(peso)}" if peso else "", f"MMA {fmt_kg(mma)}" if mma else "",
        f"base {base}" if base else "", f"régimen {regimen}" if regimen else "") if x)
    if faltan:
        return _faltan("masas", etiqueta, faltan, conocidos)

    # --- cálculo ---
    if p.get("peso_ya_neto"):
        margen_kg, txt_margen = 0.0, "peso ya neto (sin margen que restar)"
    else:
        v, u = margen["valor"], margen.get("unidad", "kg")
        margen_kg = peso * v / 100 if u == "%" else float(v)
        txt_margen = f"{v} {u}" + (f" de {fmt_kg(peso)} = {fmt_kg(margen_kg)}" if u == "%" else "")
    neto = peso - margen_kg
    exceso_kg = neto - mma
    pct = exceso_kg / mma * 100

    lineas = ["RESULTADO DEL CÁLCULO (verificado por código; no recalcules ni cambies estas cifras)",
              "Tipo de cálculo: exceso de masa",
              f"Base: {'MMA total del vehículo' if base == 'mma' else 'MMA del eje'} · "
              f"Régimen: transporte {regimen} · Categoría: {cat}",
              f"Peso medido: {fmt_kg(peso)} · margen de báscula restado: {txt_margen} · peso neto: {fmt_kg(neto)}",
              f"MMA de referencia: {fmt_kg(mma)} → exceso = {fmt_kg(exceso_kg)} = {pct:.2f} %"]
    avisos = []
    tr = [m for m in datos["baremo"]["masas"]
          if m["base"] == base and m["regimen"] == regimen and m["categoria"] == cat]
    tr.sort(key=lambda m: m["umbral"], reverse=True)

    if exceso_kg <= 0:
        lineas.append("RESULTADO: no hay exceso de masa tras restar el margen de la báscula; sin ficha de transporte.")
    else:
        elegida = next((m for m in tr if (pct >= m["umbral"] if m["incl"] else pct > m["umbral"])), None)
        if elegida is None:
            minimo = min(tr, key=lambda m: m["umbral"])
            lineas.append(f"RESULTADO TRANSPORTE: el baremo no tipifica este exceso ({pct:.2f} %): el escalón mínimo "
                          f"es {'≥' if minimo['incl'] else '>'} {minimo['umbral']:g} % ({minimo['cod']}). Sin ficha de transporte.")
        else:
            f = datos["baremo"]["fichas"][elegida["cod"]]
            lineas.append(f"FICHA DE TRANSPORTE APLICABLE (escalón {'≥' if elegida['incl'] else '>'} "
                          f"{elegida['umbral']:g} %, {elegida['gravedad']}):")
            lineas.append(lineas_ficha_transporte(f))
            if elegida["deducido"]:
                avisos.append(f"El porcentaje de {elegida['cod']} no se leía en el .txt y se ha deducido de la sección "
                              f"espejo (mismo importe): conviene contrastarlo con el PDF.")
            cercanos = [m for m in tr if abs(pct - m["umbral"]) < 0.1]
            if cercanos:
                avisos.append(f"El exceso ({pct:.2f} %) está a menos de 0,1 puntos del escalón "
                              f"{cercanos[0]['umbral']:g} %: revisa el margen de la báscula.")
    # ficha de tráfico (art. 14 RGVeh), aparte
    ft = _trafico(datos, "VEH LSV", "014", "5B", "1")
    if ft and exceso_kg > 0:
        lineas.append("\nFICHA DE TRÁFICO (aparte; el agente decide cuál formula):")
        lineas.append(lineas_ficha_trafico(ft))
    if cat == "N2" and base == "mma" and regimen == "publico" and 0 < pct < 5:
        avisos.append("Para N2 con exceso de MMA total inferior al 5 % el baremo no tipifica sanción de transporte "
                      "(confirmado por el usuario; no comprobado en el PDF).")
    avisos.append("La tolerancia de la báscula la aporta el agente según el certificado metrológico.")
    lineas.append("\nAVISOS: " + " ".join(avisos))
    return _resultado("completo", "masas", etiqueta, "\n".join(lineas))


# ==================================================================
# 3. DIMENSIONES (Anexo IX RGV, Tabla 3)
# ==================================================================
LIMITES = {
    "longitud": {
        "motor_no_autobus": (12.00, "Vehículo a motor (excepto autobuses)"),
        "remolque": (12.00, "Remolque (dimensión incluida la lanza)"),
        "articulado_no_autobus": (16.50, "Vehículo articulado, tractora + semirremolque (excepto autobuses)"),
        "tren_carretera": (18.75, "Tren de carretera (camión + remolque)"),
        "tren_carretera_portavehiculos_con_carga": (20.55, "Tren de carretera especializado en transporte de vehículos, con carga y voladizo trasero autorizado"),
        "autobus_rigido_2_ejes": (13.50, "Autobús rígido de 2 ejes"),
        "autobus_rigido_mas_2_ejes": (15.00, "Autobús rígido de más de 2 ejes"),
        "autobus_articulado_o_con_remolque": (18.75, "Autobús articulado o con remolque"),
        "euro_modular": (32.00, "Conjunto en configuración euro-modular"),
        "semirremolque_pivote_a_trasera": (12.00, "Semirremolque: distancia del eje del pivote al extremo trasero"),
        "semirremolque_pivote_a_delantera": (2.04, "Semirremolque: distancia del pivote al punto más adelantado de su parte delantera"),
    },
    "anchura": {
        "general": (2.55, "Anchura general"),
        "vehiculo_acondicionado": (2.60, "Superestructura de vehículo acondicionado (paredes de 45 mm mínimo)"),
        "autobus_presos": (2.60, "Autobús para traslado de presos"),
    },
    "altura": {
        "general": (4.00, "Altura general"),
        "autobus_urbano": (4.20, "Autobús urbano (clase I)"),
        "especial_4_50": (4.50, "Portavehículos especializado, vehículo grúa, contenedor cerrado homologado intermodal/ULD, "
                                "altura variable con animales vivos/pacas/forrajes, suministro a industrias en menos de 50 km"),
    },
}
MODIFICADORES = {
    "intermodal_45_pies": (0.15, "Transporta contenedor o caja móvil de 45 pies en operación intermodal (+15 cm de longitud)"),
    "barras_indivisibles": (1.50, "Transporta barras, vigas o tubos indivisibles que solo superan la longitud (+150 cm)"),
}


def opciones_dimension_para_prompt() -> str:
    out = []
    for mag, d in LIMITES.items():
        out.append(f"  {mag}:")
        out += [f'    "{k}": {desc}' for k, (_, desc) in d.items()]
    out.append("  modificadores (lista, solo longitud):")
    out += [f'    "{k}": {desc}' for k, (_, desc) in MODIFICADORES.items()]
    return "\n".join(out)


def resolver_dimensiones(p: dict, datos: dict) -> dict:
    etiqueta = "🧮 Cálculo: dimensiones"
    mag, medida, tipo = p.get("magnitud"), p.get("medida_m"), p.get("tipo_vehiculo")
    faltan = []
    if mag not in LIMITES:
        faltan.append("qué dimensión se mide (longitud, anchura o altura)")
    if not medida:
        faltan.append("la medida en metros")
    if mag in LIMITES and tipo not in LIMITES[mag]:
        faltan.append("el tipo de vehículo o conjunto (p. ej. rígido, articulado, tren de carretera, autobús, portavehículos...)")
    if faltan:
        return _faltan("dimensiones", etiqueta, faltan)

    limite, desc = LIMITES[mag][tipo]
    if mag == "longitud" and limite >= 10 and medida < 0.4 * limite:
        return _faltan("dimensiones", etiqueta, [
            f"confirmar que {medida:.2f} m es la longitud TOTAL del vehículo y no lo que sobresale la carga "
            f"(si es lo que sobresale, es otro cálculo: art. 15 RGC)"])
    mods = [m for m in (p.get("modificadores") or []) if m in MODIFICADORES] if mag == "longitud" else []
    extra = sum(MODIFICADORES[m][0] for m in mods)
    limite_total = limite + extra
    exceso_cm = round((medida - limite_total) * 100)
    lineas = ["RESULTADO DEL CÁLCULO (verificado por código; no recalcules ni cambies estas cifras)",
              "Tipo de cálculo: dimensiones (Anexo IX RGV, Tabla 3)",
              f"Dimensión: {mag} · Vehículo: {desc}",
              f"Límite: {limite:.2f} m" + (f" + {extra:.2f} m por " + ", ".join(mods) if mods else "")
              + f" = {limite_total:.2f} m · Medida: {medida:.2f} m"]
    if exceso_cm <= 0:
        lineas.append("RESULTADO: no supera el límite; sin infracción por dimensiones.")
        return _resultado("completo", "dimensiones", etiqueta, "\n".join(lineas))

    lineas.append(f"RESULTADO: supera el límite en {exceso_cm} cm (art. 14.1 RGVeh).")
    causa, aut = p.get("causa_exceso"), p.get("autorizacion")
    f51 = _trafico(datos, "VEH LSV", "014", "5B", "1")
    f5a = _trafico(datos, "VEH LSV", "014", "5A", "2")
    f5d = _trafico(datos, "VEH LSV", "014", "5D", "2")
    preguntas = []
    if aut == "con_autorizacion_cumple":
        lineas.append("Circula conforme a una autorización especial: no hay infracción por este exceso.")
    elif aut == "incumple_condiciones" and f5d:
        lineas.append("\nFICHA APLICABLE (incumple las condiciones de la autorización):")
        lineas.append(lineas_ficha_trafico(f5d))
    elif causa in ("carga_indivisible", "caracteristica_tecnica") and aut == "sin_autorizacion" and f5a:
        lineas.append("\nFICHA APLICABLE (sin autorización, por carga indivisible o característica técnica):")
        lineas.append(lineas_ficha_trafico(f5a))
    elif causa == "otra" and f51:
        lineas.append("\nFICHA APLICABLE (exceso general de dimensiones):")
        lineas.append(lineas_ficha_trafico(f51))
    else:
        lineas.append("\nFICHAS POSIBLES (depende de la respuesta a la pregunta pendiente):")
        if f51:
            lineas.append("  · Si el exceso no se debe a carga indivisible ni a una característica técnica:")
            lineas.append(lineas_ficha_trafico(f51, "     "))
        if f5a:
            lineas.append("  · Si se debe a carga indivisible o característica técnica y NO tiene autorización especial:")
            lineas.append(lineas_ficha_trafico(f5a, "     "))
        if f5d:
            lineas.append("  · Si tiene autorización especial pero incumple sus condiciones:")
            lineas.append(lineas_ficha_trafico(f5d, "     "))
        preguntas.append("¿El exceso se debe a carga indivisible o a una característica técnica del vehículo, "
                         "y tiene autorización especial (y la cumple)?")
    lineas.append("\nAVISOS: ninguna ficha de tráfico de este artículo menciona inmovilización; "
                  "no la menciones salvo que conste en la ficha. En tráfico el importe es fijo, no escalonado.")
    if preguntas:
        lineas.append("PREGUNTAS PENDIENTES (hazla al final, en una frase corta):")
        lineas += [f"  - {q}" for q in preguntas]
    return _resultado("completo", "dimensiones", etiqueta, "\n".join(lineas))


# ==================================================================
# 4. CARGA QUE SOBRESALE (art. 15 RGC)
# ==================================================================
def limite_carga(excl, direccion, L, W, indiv, tipo_carga, ancho_total):
    """Devuelve (límite_m, explicación). Límite 0 = no puede sobresalir nada."""
    if W is not None and W < 1:  # ap. 4: vehículos de anchura inferior a 1 m
        if direccion == "lateral":
            return 0.50, "ap. 4: vehículo de anchura inferior a 1 m: hasta 0,50 m a cada lado de su eje longitudinal"
        if direccion == "delante":
            return 0.0, "ap. 4: no puede sobresalir por la extremidad anterior"
        return 0.25, "ap. 4: por detrás hasta 0,25 m"
    if excl is False:  # ap. 3: resto de vehículos
        if direccion == "detras":
            pct = 15 if indiv else 10
            return L * pct / 100, f"ap. 3: por detrás hasta el {pct} % de la longitud del vehículo ({L:.2f} m)" \
                                  f"{' (carga indivisible)' if indiv else ' (carga divisible)'}"
        return 0.0, "ap. 1 y 3: en vehículos no destinados exclusivamente a mercancías la carga solo puede sobresalir por detrás"
    # ap. 2: vehículos destinados exclusivamente al transporte de mercancías
    if not indiv:
        return 0.0, "ap. 1 y 2: solo la carga indivisible puede sobresalir en estos vehículos"
    if direccion in ("delante", "detras") and tipo_carga == "longitud_indivisible":
        if L > 5:
            lim = 2.0 if direccion == "delante" else 3.0
            return lim, f"ap. 2.a.1.º: vehículo de más de 5 m: {lim:g} m por la parte {'anterior' if direccion == 'delante' else 'posterior'}"
        return L / 3, f"ap. 2.a.2.º: vehículo de 5 m o menos: un tercio de su longitud ({L:.2f} m) por cada extremo"
    if direccion == "lateral":
        if ancho_total is not None and ancho_total > 2.55:
            return 0.0, "ap. 2.b: el ancho total supera 2,55 m, por lo que no se permite sobresalir lateralmente"
        return 0.40, ("ap. 2.b: hasta 0,40 m por cada lateral, solo si la dimensión menor de la carga indivisible supera "
                      "el ancho del vehículo y el ancho total no pasa de 2,55 m")
    return 0.0, "ap. 1 y 2: la carga indivisible no longitudinal solo puede sobresalir lateralmente (ap. 2.b)"


def resolver_cargas(p: dict, datos: dict) -> dict:
    etiqueta = "🧮 Cálculo: carga que sobresale (art. 15 RGC)"
    if p.get("transporte_especial"):
        return _resultado("completo", "cargas", etiqueta,
                          "RESULTADO DEL CÁLCULO (verificado por código)\nTransporte especial: según el art. 15.8 RGC se estará "
                          "a lo dispuesto en su autorización. No se calcula el límite; pregunta por las condiciones de la autorización.")
    excl, direccion, sobresale = p.get("vehiculo_exclusivo_mercancias"), p.get("direccion"), p.get("sobresale_m")
    indiv, tipo, L, W = p.get("indivisible"), p.get("tipo_carga"), p.get("longitud_vehiculo_m"), p.get("ancho_vehiculo_m")
    faltan = []
    if excl not in (True, False):
        faltan.append("si el vehículo está destinado EXCLUSIVAMENTE al transporte de mercancías (decide qué apartado del art. 15 se aplica)")
    if direccion not in ("detras", "delante", "lateral"):
        faltan.append("por dónde sobresale la carga (detrás, delante o lateral)")
    if not sobresale:
        faltan.append("cuánto sobresale la carga (metros)")
    if indiv not in (True, False):
        faltan.append("si la carga es indivisible")
    if indiv is True and excl is True and tipo not in ("longitud_indivisible", "otra") and direccion in ("delante", "detras"):
        faltan.append("si es carga de longitud indivisible (vigas, postes, tubos) u otra carga indivisible")
    necesita_L = (excl is False and direccion == "detras" and not (W is not None and W < 1)) or \
                 (excl is True and indiv is True and tipo == "longitud_indivisible" and direccion in ("delante", "detras"))
    if necesita_L and not L:
        faltan.append("la longitud del vehículo (m)")
    if L and (L < 1 or L > 25):
        faltan.append(f"confirmar la longitud del vehículo ({L:g} m): no parece plausible")
    if sobresale and sobresale > 15:
        faltan.append(f"confirmar cuánto sobresale la carga ({sobresale:g} m): no parece plausible")
    if faltan:
        return _faltan("cargas", etiqueta, faltan)

    lim, expl = limite_carga(excl, direccion, L, W, indiv, tipo, p.get("ancho_total_m"))
    exceso = sobresale - lim
    nombre = {"detras": "por detrás", "delante": "por delante", "lateral": "por un lateral"}[direccion]
    lineas = ["RESULTADO DEL CÁLCULO (verificado por código; no recalcules ni cambies estas cifras)",
              "Tipo de cálculo: carga que sobresale (art. 15 RGC)",
              f"Vehículo: {'destinado exclusivamente a mercancías' if excl else 'NO destinado exclusivamente a mercancías'}"
              + (f" · longitud {L:.2f} m" if L else "") + f" · carga {'indivisible' if indiv else 'divisible'}",
              f"Sobresale {nombre}: {sobresale:.2f} m · límite permitido: {lim:.2f} m ({expl})"]
    fichas, preguntas, avisos = [], [], []
    f_exceso = _trafico(datos, "CIR LSV", "015", "5A", "1")
    f_senal = _trafico(datos, "CIR LSV", "015", "5A", "6")
    f_noct = _trafico(datos, "CIR LSV", "015", "5B", "6")
    f_later = _trafico(datos, "CIR LSV", "015", "5A", "7")

    if exceso > 1e-9:
        lineas.append(f"RESULTADO: supera el límite en {round(exceso * 100)} cm"
                      + (f" ({exceso / lim * 100:.0f} % por encima del límite)" if lim > 0 else "") + ".")
        if f_exceso:
            fichas.append(("INFRACCIÓN 1: la carga sobresale más de lo permitido (art. 15.1)", f_exceso, None))
    else:
        lineas.append(f"RESULTADO: no supera el límite ({sobresale:.2f} m ≤ {lim:.2f} m): sin infracción por el tamaño del sobresalimiento.")

    luz, visib, v20 = p.get("luz"), p.get("visibilidad_reducida"), p.get("senal_v20")
    if sobresale > 0 and direccion == "detras" and not (W is not None and W < 1):
        if v20 is False:
            resp = "titular (no hay señal V-20 en el vehículo; la ficha dice que si la señal está en el vehículo es el conductor y en otro caso el titular)"
            if f_senal:
                fichas.append(("INFRACCIÓN 2: no señalizar con la señal V-20 la carga que sobresale por detrás (art. 15.6)", f_senal, resp))
            avisos.append("Un pañuelo, trapo o cinta no es la señal V-20 (Anexo XI RGVeh): panel en el extremo posterior, "
                          "perpendicular al eje; dos paneles en V invertida si la carga ocupa toda la anchura.")
        elif v20 is True:
            lineas.append("Señalización: lleva la señal V-20 (comprueba que es el panel reglamentario y, si la carga ocupa toda la anchura, que lleva dos paneles).")
        else:
            preguntas.append("¿Lleva la carga la señal V-20 (panel reglamentario)?")
    if sobresale > 0 and direccion in ("detras", "delante"):
        if visib is True and luz is False and f_noct:
            fichas.append((f"INFRACCIÓN: circular de noche o con poca visibilidad sin la luz {'roja' if direccion == 'detras' else 'blanca'} (art. 15.6, 2.º párrafo)", f_noct, None))
        elif visib is None:
            preguntas.append("¿Circulaba entre la puesta y la salida del sol o con visibilidad reducida (exige luz roja detrás / blanca delante)?")
    if sobresale > 0 and direccion == "lateral" and visib is True and luz is False and f_later:
        fichas.append(("INFRACCIÓN: carga lateral sin señalizar (art. 15.7)", f_later, None))
    if sobresale > 0 and direccion == "lateral" and visib is None:
        preguntas.append("¿Circulaba de noche o con visibilidad reducida? (la señalización lateral del art. 15.7 se exige entonces si la carga queda a más de 0,40 m de la luz de posición)")

    for titulo, f, resp in fichas:
        lineas.append("\n" + titulo + ":")
        txt = lineas_ficha_trafico(f)
        if resp:  # la ficha trae una regla ("si la señal está en el vehículo..."): se sustituye por el resultado aplicado
            txt = re.sub(r"(?m)^(\s+)- Responsable: .*$", lambda m: f"{m.group(1)}- Responsable (aplicando la regla de la propia ficha): {resp}", txt)
        lineas.append(txt)
    if not fichas:
        lineas.append("No hay ficha aplicable con estos datos: no cites artículos ni importes.")
    if avisos:
        lineas.append("\nAVISOS: " + " ".join(avisos))
    if preguntas:
        lineas.append("\nPREGUNTAS PENDIENTES (hazlas al final, en una o dos frases cortas):")
        lineas += [f"  - {q}" for q in preguntas]
    return _resultado("completo", "cargas", etiqueta, "\n".join(lineas))


# ==================================================================
# 5. VELOCIDAD: umbral penal (art. 379.1 CP) + graduación administrativa (cuadro DGT)
# ==================================================================
UMBRAL_PENAL = {"urbana": 60, "interurbana": 80, "travesia": 80}   # km/h sobre el límite (travesías: criterio de vía interurbana)
OPCION_PUNTUAL = {100: "5M", 300: "5I", 400: "5G", 500: "5E", 600: "5A"}
OPCION_TRAMO = {100: "5N", 300: "5J", 400: "5H", 500: "5F", 600: "5B"}
OPCION_ACCESO_PUNTUAL = {600: "5C"}   # el resto (100-500 €) es la 5K, de importe variable
OPCION_ACCESO_TRAMO = {600: "5D"}     # el resto (100-500 €) es la 5L, de importe variable


def corregir_velocidad(datos, v, cinemometro, nuevo=False):
    """Resta el margen del cinemómetro si está en la tabla Y confirmado.
    Devuelve (velocidad_corregida, texto) o (None, None) si no se puede aplicar."""
    vel = (datos or {}).get("velocidad") or {}
    m = vel.get("margenes") or {}
    cfg = ((m.get("tipos") or {}).get(cinemometro) or {}).get("nuevo" if nuevo else "en_servicio")
    if not m.get("activo") or not cfg or not cfg.get("confirmado"):
        return None, None
    etapa = "nuevo (fase de conformidad)" if nuevo else "en servicio"
    if v <= cfg["hasta_kmh"]:
        return v - cfg["fijo_kmh"], f"margen de cinemómetro {cinemometro} {etapa}: {cfg['fijo_kmh']} km/h (hasta {cfg['hasta_kmh']} km/h)"
    return int(v * (1 - cfg["porcentaje"] / 100)), f"margen de cinemómetro {cinemometro} {etapa}: {cfg['porcentaje']} % (por encima de {cfg['hasta_kmh']} km/h), redondeado hacia abajo"


def _corregir_cfg(cfg, v):
    return v - cfg["fijo_kmh"] if v <= cfg["hasta_kmh"] else int(v * (1 - cfg["porcentaje"] / 100))


def correcciones_posibles(datos, v):
    """Velocidades corregidas con TODOS los márgenes de la tabla (fijo/móvil/tramo, nuevo/en servicio).
    Sirve para comprobar si el resultado es el mismo con cualquier radar, sin preguntar el tipo."""
    m = ((datos or {}).get("velocidad") or {}).get("margenes") or {}
    return sorted({_corregir_cfg(cfg, v) for etapas in (m.get("tipos") or {}).values() for cfg in etapas.values()})


def tramo_velocidad(datos, limite, v_corr):
    tabla = ((datos.get("velocidad") or {}).get("graduacion") or {}).get(str(int(limite)))
    if not tabla:
        return "sin_tabla"
    for desde, hasta, multa, pts, cal in tabla:
        if v_corr >= desde and (hasta is None or v_corr <= hasta):
            return {"desde": desde, "hasta": hasta, "multa": multa, "puntos": pts, "calificacion": cal}
    return None  # por debajo del primer tramo: sin infracción


def _ficha_velocidad(datos, opcion, art="052"):
    for f in datos["trafico"]:
        if f["art_reg"] == art and f["opcion"] == opcion and f["apartado_reg"] == "1" \
                and f["norma"].startswith("CIR") and "KM/H" in f["hecho"]:
            return f
    return None


ARTICULO_GENERICO = {"urbana": "050", "travesia": "050", "interurbana": "048"}
CRITERIO_ARTICULO = {
    "048": "genérico interurbano",
    "050": "genérico en poblado y travesías",
    "052": "límite por señal o limitación específica",
}


def articulos_velocidad(via, origen):
    """Qué artículo del RGC se cita: 48 (genérico interurbano), 50 (genérico urbano o travesía) o 52 (señalizado o específico)."""
    gen = ARTICULO_GENERICO.get(via)
    if origen == "senalizado":
        return ["052"]
    if origen == "generico":
        return [gen] if gen else ["048", "050"]
    return ([gen] if gen else ["048", "050"]) + ["052"]  # origen desconocido: se ofrecen las alternativas


def _rellenar_hecho(h: str, vel, lim) -> str:
    """'CIRCULAR A ... KM/H, ESTANDO LIMITADA LA VELOCIDAD A ... KM/H' -> con la velocidad corregida y el límite del caso."""
    partes = h.split("...")
    if len(partes) == 3:
        return f"{partes[0]}{vel:g}{partes[1]}{lim:g}{partes[2]}"
    return h


def _precepto(f):
    ap = f["apartado_reg"]
    return f"Art. {_num(f['art_reg'])}.{ap} RGC · Art. {_num(f['art_lsv'])}.{f['apartado_lsv']} LSV"


def _bloque_admin(datos, p, vc, lim, via, acceso, de_tramo):
    """Graduación administrativa para la velocidad corregida vc. Devuelve (lineas_para_el_modelo, avisos, info) o None.
    info = None si no hay infracción; si la hay, un dict con los datos para redactar la respuesta."""
    tramo = tramo_velocidad(datos, lim, vc)
    if tramo == "sin_tabla":
        return None
    if tramo is None:
        return ["RESULTADO: tras restar el margen la velocidad no supera el primer tramo del cuadro para ese límite: sin infracción por velocidad.",
                "No hay ficha aplicable: no cites artículos ni importes."], [], None
    multa, pts = tramo["multa"], tramo["puntos"]
    rango = f"{tramo['desde']} a {tramo['hasta']} km/h" if tramo["hasta"] else f"{tramo['desde']} km/h en adelante"
    avisos, nota_puntos = [], None
    if acceso and lim < 100:
        # Anexo IV: en autovías/autopistas interurbanas de acceso a ciudades con límite inferior a 100, la multa es la del cuadro
        # del límite, pero los demás efectos (puntos, penales) solo se producen por encima de 100 km/h y según el límite de 100
        t100 = tramo_velocidad(datos, 100, vc) if vc > 100 else None
        pts = t100["puntos"] if t100 else 0
        opcion = (OPCION_ACCESO_TRAMO if de_tramo else OPCION_ACCESO_PUNTUAL).get(multa, "5L" if de_tramo else "5K")
        nota_puntos = ("Tramo de acceso a ciudad con límite inferior a 100 km/h (Anexo IV): la multa es la del cuadro del límite "
                       f"{lim:g}; los puntos y demás efectos solo cuentan por encima de 100 km/h, según el cuadro del límite 100"
                       + (f" ({vc:g} km/h → {pts} puntos)." if vc > 100 else " (no supera 100 km/h: 0 puntos)."))
        lineas = [f"Tramo del cuadro para límite {lim:g}: {rango} → {multa} € (multa del cuadro del límite {lim:g})",
                  f"Puntos: {pts} ({nota_puntos})"]
    else:
        opcion = (OPCION_TRAMO if de_tramo else OPCION_PUNTUAL)[multa]
        lineas = [f"Tramo del cuadro para límite {lim:g}: {rango} → {multa} € · {pts} puntos · {tramo['calificacion']}"]
        if lim < 100 and via != "urbana":
            avisos.append("Si se detectó en un tramo de autopista o autovía interurbana de acceso a una ciudad (límite inferior a 100 km/h), indícalo: "
                          "la multa es la misma, pero los puntos y demás efectos solo cuentan por encima de 100 km/h (Anexo IV).")
    arts = articulos_velocidad(via, p.get("limite_origen"))
    fichas_art = [(art, _ficha_velocidad(datos, opcion, art)) for art in arts]
    fichas_art = [(art, f) for art, f in fichas_art if f]
    lineas.append(f"\nFICHA APLICABLE (opción {opcion}, {'velocidad media en tramo' if de_tramo else 'medición puntual'}):")
    info = {"multa": multa, "reducida": multa / 2, "puntos": pts, "opcion": opcion, "nota_puntos": nota_puntos,
            "calificacion": tramo["calificacion"], "precepto": None, "infraccion": None, "responsable": None}
    if fichas_art:
        f0 = fichas_art[0][1]
        if len(fichas_art) > 1:  # origen del límite desconocido: la multa y los puntos son los mismos, cambia el artículo
            precepto = " ó ".join(f"{_precepto(f)} ({CRITERIO_ARTICULO[art]})" for art, f in fichas_art) \
                       + ". Elige según el origen del límite (la multa y los puntos son iguales)"
        else:
            precepto = f"{_precepto(f0)} · opción {opcion} ({CRITERIO_ARTICULO[fichas_art[0][0]]})"
        infraccion = _rellenar_hecho(_hecho_limpio(f0), vc, lim)
        cal = f0["calificacion"] + (f" (art. {re.sub(r'^0+', '', f0['art_sancionador'])} LSV)" if f0["art_sancionador"] else "")
        info.update({"precepto": precepto, "infraccion": infraccion, "responsable": f0["responsable"], "calificacion": cal})
        txt = lineas_ficha_trafico(f0)
        txt = re.sub(r"(?m)^(\s+)- Puntos: .*$", lambda m: f"{m.group(1)}- Puntos: {pts} (según el cuadro de velocidades; la ficha puede traer «0, 2, 4, 6»)", txt)
        txt = re.sub(r"(?m)^(\s+- Infracción: )(.*)$", lambda m: m.group(1) + infraccion, txt)
        txt = re.sub(r"(?m)^(\s+)- Norma y artículo / Precepto: .*$", lambda m: f"{m.group(1)}- Norma y artículo / Precepto: {precepto}", txt)
        if opcion in ("5K", "5L"):  # importe variable: la ficha no trae cuantía, se toma la del cuadro (la reducida es la mitad)
            txt = re.sub(r"(?m)^(\s+)- Cuantía: .*$",
                         lambda m: f"{m.group(1)}- Cuantía: {multa} € (cuantía reducida: {multa / 2:g} €; la ficha {opcion} trae importe variable y se toma el del cuadro)", txt)
        lineas.append(txt)
    return lineas, avisos, info


def _resumen_velocidad(v, vc, info):
    return (f"Velocidad medida {v:g} km/h · velocidad corregida {vc:g} km/h · cuantía {info['multa']} € "
            f"(reducida {info['reducida']:g} €) · {info['puntos']} puntos")


def _texto_admin(v, vc, lim, txt_margen, info, avisos):
    """Respuesta lista para el usuario (sin pasar por el modelo)."""
    if info is None:
        return [f"**Velocidad medida {v:g} km/h · velocidad corregida {vc:g} km/h · sin infracción por velocidad**", "",
                f"- Límite: {lim:g} km/h", f"- Margen del cinemómetro: {txt_margen}",
                "- Tras restar el margen, la velocidad no supera el primer tramo del cuadro para ese límite."]
    out = [f"**{_resumen_velocidad(v, vc, info)}**", "",
           f"- Límite: {lim:g} km/h",
           f"- Margen del cinemómetro: {txt_margen}"]
    if info["precepto"]:
        out += [f"- Norma y artículo / Precepto: {info['precepto']}",
                f"- Infracción: {info['infraccion']}",
                f"- Calificación: {info['calificacion']}"]
    out += [f"- Cuantía: {info['multa']} €",
            f"- Cuantía reducida: {info['reducida']:g} €",
            f"- Puntos: {info['puntos']}"]
    if info["responsable"]:
        out.append(f"- Responsable: {info['responsable']}")
    if info["nota_puntos"]:
        out.append(f"- Nota: {info['nota_puntos']}")
    for a in avisos:
        out.append(f"- Nota: {a}")
    return out


def resolver_velocidad(p: dict, datos: dict) -> dict:
    etiqueta = "🧮 Cálculo: velocidad"
    v, lim, via, vc = p.get("velocidad_medida_kmh"), p.get("limite_kmh"), p.get("via"), p.get("velocidad_corregida_kmh")
    cin, nuevo, acceso = p.get("cinemometro"), bool(p.get("radar_nuevo")), bool(p.get("acceso_ciudad"))
    if not (datos.get("velocidad") or {}).get("graduacion"):
        return _sin_calculo("no se cargó datos/velocidad.json: falta el archivo junto a app.py, en la carpeta datos/")
    if not v or not lim:
        return _sin_calculo("faltan la velocidad medida o el límite")
    if v <= lim:
        return _sin_calculo(f"la velocidad medida ({v:g}) no supera el límite ({lim:g})")
    lim_eff = 100 if (acceso and lim < 100) else lim  # en accesos a ciudad los efectos penales cuentan desde 100 km/h
    conocidos = f"velocidad medida {v:g} km/h, límite {lim:g} km/h (exceso medido {v - lim:g} km/h)"

    # --- velocidad corregida: aportada, calculada con el margen del radar, o comprobada con todos los márgenes posibles ---
    if vc:
        vcs, vc_ref, txt_margen = [vc], vc, "velocidad corregida aportada por el agente"
    elif cin in ("fijo", "movil", "tramo"):
        c, txt = corregir_velocidad(datos, v, cin, nuevo)
        if c is None:
            return _faltan("velocidad", etiqueta, [
                f"la velocidad corregida, es decir, la leída menos el margen del cinemómetro {cin} según su certificado "
                "(la app no tiene confirmado el margen de ese tipo de radar)"], conocidos)
        vcs, vc_ref, txt_margen = [c], c, txt
    else:
        pos = correcciones_posibles(datos, v)
        if not pos:
            return _sin_calculo("velocidad.json no trae la tabla de márgenes")
        c_ref, _ = corregir_velocidad(datos, v, "fijo", False)
        vc_ref = c_ref if c_ref is not None else pos[0]
        vcs = pos
        txt_margen = (f"no se indicó el tipo de radar: según el margen, entre {pos[0]} y {pos[-1]} km/h; "
                      f"se usa {vc_ref} km/h (radar fijo en servicio)")
    varios = len(vcs) > 1

    # --- umbral penal (art. 379.1 CP) ---
    delito, umbral = False, None
    if v - lim_eff > min(UMBRAL_PENAL.values()):
        if via not in UMBRAL_PENAL:
            return _faltan("velocidad", etiqueta, ["si es vía urbana, interurbana o travesía (el umbral penal es +60 km/h en urbana y +80 km/h en interurbana)"], conocidos)
        umbral = UMBRAL_PENAL[via]
        if min(vcs) - lim_eff > umbral:
            delito = True
        elif max(vcs) - lim_eff <= umbral:
            delito = False
        else:
            return _faltan("velocidad", etiqueta, [
                "el tipo de cinemómetro (fijo o móvil) o la velocidad corregida con su margen: el exceso está cerca del umbral penal y el margen decide si es delito"], conocidos)

    de_tramo = cin == "tramo"
    if not delito and varios and len({str(tramo_velocidad(datos, lim, x)) for x in vcs}) > 1:
        return _faltan("velocidad", etiqueta, [
            "el tipo de cinemómetro (fijo o móvil) o la velocidad corregida con su margen: según el margen cambia el tramo de la sanción"], conocidos)

    lineas = ["RESULTADO DEL CÁLCULO (verificado por código; no recalcules ni cambies estas cifras)",
              "Tipo de cálculo: velocidad" + (", umbral penal del art. 379.1 CP y graduación administrativa" if delito else ", graduación de la sanción (Anexo IV de la LSV)"),
              f"Velocidad medida: {v:g} km/h · límite: {lim:g} km/h" + (" (a efectos penales se toma el límite de 100 km/h por tratarse de un acceso a ciudad)" if lim_eff != lim else ""),
              f"Margen del cinemómetro: {txt_margen}",
              f"Velocidad corregida: {vc_ref:g} km/h"]
    avisos = []
    if varios and not delito:
        avisos.append("No se indicó el tipo de radar: con cualquier margen (fijo o móvil, nuevo o en servicio) el tramo de la sanción es el mismo. "
                      "Si quieres el dato exacto, indica el tipo de radar.")
    admin = _bloque_admin(datos, p, vc_ref, lim, via, acceso, de_tramo)
    if admin is None:
        return _sin_calculo(f"el límite de {lim:g} km/h no figura en el cuadro del Anexo IV")  # flujo normal
    alt, avisos_alt, info = admin
    avisos += avisos_alt

    if delito:
        nombre_via = {"urbana": "urbana", "interurbana": "interurbana", "travesia": "travesía (se aplica el umbral de vía interurbana)"}[via]
        exceso_txt = (f"{vc_ref - lim_eff:g} km/h" if not varios else f"entre {min(vcs) - lim_eff:g} y {max(vcs) - lim_eff:g} km/h con cualquier margen")
        lineas.append(f"Vía {nombre_via} · umbral penal: más de {umbral} km/h sobre el límite. Exceso corregido: {exceso_txt}")
        lineas.append(f"RESULTADO PENAL: PRESUNTO DELITO del art. 379.1 CP (el exceso corregido supera {umbral} km/h"
                      + (" con cualquier margen de cinemómetro)." if varios else ")."))
        lineas.append("Consecuencia: procede el atestado por presunto delito contra la seguridad vial. Se rellena igualmente la denuncia administrativa "
                      "por si se desestima la vía penal; la vía penal prevalece.")
        lineas.append("\nALTERNATIVA ADMINISTRATIVA (por si se desestima la vía penal):")
        lineas += alt
        if varios and len({str(tramo_velocidad(datos, lim, x)) for x in vcs}) > 1:
            avisos.append("Según el margen del radar, el tramo administrativo podría ser otro: indica el tipo de radar para fijarlo.")
        if avisos:
            lineas.append("AVISOS: " + " ".join(avisos))
        if varios:
            frase = (f"La velocidad corregida, con cualquier margen de radar, está entre {min(vcs):g} y {max(vcs):g} km/h: supera el límite entre "
                     f"{min(vcs) - lim_eff:g} y {max(vcs) - lim_eff:g} km/h")
        else:
            frase = f"La velocidad corregida ({vc_ref:g} km/h) supera el límite en {vc_ref - lim_eff:g} km/h"
        cab = [f"**Vía penal: presunto delito del art. 379.1 CP.** {frase}, y el umbral penal es más de {umbral} km/h. "
               "Procede el atestado por presunto delito contra la seguridad vial.", "",
               "**Alternativa administrativa** (se rellena por si se desestima la vía penal; la penal prevalece):", ""]
        respuesta = "\n".join(cab + _texto_admin(v, vc_ref, lim, txt_margen, info, avisos))
        return _resultado("completo", "velocidad", etiqueta, "\n".join(lineas), respuesta)

    lineas += alt
    if avisos:
        lineas.append("AVISOS: " + " ".join(avisos))
    respuesta = "\n".join(_texto_admin(v, vc_ref, lim, txt_margen, info, avisos))
    return _resultado("completo", "velocidad", etiqueta, "\n".join(lineas), respuesta)


# ------------------------------------------------------------------
# Lector de velocidad directamente del texto (sin IA): es el caso más frecuente y no debe depender del modelo
# ------------------------------------------------------------------
_RE_LIMITE = [
    re.compile(r"limitad[oa]s?\s*(?:la\s+velocidad\s*)?(?:a|en|hasta)?\s*(\d{2,3})\b", re.I),
    re.compile(r"l[ií]mite\s+(?:\w+\s+){0,2}?(?:de|es|era|a|:)?\s*(\d{2,3})\b", re.I),
    re.compile(r"velocidad\s+m[aá]xima(?:\s+permitida)?\s*(?:de|es|era|:)?\s*(\d{2,3})\b", re.I),
    re.compile(r"permitid[ao]s?\s*(?:es|son|de|:)?\s*(\d{2,3})\b", re.I),
    re.compile(r"(?:v[ií]a|tramo|zona|carretera|calle|autov[ií]a|autopista|traves[ií]a|urbano|urbana|interurbano|interurbana)\s+(?:de\s+)?(\d{2,3})\b", re.I),
]
_RE_CORREGIDA = [re.compile(r"corregid[ao]s?\s*(?:de|a|es|:)?\s*(\d{2,3})\b", re.I),
                 re.compile(r"(\d{2,3})\s*(?:km/?h)?\s*(?:ya\s+)?corregid", re.I)]
_RE_NUM = re.compile(r"(?<![\d.,])(\d{2,3})(?![\d])(?:[.,]\d+)?")
_RE_OTRA_UNIDAD = re.compile(r"\s*(?:metros?|m\b|kg|kilos?|tonelad|t\b|a[ñn]os?|puntos?|€|euros?|%|horas?|minutos?|min\b|cm\b|plazas?|mg)", re.I)
_RE_NO_VELOCIDAD = re.compile(r"\b(?:kg|kilos|tonelad\w*|mma|descansa\w*|tac[oó]graf\w*)\b|conducci[oó]n continua|horas de conducci", re.I)


def _lee_parcial(texto: str) -> dict:
    """Lee lo que haya en un mensaje, sin exigir que esté completo."""
    t = texto or ""
    out = {"limite_kmh": None, "velocidad_medida_kmh": None, "velocidad_corregida_kmh": None}
    span_lim = None
    for rx in _RE_LIMITE:
        m = rx.search(t)
        if m and 10 <= int(m.group(1)) <= 140:
            out["limite_kmh"], span_lim = int(m.group(1)), m.span(1)
            break
    span_corr = None
    for rx in _RE_CORREGIDA:
        m = rx.search(t)
        if m:
            out["velocidad_corregida_kmh"], span_corr = int(m.group(1)), m.span(1)
            break
    cand = []
    for m in _RE_NUM.finditer(t):
        n, sp = int(m.group(1)), m.span(1)
        if sp == span_lim or sp == span_corr or not (20 <= n <= 350):
            continue
        resto = t[m.end():m.end() + 14]
        if _RE_OTRA_UNIDAD.match(resto) and not re.match(r"\s*km", resto, re.I):
            continue
        con_unidad = bool(re.match(r"\s*(?:km/?h|kil[oó]metros)", resto, re.I))
        cand.append((0 if con_unidad else 1, m.start(), n))
    lim = out["limite_kmh"]
    cand = [c for c in cand if lim is None or c[2] > lim]
    if cand and lim is not None:
        out["velocidad_medida_kmh"] = sorted(cand)[0][2]
    low = t.lower()
    if re.search(r"interurban|carretera|autov[ií]a|autopista|nacional|comarcal", low) and not re.search(r"\burban[oa]\b", low):
        out["via"] = "interurbana"
    elif re.search(r"interurban", low):
        out["via"] = "interurbana"
    elif re.search(r"traves[ií]a", low):
        out["via"] = "travesia"
    elif re.search(r"urban|poblad|ciudad|\bcalle\b", low):
        out["via"] = "urbana"
    else:
        out["via"] = None
    if re.search(r"(?:radar|cinem[oó]metro)\s+de\s+tramo|velocidad\s+media|control\s+de\s+tramo", low):
        out["cinemometro"] = "tramo"
    elif re.search(r"(?:radar|cinem[oó]metro)\s+m[oó]vil|m[oó]vil\s+(?:de\s+)?(?:radar|cinem)", low):
        out["cinemometro"] = "movil"
    elif re.search(r"(?:radar|cinem[oó]metro)\s+fijo|\bfijo\b", low):
        out["cinemometro"] = "fijo"
    else:
        out["cinemometro"] = None
    out["radar_nuevo"] = True if re.search(r"radar\s+nuevo|nuevo\s+radar|primer\s+a[ñn]o", low) else None
    out["acceso_ciudad"] = True if re.search(r"\bacceso\b|\baccesos\b|circunvalaci", low) else None
    if re.search(r"se[ñn]al|panel|cartel", low):
        out["limite_origen"] = "senalizado"
    elif re.search(r"gen[eé]ric", low):
        out["limite_origen"] = "generico"
    else:
        out["limite_origen"] = None
    return out


def extraer_velocidad_conversacion(pregunta: str, previos_usuario: list, calculo_reciente: bool = False):
    """Devuelve el dict 'velocidad' para resolver_velocidad, o None si el mensaje no es un caso de velocidad.
    Si el mensaje actual no trae velocidad y límite pero el cálculo anterior era de velocidad ("radar fijo", "urbana"...),
    completa lo que falte con los mensajes anteriores del agente."""
    if _RE_NO_VELOCIDAD.search(pregunta or ""):
        return None
    actual = _lee_parcial(pregunta)
    if actual["velocidad_medida_kmh"] and actual["limite_kmh"]:
        base = actual
    elif calculo_reciente and previos_usuario:
        base = dict(actual)
        for texto in reversed(previos_usuario[-3:]):
            ant = _lee_parcial(texto)
            if ant["velocidad_medida_kmh"] and ant["limite_kmh"]:
                for k, v in ant.items():
                    if base.get(k) in (None, False):
                        base[k] = v
                break
    else:
        return None
    if not (base.get("velocidad_medida_kmh") and base.get("limite_kmh")) or base["velocidad_medida_kmh"] <= base["limite_kmh"]:
        return None
    return {k: v for k, v in base.items() if v is not None}


# ==================================================================
# 6. ALCOHOLEMIA (aire espirado): art. 20 RGC + umbral penal (art. 379.2 CP)
# ==================================================================
import math


def _mgl(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".").replace(".", ",")


def _suelo3(x: float) -> float:
    return math.floor(round(x * 1000, 6)) / 1000


def _corregir_alcohol(cfg, m, fijo=None):
    fijo = cfg["fijo_mgl"] if fijo is None else fijo
    if m <= cfg["hasta_mgl"]:
        return _suelo3(m - fijo)
    return _suelo3(m * (1 - cfg["porcentaje"] / 100))


def _nivel_alcohol(corr, limite):
    if corr > 2 * limite:
        return "alto"
    if corr > limite:
        return "bajo"
    return None


def _ficha_alcohol(datos, opcion):
    for f in datos["trafico"]:
        if f["norma"] == "CIR LSV" and f["art_reg"] == "020" and f["apartado_reg"] == "1" \
                and f["opcion"] == opcion and "AIRE ESPIRADO" in f["hecho"]:
            return f
    return None


def _hecho_corto(h: str) -> str:
    return re.sub(r"\s*\(.*$", "", h).strip().rstrip(".") + "."


NOMBRE_CONDUCTOR = {"general": "conductor en general (límite 0,25 mg/l)",
                    "profesional_novel": "conductor profesional, novel o de transporte especial (límite 0,15 mg/l)"}


def resolver_alcohol(p: dict, datos: dict) -> dict:
    etiqueta = "🧮 Cálculo: alcoholemia"
    alc = datos.get("alcohol")
    if not alc:
        return _sin_calculo("no se cargó datos/alcohol.json: falta el archivo junto a app.py, en la carpeta datos/")
    meds = [float(x) for x in (p.get("mediciones_mgl") or []) if x is not None and 0 < float(x) <= 5]
    if not meds:
        return _sin_calculo("no hay ninguna tasa en mg/l")
    m = min(meds)
    cfg = alc["margenes"]["en_servicio"]
    conocidos = "tasa(s) medida(s) " + " y ".join(_mgl(x) for x in meds) + " mg/l"

    # --- tasa corregida: aportada, calculada con el margen, o comprobada con el rango plausible ---
    tc = p.get("tasa_corregida_mgl")
    if tc:
        corrs, corr_ref, txt_margen = [float(tc)], float(tc), "tasa corregida aportada por el agente"
    elif m > cfg["hasta_mgl"]:
        corr_ref = _corregir_alcohol(cfg, m)
        corrs = [corr_ref]
        txt_margen = f"etilómetro en servicio, {str(cfg['porcentaje']).replace('.', ',')} % por encima de {_mgl(cfg['hasta_mgl'])} mg/l (Orden ITC/3707/2006)"
    else:
        corr_ref = _corregir_alcohol(cfg, m)
        corrs = sorted({_corregir_alcohol(cfg, m), _corregir_alcohol(cfg, m, alc["margenes"]["plausible_minimo_fijo_mgl"])})
        txt_margen = (f"etilómetro en servicio, {_mgl(cfg['fijo_mgl'])} mg/l hasta {_mgl(cfg['hasta_mgl'])} mg/l (Orden ITC/3707/2006)"
                      + ("" if cfg["confirmado_fijo"] else " [margen sin confirmar en la app]"))

    # --- tipo de conductor ---
    cond = p.get("conductor")
    conductores = [cond] if cond in alc["limites_mgl"] else list(alc["limites_mgl"])
    reinc = p.get("reincidente")

    # ¿el resultado es el mismo con todos los márgenes y tipos de conductor posibles? Si no, se pregunta lo que lo decide
    niveles = {(c, x): _nivel_alcohol(x, alc["limites_mgl"][c]) for c in conductores for x in corrs}
    cambia_margen = any(len({niveles[(c, x)] for x in corrs}) > 1 for c in conductores)
    cambia_conductor = len(conductores) > 1 and any(len({niveles[(c, x)] for c in conductores}) > 1 for x in corrs)
    if cambia_margen or cambia_conductor:
        pide = []
        if cambia_margen:
            pide.append("la tasa corregida, es decir, la medida menos el margen del etilómetro según su certificado "
                        "(el margen cambia el tramo de la sanción)")
        if cambia_conductor:
            pide.append("si el conductor es en general o profesional, novel, de urgencia o de mercancías peligrosas "
                        "(los límites son 0,25 y 0,15 mg/l y cambia el tramo)")
        return _faltan("alcohol", etiqueta, pide, conocidos)
    c_ref = conductores[0] if len(conductores) == 1 else "general"
    limite = alc["limites_mgl"][c_ref]
    nivel = niveles[(c_ref, corrs[0])]

    # --- vía penal (norma del RAG) ---
    pen = alc["penal"]
    objetivo = m >= pen["objetivo_min_medida_mgl"]
    ba, bb = pen["banda_sintomas_mgl"]
    banda = ba <= m <= bb
    sintomas = p.get("sintomas")
    delito = objetivo or (banda and sintomas is True)

    def info_admin():
        if nivel is None:
            return None
        ops = alc["opciones"][c_ref]
        opcion = ops["alto"] if nivel == "alto" else (ops["bajo_reincidente"] if reinc is True else ops["bajo"])
        f = _ficha_alcohol(datos, opcion)
        if not f:
            return None
        return {"opcion": opcion, "ficha": f, "multa": f["multa"], "puntos": int(re.sub(r"\D", "", f["puntos"]) or 0), "nivel": nivel}

    info = info_admin()

    def resumen():
        if info is None:
            return f"Tasa medida {_mgl(m)} mg/l · tasa corregida {_mgl(corr_ref)} mg/l · sin infracción administrativa por alcohol"
        return (f"Tasa medida {_mgl(m)} mg/l · tasa corregida {_mgl(corr_ref)} mg/l · cuantía {info['multa']:g} € "
                f"(reducida {info['multa'] / 2:g} €) · {info['puntos']} puntos")

    def cuerpo(con_ficha=True):
        l = [f"- Conductor: {NOMBRE_CONDUCTOR[c_ref]}" + ("" if len(conductores) == 1 else " (no se indicó; con 0,15 mg/l de límite el tramo es el mismo)"),
             "- Mediciones: " + (" y ".join(_mgl(x) for x in meds) + f" mg/l → se toma la menor ({_mgl(m)}), como indican las fichas"
                                if len(meds) > 1 else f"{_mgl(m)} mg/l"),
             f"- Margen del etilómetro: {txt_margen}",
             f"- Límite: {_mgl(limite)} mg/l (doble: {_mgl(2 * limite)} mg/l)"]
        if info is None:
            l.append(f"- La tasa corregida ({_mgl(corr_ref)} mg/l) no supera {_mgl(limite)} mg/l: sin infracción administrativa por alcohol.")
            return l
        f = info["ficha"]
        cal = f["calificacion"] + (f" (art. {re.sub(r'^0+', '', f['art_sancionador'])} LSV)" if f["art_sancionador"] else "")
        l += [f"- Norma y artículo / Precepto: {_precepto(f)} · opción {info['opcion']}",
              f"- Infracción: {_hecho_corto(f['hecho'])}",
              f"- Calificación: {cal}",
              f"- Cuantía: {info['multa']:g} €",
              f"- Cuantía reducida: {info['multa'] / 2:g} €",
              f"- Puntos: {info['puntos']}",
              f"- Responsable: {f['responsable']}"]
        if nivel == "bajo" and reinc is not True:
            ops = alc["opciones"][c_ref]
            fr = _ficha_alcohol(datos, ops["bajo_reincidente"])
            if fr:
                pts_r = int(re.sub(r"\D", "", fr["puntos"]) or 0)
                l.append(f"- Nota: si consta sanción firme por el art. 20 en el año anterior a los hechos, la cuantía es de {fr['multa']:g} € "
                         f"con {pts_r} puntos (opción {ops['bajo_reincidente']}).")
        if nivel == "bajo" and reinc is True:
            l.append("- Nota: se aplica la cuantía de reincidencia (sanción firme por el art. 20 en el año anterior a los hechos).")
        l.append("- Nota: posible inmovilización del vehículo (art. 104.1.d LSV: pruebas con resultado positivo); la retirada y el depósito solo en los casos del art. 105.")
        return l

    bloque_modelo = ["RESULTADO DEL CÁLCULO (verificado por código)", "Tipo de cálculo: alcoholemia en aire espirado",
                     f"Tasa medida (la menor): {_mgl(m)} mg/l · margen: {txt_margen} · tasa corregida: {_mgl(corr_ref)} mg/l"]
    if delito:
        motivo = (f"La menor de las mediciones ({_mgl(m)} mg/l) es de al menos {_mgl(pen['objetivo_min_medida_mgl'])} mg/l: tasa objetiva superior a 0,60 mg/l "
                  "tras descontar el error del etilómetro." if objetivo else
                  f"Tasa entre {_mgl(ba)} y {_mgl(bb)} mg/l con sintomatología o conducción anómala: influencia del alcohol en la conducción.")
        cab = [f"**Vía penal: presunto delito del art. 379.2 CP.** {motivo} Procede el atestado por presunto delito contra la seguridad vial.", "",
               "**Alternativa administrativa** (se rellena por si se desestima la vía penal; la penal prevalece):", ""]
        respuesta = "\n".join(cab + [f"**{resumen()}**", ""] + cuerpo())
        return _resultado("completo", "alcohol", etiqueta, "\n".join(bloque_modelo + [respuesta]), respuesta)
    extra = []
    if banda and sintomas is None:
        extra.append(f"- Nota penal: con tasas entre {_mgl(ba)} y {_mgl(bb)} mg/l puede haber delito (art. 379.2 CP) si hay sintomatología o conducción anómala "
                     "(maniobras anómalas, accidente, signos externos). Si los hay, indícalo y se tramita por vía penal.")
    if objetivo is False and m > bb:
        pass
    respuesta = "\n".join([f"**{resumen()}**", ""] + cuerpo() + extra)
    return _resultado("completo", "alcohol", etiqueta, "\n".join(bloque_modelo + [respuesta]), respuesta)


# ------------------------------------------------------------------
# Lector de alcoholemia directamente del texto (sin IA)
# ------------------------------------------------------------------
_RE_ALC_CONTEXTO = re.compile(r"alcohol|etil[oó]metro|alcoholemia|soplar|sopl[oó]|aire espirado|mg/?l|tasa", re.I)
_RE_ALC_NUM = re.compile(r"(?<![\d.,])(\d[.,]\d{1,3}|[.,]\d{1,3})(?![\d])\s*(mg/?l|mg|miligramos?|g/?l|gr/?l|gramos?|g\b)?", re.I)


def _lee_parcial_alcohol(texto: str) -> dict:
    t = texto or ""
    out = {"mediciones_mgl": [], "conductor": None, "reincidente": None, "sintomas": None, "tasa_corregida_mgl": None}
    mc = re.search(r"corregid[ao]s?\s*(?:de|a|es|:)?\s*(\d?[.,]\d{1,3})", t, re.I) or re.search(r"(\d?[.,]\d{1,3})\s*(?:mg/?l)?\s*(?:ya\s+)?corregid", t, re.I)
    corr_span = mc.span(1) if mc else None
    if mc:
        out["tasa_corregida_mgl"] = float(mc.group(1).replace(",", "."))
    sangre = False
    for m in _RE_ALC_NUM.finditer(t):
        if corr_span and m.span(1) == corr_span:
            continue
        unidad = (m.group(2) or "").lower()
        if unidad.startswith(("g/", "gr", "gramo")) or unidad == "g":
            sangre = True
            continue
        v = float(m.group(1).replace(",", "."))
        if 0 < v <= 5 and (unidad.startswith(("mg", "miligramo")) or re.search(r"mg/?l|miligramos|alcohol|etil|soplar|tasa|alcoholemia", t, re.I)):
            resto = t[m.end():m.end() + 12]
            if re.match(r"\s*(?:km|m\b|metros|kg|%|euros|€|h\b|horas)", resto, re.I):
                continue
            out["mediciones_mgl"].append(v)
    out["sangre"] = sangre
    low = t.lower()
    if re.search(r"profesional|novel|autob[uú]s|autocar|escolar|mercanc[ií]as peligrosas|urgencia|ambulancia|transporte especial", low):
        out["conductor"] = "profesional_novel"
    elif re.search(r"conductor general|\bgeneral\b|particular", low):
        out["conductor"] = "general"
    if re.search(r"reincident|sanci[oó]n firme|sancionad[oa]\s+(?:por\s+alcohol\s+)?(?:en|durante)\s+el\s+[uú]ltimo\s+a[ñn]o", low):
        out["reincidente"] = True
    elif re.search(r"sin antecedentes|primera vez|no (?:es )?reincident", low):
        out["reincidente"] = False
    if re.search(r"s[ií]ntomas|signos|inestab|aliento|maniobra|accidente|colisi[oó]n|conducci[oó]n an[oó]mala|zigzag|ojos", low) \
            and not re.search(r"sin s[ií]ntomas|sin signos|no present\w+ s[ií]ntomas", low):
        out["sintomas"] = True
    return out


def extraer_alcohol_conversacion(pregunta: str, previos_usuario: list, calculo_reciente: bool = False):
    """Devuelve el dict 'alcohol' para resolver_alcohol, o None si el mensaje no es un caso de alcohol en aire espirado."""
    t = pregunta or ""
    if re.search(r"km/?h|kilogramos|\bkg\b|\bMMA\b|tonelad", t, re.I):
        return None
    actual = _lee_parcial_alcohol(t) if _RE_ALC_CONTEXTO.search(t) else None
    if actual and actual["mediciones_mgl"] and not actual["sangre"]:
        base = actual
    elif calculo_reciente and previos_usuario:
        base = dict(actual or _lee_parcial_alcohol(t))
        for texto in reversed(previos_usuario[-3:]):
            ant = _lee_parcial_alcohol(texto)
            if ant["mediciones_mgl"] and not ant["sangre"]:
                for k, v in ant.items():
                    if base.get(k) in (None, False, []):
                        base[k] = v
                break
    else:
        return None
    if not base.get("mediciones_mgl") or base.get("sangre"):
        return None
    base = {k: v for k, v in base.items() if v not in (None, []) and k != "sangre"}
    return base


# ==================================================================
# Entrada única
# ==================================================================
def calcular(extraccion: dict, datos: dict) -> dict:
    tipo = (extraccion or {}).get("calculo")
    if datos is None or tipo not in ("tiempos", "masas", "dimensiones", "cargas", "velocidad", "alcohol"):
        return _resultado("sin_calculo", None, "", "")
    if tipo == "tiempos":
        return resolver_tiempos(extraccion.get("tiempos") or {}, datos)
    if tipo == "masas":
        return resolver_masas(extraccion.get("masas") or {}, datos)
    if tipo == "cargas":
        return resolver_cargas(extraccion.get("cargas") or {}, datos)
    if tipo == "velocidad":
        return resolver_velocidad(extraccion.get("velocidad") or {}, datos)
    if tipo == "alcohol":
        return resolver_alcohol(extraccion.get("alcohol") or {}, datos)
    return resolver_dimensiones(extraccion.get("dimensiones") or {}, datos)
