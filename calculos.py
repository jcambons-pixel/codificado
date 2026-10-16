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
        return {"baremo": baremo, "trafico": trafico}
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
        f"{indent}- Infracción: {f['hecho'][:400]}",
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


def _resultado(estado, tipo, etiqueta, texto):
    return {"estado": estado, "tipo": tipo, "etiqueta": etiqueta, "texto": texto}


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
# Entrada única
# ==================================================================
def calcular(extraccion: dict, datos: dict) -> dict:
    tipo = (extraccion or {}).get("calculo")
    if datos is None or tipo not in ("tiempos", "masas", "dimensiones"):
        return _resultado("sin_calculo", None, "", "")
    if tipo == "tiempos":
        return resolver_tiempos(extraccion.get("tiempos") or {}, datos)
    if tipo == "masas":
        return resolver_masas(extraccion.get("masas") or {}, datos)
    return resolver_dimensiones(extraccion.get("dimensiones") or {}, datos)
