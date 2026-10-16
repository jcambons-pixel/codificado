# -*- coding: utf-8 -*-
"""
Búsqueda léxica (BM25 simplificado) sobre las fichas de tráfico y transporte.

Complementa a Pinecone: la búsqueda semántica falla con fichas muy cortas o genéricas
(p. ej. "carga que sobresale de la proyección en planta"), mientras que las palabras clave
("sobresale", "carga", "delante") las encuentran siempre. No necesita red ni claves.
"""
import math
import re
import unicodedata
from collections import Counter

STOP = set("""
de la el los las un una unos unas y o u e en con sin por para que se su sus al del lo como mas pero si no es son ser
esta este estos estas ese esa esos esas ha han hay muy ya tambien mismo misma cuando donde segun sobre entre
""".split())


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _tokens(texto: str) -> list:
    out = []
    for w in re.findall(r"[a-z0-9]+", _norm(texto)):
        if len(w) < 3 or w in STOP:
            continue
        out.append(w[:6])  # raíz tosca: sobresale/sobresalga -> "sobres", delante/delantera -> "delant"
    return out


class Indice:
    def __init__(self, docs: list):
        """docs: lista de (clave, texto)."""
        self.claves = [c for c, _ in docs]
        self.tf = [Counter(_tokens(t)) for _, t in docs]
        self.largo = [sum(c.values()) for c in self.tf]
        self.medio = (sum(self.largo) / len(self.largo)) if self.largo else 1.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.tf)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def buscar(self, consulta: str, k: int = 3) -> list:
        q = set(_tokens(consulta))
        puntuaciones = []
        for i, c in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = c.get(w, 0)
                if f:
                    s += self.idf[w] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * self.largo[i] / self.medio))
            if s > 0:
                puntuaciones.append((s, i))
        puntuaciones.sort(reverse=True)
        return [self.claves[i] for _, i in puntuaciones[:k]]


# ------------------------------------------------------------------
# Construcción de índices a partir de los JSON de preparar_datos.py
# ------------------------------------------------------------------
def construir_indices(datos: dict) -> dict:
    trafico = datos["trafico"]
    docs_tr = [(f["id"], f"{f['hecho']} {f['hecho']} {f['comentario']}") for f in trafico]
    hojas = datos["baremo"]["fichas"]
    docs_tp = [(cod, f"{f['contexto']} {f['concepto']} {f['concepto']}") for cod, f in hojas.items()]
    return {
        "trafico": Indice(docs_tr),
        "transporte": Indice(docs_tp),
        "_trafico_por_id": {f["id"]: f for f in trafico},
    }


def texto_ficha_trafico(f: dict) -> str:
    art = f"{f['art_reg']} / {f['art_lsv']}" if f["art_lsv"] else f["art_reg"]
    ap = f"{f['apartado_reg']} / {f['apartado_lsv']}" if f["apartado_lsv"] else f["apartado_reg"]
    red = f" (reducida: {f['multa_reducida']})" if f["multa_reducida"] is not None else ""
    cal = f["calificacion"] + (f" {f['art_sancionador']}" if f["art_sancionador"] else "")
    return (
        f"# Norma: {f['norma']}\nArtículo: {art}\nApartado: {ap}\nOpción: {f['opcion']}\n"
        f"Puntos: {f['puntos']}\nCalificación: {cal}\nHecho infringido: {f['hecho']}\n"
        f"Multa: {f['multa']}{red}\nResponsable: {f['responsable']}\nComentario: {f['comentario']}"
    )


def texto_ficha_transporte(f: dict) -> str:
    return (
        f"## Sanción - Código {f['cod']}\n**Texto:** Concepto: {f['concepto']} | Infracción: {f['norma_infringida']} | "
        f"Sanción: {f['norma_sancionadora']} | Cuantía: {f['cuantia_raw']}"
    )


def buscar_lexico(indices: dict, datos: dict, modo: str, consultas: list, k: int = 3) -> list:
    """Devuelve una lista por consulta, con tuplas (clave_dedupe, texto_ficha)."""
    resultado = []
    for q in consultas:
        lista = []
        if modo == "MULTA_TRAFICO":
            for fid in indices["trafico"].buscar(q, k):
                f = indices["_trafico_por_id"][fid]
                clave = _norm(re.sub(r"\s+", " ", f["hecho"]))[:70]
                lista.append((clave, texto_ficha_trafico(f)))
        elif modo == "MULTA_TRANSPORTE":
            for cod in indices["transporte"].buscar(q, k):
                f = datos["baremo"]["fichas"][cod]
                lista.append((f"codigo {cod.lower()}", texto_ficha_transporte(f)))
        resultado.append(lista)
    return resultado


def fusionar(fragmentos_pinecone: list, listas_lexicas: list, limite: int) -> list:
    """Reparte por turnos entre los fragmentos de Pinecone y las listas léxicas, sin repetir fichas."""
    reunidos = []  # (texto, texto_normalizado)

    def ya_esta(clave: str) -> bool:
        return bool(clave) and any(clave in n for _, n in reunidos)

    def meter(texto: str, clave: str = ""):
        if texto in [t for t, _ in reunidos] or ya_esta(clave):
            return
        reunidos.append((texto, _norm(re.sub(r"\s+", " ", texto))))

    colas = [[(t, "") for t in fragmentos_pinecone]] + [[(t, c) for c, t in lst] for lst in listas_lexicas]
    for i in range(max((len(c) for c in colas), default=0)):
        for cola in colas:
            if i < len(cola):
                meter(cola[i][0], cola[i][1])
            if len(reunidos) >= limite:
                return [t for t, _ in reunidos]
    return [t for t, _ in reunidos]
