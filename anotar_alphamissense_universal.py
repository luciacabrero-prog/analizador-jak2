"""
ANOTADOR UNIVERSAL DE ALPHAMISSENSE

Uso desde la carpeta MEMORIAL:
    python ".\Alphamissense\anotar_alphamissense_universal.py" ".\Alphamissense\mutaciones.csv"

También puede ejecutarse sin indicar archivo:
    python ".\Alphamissense\anotar_alphamissense_universal.py"

En ese caso buscará "mutaciones.csv" en la misma carpeta del script.

COLUMNAS ADMITIDAS
------------------
El CSV puede contener cualquiera de estas combinaciones:

Opción recomendada:
- gen
- transcrito
- hgvs_c
- hgvs_p (opcional)
- mutacion (opcional)

Opción directa:
- variante_hgvs_consulta

Ejemplos válidos de variante_hgvs_consulta:
- ENST00000381652:c.1849G>T
- NM_004972.4:c.1849G>T
- JAK2:p.Val617Phe

El programa:
1. Lee variantes de cualquier gen humano.
2. Construye la consulta HGVS.
3. Consulta Ensembl VEP REST con la anotación AlphaMissense.
4. Selecciona la consecuencia transcriptómica más relevante.
5. Guarda un CSV anotado.
6. Genera una gráfica de puntuaciones y otra de clasificaciones cuando hay datos.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


ENSEMBL_REST = "https://rest.ensembl.org"
SPECIES = "homo_sapiens"
BATCH_SIZE = 100
PAUSA_ENTRE_LOTES = 0.35
TIMEOUT = 90


def texto_valido(valor: Any) -> bool:
    if valor is None:
        return False
    if isinstance(valor, float) and math.isnan(valor):
        return False
    return bool(str(valor).strip())


def limpiar_texto(valor: Any) -> str:
    return str(valor).strip() if texto_valido(valor) else ""


def quitar_version_transcrito(transcrito: str) -> str:
    return transcrito.split(".")[0] if transcrito else ""


def construir_hgvs(fila: pd.Series) -> tuple[str, str]:
    """
    Devuelve:
    - consulta HGVS para Ensembl
    - explicación del origen de la consulta
    """
    directa = limpiar_texto(fila.get("variante_hgvs_consulta"))
    if directa:
        return directa, "variante_hgvs_consulta"

    transcrito = limpiar_texto(fila.get("transcrito"))
    hgvs_c = limpiar_texto(fila.get("hgvs_c"))
    hgvs_p = limpiar_texto(fila.get("hgvs_p"))
    gen = limpiar_texto(fila.get("gen"))

    if transcrito and hgvs_c:
        return f"{transcrito}:{hgvs_c}", "transcrito + hgvs_c"

    if gen and hgvs_c:
        return f"{gen}:{hgvs_c}", "gen + hgvs_c"

    if gen and hgvs_p:
        return f"{gen}:{hgvs_p}", "gen + hgvs_p"

    return "", "sin datos HGVS suficientes"


def normalizar_clasificacion(valor: Any) -> str:
    texto = limpiar_texto(valor).lower()
    texto = texto.replace("_", " ").replace("-", " ")

    if not texto:
        return ""

    if "pathogenic" in texto or "patog" in texto:
        return "Probablemente patogénica"
    if "benign" in texto or "benign" in texto:
        return "Probablemente benigna"
    if "ambig" in texto or "uncertain" in texto or "indetermin" in texto:
        return "Ambigua"

    return limpiar_texto(valor)


def convertir_numero(valor: Any) -> float | None:
    if valor is None:
        return None

    if isinstance(valor, (int, float)):
        numero = float(valor)
        return numero if math.isfinite(numero) else None

    texto = str(valor).strip().replace(",", ".")

    coincidencia = re.search(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", texto)
    if coincidencia:
        try:
            return float(coincidencia.group(0))
        except ValueError:
            return None

    return None


def recorrer_claves(objeto: Any, ruta: str = ""):
    if isinstance(objeto, dict):
        for clave, valor in objeto.items():
            nueva_ruta = f"{ruta}.{clave}" if ruta else clave
            yield nueva_ruta, clave, valor
            yield from recorrer_claves(valor, nueva_ruta)

    elif isinstance(objeto, list):
        for indice, valor in enumerate(objeto):
            nueva_ruta = f"{ruta}[{indice}]"
            yield from recorrer_claves(valor, nueva_ruta)


def extraer_alphamissense_de_bloque(bloque: dict[str, Any]) -> tuple[float | None, str, str]:
    """
    Extrae AlphaMissense de forma tolerante a posibles cambios de nombres
    en la respuesta de Ensembl.
    """
    score = None
    prediccion = ""
    fuente = ""

    candidatos = []

    for ruta, clave, valor in recorrer_claves(bloque):
        clave_normalizada = clave.lower().replace("-", "_")
        ruta_normalizada = ruta.lower().replace("-", "_")

        if "alphamissense" not in clave_normalizada and "alphamissense" not in ruta_normalizada:
            continue

        candidatos.append((ruta, clave_normalizada, valor))

        if isinstance(valor, dict):
            for subclave, subvalor in valor.items():
                sub = subclave.lower()
                if score is None and ("score" in sub or "pathogenicity" in sub):
                    score = convertir_numero(subvalor)
                    fuente = ruta
                if not prediccion and (
                    "pred" in sub or "class" in sub or "label" in sub
                ):
                    prediccion = normalizar_clasificacion(subvalor)
                    fuente = ruta

        elif isinstance(valor, str):
            if score is None:
                score = convertir_numero(valor)

            if not prediccion:
                prediccion = normalizar_clasificacion(valor)

            fuente = ruta

        elif isinstance(valor, (int, float)):
            if score is None:
                score = convertir_numero(valor)
                fuente = ruta

    # Segundo intento: campos separados con nombres habituales
    if score is None or not prediccion:
        for ruta, clave, valor in recorrer_claves(bloque):
            clave_normalizada = clave.lower().replace("-", "_")

            if score is None and clave_normalizada in {
                "alphamissense_score",
                "am_score",
                "pathogenicity_score",
            }:
                score = convertir_numero(valor)
                fuente = ruta

            if not prediccion and clave_normalizada in {
                "alphamissense_prediction",
                "alphamissense_class",
                "am_pathogenicity",
                "am_class",
            }:
                prediccion = normalizar_clasificacion(valor)
                fuente = ruta

    # Clasificación por umbral solo si no viene dada por Ensembl
    if not prediccion and score is not None:
        if score < 0.34:
            prediccion = "Probablemente benigna"
        elif score > 0.564:
            prediccion = "Probablemente patogénica"
        else:
            prediccion = "Ambigua"

    return score, prediccion, fuente


def prioridad_consecuencia(consecuencia: dict[str, Any], transcrito_deseado: str) -> tuple:
    transcript_id = limpiar_texto(consecuencia.get("transcript_id"))
    deseado_sin_version = quitar_version_transcrito(transcrito_deseado)
    transcript_sin_version = quitar_version_transcrito(transcript_id)

    coincide_transcrito = int(
        bool(deseado_sin_version)
        and transcript_sin_version == deseado_sin_version
    )

    mane = int(
        texto_valido(consecuencia.get("mane_select"))
        or texto_valido(consecuencia.get("mane_plus_clinical"))
    )
    pick = int(consecuencia.get("pick") in {1, "1", True})
    canonical = int(consecuencia.get("canonical") in {1, "1", True})

    score, prediccion, _ = extraer_alphamissense_de_bloque(consecuencia)
    tiene_am = int(score is not None or bool(prediccion))

    return (
        coincide_transcrito,
        tiene_am,
        mane,
        pick,
        canonical,
    )


def seleccionar_consecuencia(
    resultado_vep: dict[str, Any],
    transcrito_deseado: str,
) -> dict[str, Any] | None:
    consecuencias = resultado_vep.get("transcript_consequences") or []

    if not consecuencias:
        return None

    consecuencias_ordenadas = sorted(
        consecuencias,
        key=lambda bloque: prioridad_consecuencia(
            bloque,
            transcrito_deseado,
        ),
        reverse=True,
    )

    return consecuencias_ordenadas[0]


def consultar_lote(hgvs_notations: list[str]) -> list[dict[str, Any]]:
    url = f"{ENSEMBL_REST}/vep/{SPECIES}/hgvs"

    parametros = {
        "AlphaMissense": 1,
        "canonical": 1,
        "mane": 1,
        "pick": 0,
        "hgvs": 1,
        "protein": 1,
        "transcript_version": 1,
        "ambiguous_hgvs": 1,
    }

    respuesta = requests.post(
        url,
        params=parametros,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"hgvs_notations": hgvs_notations},
        timeout=TIMEOUT,
    )

    if respuesta.status_code == 429:
        espera = int(respuesta.headers.get("Retry-After", "5"))
        print(f"Ensembl pide esperar {espera} segundos...")
        time.sleep(espera)
        return consultar_lote(hgvs_notations)

    respuesta.raise_for_status()
    contenido = respuesta.json()

    if not isinstance(contenido, list):
        raise RuntimeError(
            "La respuesta de Ensembl no tiene el formato esperado."
        )

    return contenido


def indexar_respuestas(
    respuestas: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    indice: dict[str, list[dict[str, Any]]] = {}

    for respuesta in respuestas:
        entrada = limpiar_texto(respuesta.get("input"))
        if entrada:
            indice.setdefault(entrada, []).append(respuesta)

    return indice


def anotar_dataframe(datos: pd.DataFrame) -> pd.DataFrame:
    datos = datos.copy()
    datos.columns = datos.columns.astype(str).str.strip()

    # Si el usuario vuelve a subir un CSV ya anotado, eliminamos primero
    # las columnas de salida antiguas para evitar nombres duplicados.
    columnas_generadas = {
        "origen_consulta_hgvs",
        "alphamissense_score",
        "alphamissense_clasificacion",
        "transcrito_resultado",
        "gen_resultado",
        "consecuencia_mas_severa",
        "hgvsc_resultado",
        "hgvsp_resultado",
        "canonical",
        "mane_select",
        "fuente_campo_alphamissense",
        "estado_anotacion",
    }

    columnas_a_eliminar = [
        columna
        for columna in datos.columns
        if columna in columnas_generadas
    ]

    if columnas_a_eliminar:
        datos = datos.drop(columns=columnas_a_eliminar)

    consultas = []
    origenes = []

    for _, fila in datos.iterrows():
        consulta, origen = construir_hgvs(fila)
        consultas.append(consulta)
        origenes.append(origen)

    datos["variante_hgvs_consulta"] = consultas
    datos["origen_consulta_hgvs"] = origenes

    consultas_validas = [
        consulta for consulta in dict.fromkeys(consultas) if consulta
    ]

    if not consultas_validas:
        raise ValueError(
            "No se pudo construir ninguna consulta HGVS. "
            "Incluye 'variante_hgvs_consulta' o las columnas "
            "'transcrito' + 'hgvs_c'."
        )

    todas_respuestas: list[dict[str, Any]] = []

    for inicio in range(0, len(consultas_validas), BATCH_SIZE):
        lote = consultas_validas[inicio : inicio + BATCH_SIZE]

        print(
            f"Consultando variantes {inicio + 1}-"
            f"{inicio + len(lote)} de {len(consultas_validas)}..."
        )

        try:
            todas_respuestas.extend(consultar_lote(lote))
        except requests.RequestException as error:
            print(f"Error en el lote: {error}")

        time.sleep(PAUSA_ENTRE_LOTES)

    indice = indexar_respuestas(todas_respuestas)
    anotaciones = []

    for _, fila in datos.iterrows():
        consulta = limpiar_texto(fila["variante_hgvs_consulta"])
        transcrito_deseado = limpiar_texto(fila.get("transcrito"))

        anotacion = {
            "alphamissense_score": pd.NA,
            "alphamissense_clasificacion": "Sin predicción",
            "transcrito_resultado": "",
            "gen_resultado": "",
            "consecuencia_mas_severa": "",
            "hgvsc_resultado": "",
            "hgvsp_resultado": "",
            "canonical": "",
            "mane_select": "",
            "fuente_campo_alphamissense": "",
            "estado_anotacion": "",
        }

        if not consulta:
            anotacion["estado_anotacion"] = "Consulta HGVS vacía"
            anotaciones.append(anotacion)
            continue

        candidatas = indice.get(consulta, [])

        if not candidatas:
            anotacion["estado_anotacion"] = (
                "Ensembl no devolvió resultados para esta consulta"
            )
            anotaciones.append(anotacion)
            continue

        # Si hay varias interpretaciones, escoger la de mayor prioridad
        mejores = []

        for resultado_vep in candidatas:
            consecuencia = seleccionar_consecuencia(
                resultado_vep,
                transcrito_deseado,
            )

            if consecuencia is not None:
                prioridad = prioridad_consecuencia(
                    consecuencia,
                    transcrito_deseado,
                )
                mejores.append((prioridad, resultado_vep, consecuencia))

        if not mejores:
            anotacion["estado_anotacion"] = (
                "Sin consecuencias transcriptómicas"
            )
            anotaciones.append(anotacion)
            continue

        _, resultado_vep, consecuencia = max(
            mejores,
            key=lambda elemento: elemento[0],
        )

        score, clasificacion, fuente = (
            extraer_alphamissense_de_bloque(consecuencia)
        )

        if score is None and not clasificacion:
            score, clasificacion, fuente = (
                extraer_alphamissense_de_bloque(resultado_vep)
            )

        anotacion.update(
            {
                "alphamissense_score": (
                    score if score is not None else pd.NA
                ),
                "alphamissense_clasificacion": (
                    clasificacion or "Sin predicción"
                ),
                "transcrito_resultado": limpiar_texto(
                    consecuencia.get("transcript_id")
                ),
                "gen_resultado": limpiar_texto(
                    consecuencia.get("gene_symbol")
                    or consecuencia.get("gene_id")
                ),
                "consecuencia_mas_severa": limpiar_texto(
                    resultado_vep.get("most_severe_consequence")
                ),
                "hgvsc_resultado": limpiar_texto(
                    consecuencia.get("hgvsc")
                ),
                "hgvsp_resultado": limpiar_texto(
                    consecuencia.get("hgvsp")
                ),
                "canonical": limpiar_texto(
                    consecuencia.get("canonical")
                ),
                "mane_select": limpiar_texto(
                    consecuencia.get("mane_select")
                ),
                "fuente_campo_alphamissense": fuente,
                "estado_anotacion": (
                    "Anotada"
                    if score is not None or clasificacion
                    else "Sin predicción AlphaMissense"
                ),
            }
        )

        anotaciones.append(anotacion)

    anotaciones_df = pd.DataFrame(anotaciones)

    return pd.concat(
        [datos.reset_index(drop=True), anotaciones_df],
        axis=1,
    )


def crear_graficas(datos: pd.DataFrame, carpeta_salida: Path) -> None:
    if plt is None:
        print(
            "Matplotlib no está instalado; se omiten las gráficas."
        )
        return

    scores = pd.to_numeric(
        datos["alphamissense_score"],
        errors="coerce",
    )

    datos_score = datos[scores.notna()].copy()
    datos_score["alphamissense_score"] = scores[scores.notna()]

    if not datos_score.empty:
        etiquetas = []

        for indice, fila in datos_score.iterrows():
            etiqueta = (
                limpiar_texto(fila.get("mutacion"))
                or limpiar_texto(fila.get("hgvs_p"))
                or limpiar_texto(fila.get("variante_hgvs_consulta"))
                or str(indice)
            )

            gen = limpiar_texto(fila.get("gen"))
            if gen:
                etiqueta = f"{gen} {etiqueta}"

            etiquetas.append(etiqueta)

        figura, eje = plt.subplots(
            figsize=(max(9, 0.8 * len(datos_score)), 6)
        )

        eje.bar(
            range(len(datos_score)),
            datos_score["alphamissense_score"],
        )
        eje.axhline(
            0.34,
            linestyle="--",
            linewidth=1,
            label="Límite benigno (0,34)",
        )
        eje.axhline(
            0.564,
            linestyle="--",
            linewidth=1,
            label="Límite patogénico (0,564)",
        )

        eje.set_xticks(range(len(etiquetas)))
        eje.set_xticklabels(
            etiquetas,
            rotation=60,
            ha="right",
        )
        eje.set_ylim(0, 1)
        eje.set_ylabel("Puntuación AlphaMissense")
        eje.set_title(
            "Puntuaciones AlphaMissense de las variantes"
        )
        eje.legend()
        figura.tight_layout()
        figura.savefig(
            carpeta_salida / "alphamissense_scores.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figura)

    conteo = (
        datos["alphamissense_clasificacion"]
        .fillna("Sin predicción")
        .value_counts()
    )

    if not conteo.empty:
        figura, eje = plt.subplots(figsize=(8, 5))
        conteo.plot(kind="bar", ax=eje)
        eje.set_xlabel("Clasificación")
        eje.set_ylabel("Número de variantes")
        eje.set_title("Clasificaciones AlphaMissense")
        eje.tick_params(axis="x", rotation=30)
        figura.tight_layout()
        figura.savefig(
            carpeta_salida / "alphamissense_clasificaciones.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figura)


def nombre_salida(ruta_entrada: Path) -> Path:
    return ruta_entrada.with_name(
        f"{ruta_entrada.stem}_AlphaMissense.csv"
    )


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Anota variantes missense de cualquier gen humano "
            "con AlphaMissense mediante Ensembl VEP."
        )
    )

    parser.add_argument(
        "archivo",
        nargs="?",
        default=None,
        help="Ruta del CSV de entrada.",
    )

    parser.add_argument(
        "--salida",
        default=None,
        help="Ruta opcional del CSV de salida.",
    )

    return parser.parse_args()


def main() -> None:
    argumentos = parsear_argumentos()
    carpeta_script = Path(__file__).resolve().parent

    ruta_entrada = (
        Path(argumentos.archivo).expanduser().resolve()
        if argumentos.archivo
        else carpeta_script / "mutaciones.csv"
    )

    if not ruta_entrada.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo de entrada:\n{ruta_entrada}\n\n"
            "Indica la ruta al ejecutar el script o guarda "
            "'mutaciones.csv' junto al programa."
        )

    ruta_salida = (
        Path(argumentos.salida).expanduser().resolve()
        if argumentos.salida
        else nombre_salida(ruta_entrada)
    )

    carpeta_salida = ruta_salida.parent
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo: {ruta_entrada}")
    datos = pd.read_csv(ruta_entrada)

    resultado = anotar_dataframe(datos)
    resultado.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8-sig",
    )

    crear_graficas(resultado, carpeta_salida)

    print("\nAnotación terminada.")
    print(f"CSV guardado en: {ruta_salida}")

    columnas_resumen = [
        columna
        for columna in [
            "gen",
            "mutacion",
            "hgvs_c",
            "hgvs_p",
            "alphamissense_score",
            "alphamissense_clasificacion",
            "estado_anotacion",
        ]
        if columna in resultado.columns
    ]

    if columnas_resumen:
        print("\nResumen:")
        print(resultado[columnas_resumen].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
