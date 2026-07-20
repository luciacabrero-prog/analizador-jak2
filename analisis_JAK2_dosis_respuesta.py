"""
ANÁLISIS DE MUTACIONES JAK2 Y CURVAS DOSIS-RESPUESTA

El archivo CSV de entrada debe contener estas columnas:
- mutacion: por ejemplo WT, V617F, R683G
- farmaco: por ejemplo Ruxolitinib, Fedratinib
- concentracion_nM: concentración del fármaco en nM
- viabilidad_porcentaje: viabilidad celular entre 0 y 100
- replica: identificador de la réplica experimental (opcional)

El programa:
1. Lee la base de datos.
2. Calcula media y desviación estándar por condición.
3. Ajusta una curva logística de cuatro parámetros.
4. Estima el IC50 para cada combinación mutación-fármaco.
5. Guarda tablas de IC50.
6. Genera curvas separadas para cada fármaco.
7. Genera una figura conjunta con todos los fármacos.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO_DATOS = "ejemplo_datos_JAK2.csv"
CARPETA_RESULTADOS = "resultados_JAK2"

COLUMNAS_OBLIGATORIAS = {
    "mutacion",
    "farmaco",
    "concentracion_nM",
    "viabilidad_porcentaje",
}


# ============================================================
# MODELO DOSIS-RESPUESTA
# ============================================================

def modelo_logistico_4p(concentracion, minimo, maximo, ic50, pendiente_hill):
    """
    Modelo logístico de cuatro parámetros.

    minimo:
        Respuesta mínima estimada.
    maximo:
        Respuesta máxima estimada.
    ic50:
        Concentración que produce la mitad del efecto.
    pendiente_hill:
        Pendiente de la curva.
    """
    concentracion = np.asarray(concentracion, dtype=float)

    return minimo + (maximo - minimo) / (
        1.0 + (concentracion / ic50) ** pendiente_hill
    )


def ajustar_curva(concentraciones, viabilidades):
    """
    Ajusta el modelo logístico de cuatro parámetros.

    Devuelve:
    - parámetros ajustados
    - matriz de covarianza
    - mensaje de error, si el ajuste falla
    """
    x = np.asarray(concentraciones, dtype=float)
    y = np.asarray(viabilidades, dtype=float)

    mascara = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
    )
    x = x[mascara]
    y = y[mascara]

    if len(np.unique(x)) < 4:
        return None, None, "Se necesitan al menos 4 concentraciones positivas diferentes."

    estimacion_inicial = [
        max(0.0, float(np.min(y))),
        min(120.0, float(np.max(y))),
        float(np.median(x)),
        1.0,
    ]

    limites_inferiores = [0.0, 50.0, max(np.min(x) / 1000.0, 1e-12), 0.05]
    limites_superiores = [100.0, 150.0, np.max(x) * 1000.0, 10.0]

    try:
        parametros, covarianza = curve_fit(
            modelo_logistico_4p,
            x,
            y,
            p0=estimacion_inicial,
            bounds=(limites_inferiores, limites_superiores),
            maxfev=50000,
        )
        return parametros, covarianza, None

    except (RuntimeError, ValueError, FloatingPointError) as error:
        return None, None, str(error)


# ============================================================
# LECTURA Y VALIDACIÓN
# ============================================================

def cargar_datos(ruta_csv):
    datos = pd.read_csv(ruta_csv)

    columnas_faltantes = COLUMNAS_OBLIGATORIAS - set(datos.columns)
    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas obligatorias en el CSV: "
            + ", ".join(sorted(columnas_faltantes))
        )

    datos = datos.copy()

    datos["mutacion"] = datos["mutacion"].astype(str).str.strip()
    datos["farmaco"] = datos["farmaco"].astype(str).str.strip()
    datos["concentracion_nM"] = pd.to_numeric(
        datos["concentracion_nM"], errors="coerce"
    )
    datos["viabilidad_porcentaje"] = pd.to_numeric(
        datos["viabilidad_porcentaje"], errors="coerce"
    )

    datos = datos.dropna(
        subset=[
            "mutacion",
            "farmaco",
            "concentracion_nM",
            "viabilidad_porcentaje",
        ]
    )

    datos = datos[
        (datos["concentracion_nM"] >= 0)
        & (datos["viabilidad_porcentaje"] >= 0)
        & (datos["viabilidad_porcentaje"] <= 150)
    ]

    if datos.empty:
        raise ValueError("No quedan datos válidos después de la limpieza.")

    return datos


def resumir_replicas(datos):
    """
    Calcula media, desviación estándar y número de réplicas.
    """
    resumen = (
        datos.groupby(
            ["farmaco", "mutacion", "concentracion_nM"],
            as_index=False,
        )
        .agg(
            viabilidad_media=("viabilidad_porcentaje", "mean"),
            desviacion_estandar=("viabilidad_porcentaje", "std"),
            numero_replicas=("viabilidad_porcentaje", "size"),
        )
    )

    resumen["desviacion_estandar"] = resumen[
        "desviacion_estandar"
    ].fillna(0.0)

    return resumen


# ============================================================
# CÁLCULO DE IC50
# ============================================================

def calcular_ic50(resumen):
    resultados = []

    grupos = resumen.groupby(["farmaco", "mutacion"])

    for (farmaco, mutacion), grupo in grupos:
        grupo = grupo.sort_values("concentracion_nM")

        parametros, covarianza, error = ajustar_curva(
            grupo["concentracion_nM"],
            grupo["viabilidad_media"],
        )

        if parametros is None:
            resultados.append(
                {
                    "farmaco": farmaco,
                    "mutacion": mutacion,
                    "ic50_nM": np.nan,
                    "respuesta_minima": np.nan,
                    "respuesta_maxima": np.nan,
                    "pendiente_hill": np.nan,
                    "r2": np.nan,
                    "estado_ajuste": f"Error: {error}",
                }
            )
            continue

        minimo, maximo, ic50, pendiente = parametros

        predichos = modelo_logistico_4p(
            grupo["concentracion_nM"].to_numpy(),
            *parametros,
        )
        observados = grupo["viabilidad_media"].to_numpy()

        suma_residuos = np.sum((observados - predichos) ** 2)
        suma_total = np.sum((observados - np.mean(observados)) ** 2)

        if suma_total > 0:
            r2 = 1.0 - suma_residuos / suma_total
        else:
            r2 = np.nan

        resultados.append(
            {
                "farmaco": farmaco,
                "mutacion": mutacion,
                "ic50_nM": ic50,
                "respuesta_minima": minimo,
                "respuesta_maxima": maximo,
                "pendiente_hill": pendiente,
                "r2": r2,
                "estado_ajuste": "Correcto",
            }
        )

    return pd.DataFrame(resultados)


# ============================================================
# GRÁFICAS
# ============================================================

def obtener_parametros(ic50_df, farmaco, mutacion):
    fila = ic50_df[
        (ic50_df["farmaco"] == farmaco)
        & (ic50_df["mutacion"] == mutacion)
        & (ic50_df["estado_ajuste"] == "Correcto")
    ]

    if fila.empty:
        return None

    fila = fila.iloc[0]

    return (
        fila["respuesta_minima"],
        fila["respuesta_maxima"],
        fila["ic50_nM"],
        fila["pendiente_hill"],
    )


def dibujar_curva_en_eje(eje, grupo, parametros, etiqueta):
    grupo = grupo.sort_values("concentracion_nM")

    x = grupo["concentracion_nM"].to_numpy(dtype=float)
    y = grupo["viabilidad_media"].to_numpy(dtype=float)
    error_y = grupo["desviacion_estandar"].to_numpy(dtype=float)

    concentraciones_positivas = x[x > 0]

    if len(concentraciones_positivas) == 0:
        return

    eje.errorbar(
        x,
        y,
        yerr=error_y,
        fmt="o",
        capsize=3,
        alpha=0.85,
    )

    if parametros is not None:
        x_curva = np.logspace(
            np.log10(np.min(concentraciones_positivas)),
            np.log10(np.max(concentraciones_positivas)),
            300,
        )

        y_curva = modelo_logistico_4p(x_curva, *parametros)

        ic50 = parametros[2]
        etiqueta_curva = f"{etiqueta} | IC50 = {ic50:.2f} nM"

        eje.plot(x_curva, y_curva, label=etiqueta_curva)
    else:
        eje.plot([], [], label=f"{etiqueta} | ajuste no disponible")


def crear_grafica_farmaco(farmaco, resumen, ic50_df, carpeta_salida):
    datos_farmaco = resumen[resumen["farmaco"] == farmaco]

    figura, eje = plt.subplots(figsize=(10, 7))

    for mutacion, grupo in datos_farmaco.groupby("mutacion"):
        parametros = obtener_parametros(ic50_df, farmaco, mutacion)
        dibujar_curva_en_eje(
            eje=eje,
            grupo=grupo,
            parametros=parametros,
            etiqueta=mutacion,
        )

    eje.set_xscale("log")
    eje.set_xlabel("Concentración del fármaco (nM)")
    eje.set_ylabel("Viabilidad celular (%)")
    eje.set_title(f"Curvas dosis-respuesta de {farmaco}")
    eje.set_ylim(-5, 115)
    eje.grid(True, which="both", alpha=0.25)
    eje.legend(fontsize=8)

    figura.tight_layout()

    nombre_seguro = (
        str(farmaco)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    ruta = carpeta_salida / f"curvas_{nombre_seguro}.png"
    figura.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(figura)


def crear_grafica_conjunta(resumen, ic50_df, carpeta_salida):
    farmacos = sorted(resumen["farmaco"].unique())
    numero_farmacos = len(farmacos)

    numero_columnas = 2
    numero_filas = int(np.ceil(numero_farmacos / numero_columnas))

    figura, ejes = plt.subplots(
        numero_filas,
        numero_columnas,
        figsize=(14, 5.5 * numero_filas),
        squeeze=False,
    )

    ejes_planos = ejes.flatten()

    for indice, farmaco in enumerate(farmacos):
        eje = ejes_planos[indice]
        datos_farmaco = resumen[resumen["farmaco"] == farmaco]

        for mutacion, grupo in datos_farmaco.groupby("mutacion"):
            parametros = obtener_parametros(ic50_df, farmaco, mutacion)
            dibujar_curva_en_eje(
                eje=eje,
                grupo=grupo,
                parametros=parametros,
                etiqueta=mutacion,
            )

        eje.set_xscale("log")
        eje.set_xlabel("Concentración (nM)")
        eje.set_ylabel("Viabilidad (%)")
        eje.set_title(farmaco)
        eje.set_ylim(-5, 115)
        eje.grid(True, which="both", alpha=0.25)
        eje.legend(fontsize=7)

    for indice_vacio in range(numero_farmacos, len(ejes_planos)):
        ejes_planos[indice_vacio].axis("off")

    figura.suptitle(
        "Curvas dosis-respuesta de mutaciones JAK2",
        fontsize=16,
        y=1.01,
    )
    figura.tight_layout()

    ruta = carpeta_salida / "todas_las_curvas_JAK2.png"
    figura.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(figura)


def crear_heatmap_ic50(ic50_df, carpeta_salida):
    tabla = ic50_df.pivot(
        index="mutacion",
        columns="farmaco",
        values="ic50_nM",
    )

    if tabla.empty:
        return

    figura, eje = plt.subplots(
        figsize=(
            max(8, 2 * len(tabla.columns)),
            max(5, 0.8 * len(tabla.index)),
        )
    )

    imagen = eje.imshow(
        np.log10(tabla.to_numpy(dtype=float)),
        aspect="auto",
    )

    eje.set_xticks(range(len(tabla.columns)))
    eje.set_xticklabels(tabla.columns, rotation=45, ha="right")
    eje.set_yticks(range(len(tabla.index)))
    eje.set_yticklabels(tabla.index)

    eje.set_title("IC50 por mutación JAK2 y fármaco")
    eje.set_xlabel("Fármaco")
    eje.set_ylabel("Mutación")

    barra = figura.colorbar(imagen, ax=eje)
    barra.set_label("log10(IC50 en nM)")

    for fila in range(len(tabla.index)):
        for columna in range(len(tabla.columns)):
            valor = tabla.iloc[fila, columna]
            if np.isfinite(valor):
                eje.text(
                    columna,
                    fila,
                    f"{valor:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    figura.tight_layout()
    figura.savefig(
        carpeta_salida / "mapa_calor_IC50.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figura)


# ============================================================
# TABLAS
# ============================================================

def guardar_tablas_por_farmaco(ic50_df, carpeta_salida):
    carpeta_tablas = carpeta_salida / "tablas_por_farmaco"
    carpeta_tablas.mkdir(parents=True, exist_ok=True)

    for farmaco, tabla in ic50_df.groupby("farmaco"):
        tabla = tabla.sort_values("ic50_nM")

        nombre_seguro = (
            str(farmaco)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        tabla.to_csv(
            carpeta_tablas / f"tabla_IC50_{nombre_seguro}.csv",
            index=False,
        )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    ruta_script = Path(__file__).resolve().parent
    ruta_datos = ruta_script / ARCHIVO_DATOS
    carpeta_salida = ruta_script / CARPETA_RESULTADOS
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo datos desde: {ruta_datos}")

    datos = cargar_datos(ruta_datos)
    resumen = resumir_replicas(datos)
    resultados_ic50 = calcular_ic50(resumen)

    datos.to_csv(
        carpeta_salida / "datos_limpios.csv",
        index=False,
    )
    resumen.to_csv(
        carpeta_salida / "datos_resumidos.csv",
        index=False,
    )
    resultados_ic50.to_csv(
        carpeta_salida / "tabla_general_IC50.csv",
        index=False,
    )

    guardar_tablas_por_farmaco(
        resultados_ic50,
        carpeta_salida,
    )

    for farmaco in sorted(resumen["farmaco"].unique()):
        crear_grafica_farmaco(
            farmaco,
            resumen,
            resultados_ic50,
            carpeta_salida,
        )

    crear_grafica_conjunta(
        resumen,
        resultados_ic50,
        carpeta_salida,
    )

    crear_heatmap_ic50(
        resultados_ic50,
        carpeta_salida,
    )

    print("\nAnálisis terminado.")
    print(f"Resultados guardados en: {carpeta_salida}")
    print("\nResumen de IC50:")
    print(
        resultados_ic50[
            ["farmaco", "mutacion", "ic50_nM", "r2", "estado_ajuste"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
