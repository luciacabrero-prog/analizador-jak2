from io import BytesIO
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit


st.set_page_config(
    page_title="Analizador JAK2",
    page_icon="🧬",
    layout="wide",
)

COLUMNAS_OBLIGATORIAS = {
    "mutacion",
    "farmaco",
    "concentracion_nM",
    "viabilidad_porcentaje",
}


def modelo_logistico_4p(concentracion, minimo, maximo, ic50, pendiente_hill):
    concentracion = np.asarray(concentracion, dtype=float)
    return minimo + (maximo - minimo) / (
        1.0 + (concentracion / ic50) ** pendiente_hill
    )


def ajustar_curva(concentraciones, viabilidades):
    x = np.asarray(concentraciones, dtype=float)
    y = np.asarray(viabilidades, dtype=float)

    mascara = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[mascara]
    y = y[mascara]

    if len(np.unique(x)) < 4:
        return None, None, (
            "Se necesitan al menos 4 concentraciones positivas diferentes."
        )

    estimacion_inicial = [
        max(0.0, float(np.min(y))),
        min(120.0, float(np.max(y))),
        float(np.median(x)),
        1.0,
    ]

    limites_inferiores = [
        0.0,
        50.0,
        max(np.min(x) / 1000.0, 1e-12),
        0.05,
    ]
    limites_superiores = [
        100.0,
        150.0,
        np.max(x) * 1000.0,
        10.0,
    ]

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


def cargar_datos(archivo):
    datos = pd.read_csv(archivo)

    datos.columns = datos.columns.astype(str).str.strip()

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
        datos["concentracion_nM"],
        errors="coerce",
    )
    datos["viabilidad_porcentaje"] = pd.to_numeric(
        datos["viabilidad_porcentaje"],
        errors="coerce",
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
        raise ValueError(
            "No quedan datos válidos después de la limpieza."
        )

    return datos


def resumir_replicas(datos):
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

    resumen["desviacion_estandar"] = (
        resumen["desviacion_estandar"].fillna(0.0)
    )

    return resumen


def calcular_ic50(resumen):
    resultados = []

    for (farmaco, mutacion), grupo in resumen.groupby(
        ["farmaco", "mutacion"]
    ):
        grupo = grupo.sort_values("concentracion_nM")

        parametros, _, error = ajustar_curva(
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

        r2 = (
            1.0 - suma_residuos / suma_total
            if suma_total > 0
            else np.nan
        )

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

        eje.plot(
            x_curva,
            y_curva,
            label=f"{etiqueta} | IC50 = {ic50:.2f} nM",
        )
    else:
        eje.plot(
            [],
            [],
            label=f"{etiqueta} | ajuste no disponible",
        )


def crear_figura_farmaco(farmaco, resumen, ic50_df):
    datos_farmaco = resumen[resumen["farmaco"] == farmaco]

    figura, eje = plt.subplots(figsize=(10, 7))

    for mutacion, grupo in datos_farmaco.groupby("mutacion"):
        parametros = obtener_parametros(
            ic50_df,
            farmaco,
            mutacion,
        )
        dibujar_curva_en_eje(
            eje,
            grupo,
            parametros,
            mutacion,
        )

    eje.set_xscale("log")
    eje.set_xlabel("Concentración del fármaco (nM)")
    eje.set_ylabel("Viabilidad celular (%)")
    eje.set_title(f"Curvas dosis-respuesta de {farmaco}")
    eje.set_ylim(-5, 115)
    eje.grid(True, which="both", alpha=0.25)
    eje.legend(fontsize=8)
    figura.tight_layout()

    return figura


def crear_figura_conjunta(resumen, ic50_df):
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
            parametros = obtener_parametros(
                ic50_df,
                farmaco,
                mutacion,
            )
            dibujar_curva_en_eje(
                eje,
                grupo,
                parametros,
                mutacion,
            )

        eje.set_xscale("log")
        eje.set_xlabel("Concentración (nM)")
        eje.set_ylabel("Viabilidad (%)")
        eje.set_title(farmaco)
        eje.set_ylim(-5, 115)
        eje.grid(True, which="both", alpha=0.25)
        eje.legend(fontsize=7)

    for indice in range(numero_farmacos, len(ejes_planos)):
        ejes_planos[indice].axis("off")

    figura.suptitle(
        "Curvas dosis-respuesta de mutaciones JAK2",
        fontsize=16,
        y=1.01,
    )
    figura.tight_layout()

    return figura


def crear_figura_heatmap(ic50_df):
    tabla = ic50_df.pivot(
        index="mutacion",
        columns="farmaco",
        values="ic50_nM",
    )

    tabla_valida = tabla.where(tabla > 0)

    figura, eje = plt.subplots(
        figsize=(
            max(8, 2 * len(tabla.columns)),
            max(5, 0.8 * len(tabla.index)),
        )
    )

    imagen = eje.imshow(
        np.log10(tabla_valida.to_numpy(dtype=float)),
        aspect="auto",
    )

    eje.set_xticks(range(len(tabla.columns)))
    eje.set_xticklabels(
        tabla.columns,
        rotation=45,
        ha="right",
    )
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

    return figura


def figura_a_png(figura):
    buffer = BytesIO()
    figura.savefig(
        buffer,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    buffer.seek(0)
    return buffer


st.title("🧬 Analizador de curvas dosis-respuesta JAK2")

st.write(
    "Sube un archivo CSV y la aplicación calculará medias, "
    "desviaciones estándar, curvas dosis-respuesta, IC50 y R²."
)

with st.expander("Formato obligatorio del CSV"):
    st.write(
        "El archivo debe contener las columnas: "
        "`mutacion`, `farmaco`, `concentracion_nM` y "
        "`viabilidad_porcentaje`. La columna `replica` es opcional."
    )

archivo = st.file_uploader(
    "Selecciona el archivo CSV",
    type=["csv"],
)

if archivo is not None:
    try:
        datos = cargar_datos(archivo)

        st.success("Archivo cargado y validado correctamente.")

        st.subheader("Datos limpios")
        st.dataframe(
            datos,
            use_container_width=True,
        )

        st.write(
            f"Filas válidas: **{len(datos)}** · "
            f"Fármacos: **{datos['farmaco'].nunique()}** · "
            f"Mutaciones: **{datos['mutacion'].nunique()}**"
        )

        farmacos = sorted(datos["farmaco"].unique())

        farmacos_seleccionados = st.multiselect(
            "Selecciona los fármacos que quieres analizar",
            farmacos,
            default=farmacos,
        )

        if st.button("Analizar datos", type="primary"):
            if not farmacos_seleccionados:
                st.warning(
                    "Selecciona al menos un fármaco."
                )
                st.stop()

            datos_filtrados = datos[
                datos["farmaco"].isin(farmacos_seleccionados)
            ]

            with st.spinner("Calculando curvas e IC50..."):
                resumen = resumir_replicas(datos_filtrados)
                resultados_ic50 = calcular_ic50(resumen)

            st.success("Análisis terminado.")

            st.subheader("Tabla general de IC50")

            tabla_mostrada = resultados_ic50.copy()
            columnas_numericas = [
                "ic50_nM",
                "respuesta_minima",
                "respuesta_maxima",
                "pendiente_hill",
                "r2",
            ]
            tabla_mostrada[columnas_numericas] = (
                tabla_mostrada[columnas_numericas].round(3)
            )

            st.dataframe(
                tabla_mostrada,
                use_container_width=True,
            )

            st.download_button(
                "Descargar tabla general de IC50",
                resultados_ic50.to_csv(index=False).encode("utf-8"),
                file_name="tabla_general_IC50.csv",
                mime="text/csv",
            )

            st.download_button(
                "Descargar datos resumidos",
                resumen.to_csv(index=False).encode("utf-8"),
                file_name="datos_resumidos.csv",
                mime="text/csv",
            )

            st.header("Curvas por fármaco")

            for farmaco in farmacos_seleccionados:
                figura = crear_figura_farmaco(
                    farmaco,
                    resumen,
                    resultados_ic50,
                )

                st.subheader(farmaco)
                st.pyplot(figura)

                st.download_button(
                    f"Descargar gráfica de {farmaco}",
                    data=figura_a_png(figura),
                    file_name=f"curvas_{farmaco}.png",
                    mime="image/png",
                    key=f"descarga_{farmaco}",
                )

                plt.close(figura)

            st.header("Figura conjunta")

            figura_conjunta = crear_figura_conjunta(
                resumen,
                resultados_ic50,
            )
            st.pyplot(figura_conjunta)

            st.download_button(
                "Descargar figura conjunta",
                data=figura_a_png(figura_conjunta),
                file_name="todas_las_curvas_JAK2.png",
                mime="image/png",
            )

            plt.close(figura_conjunta)

            st.header("Mapa de calor de IC50")

            figura_heatmap = crear_figura_heatmap(
                resultados_ic50
            )
            st.pyplot(figura_heatmap)

            st.download_button(
                "Descargar mapa de calor",
                data=figura_a_png(figura_heatmap),
                file_name="mapa_calor_IC50.png",
                mime="image/png",
            )

            plt.close(figura_heatmap)

            st.header("Interpretación básica")

            for _, fila in resultados_ic50.iterrows():
                if fila["estado_ajuste"] != "Correcto":
                    st.warning(
                        f"{fila['farmaco']} – {fila['mutacion']}: "
                        f"{fila['estado_ajuste']}"
                    )
                    continue

                if np.isfinite(fila["r2"]):
                    if fila["r2"] >= 0.9:
                        calidad = "ajuste muy bueno"
                    elif fila["r2"] >= 0.75:
                        calidad = "ajuste aceptable"
                    else:
                        calidad = "ajuste bajo; conviene revisar los datos"
                else:
                    calidad = "R² no disponible"

                st.write(
                    f"**{fila['farmaco']} – {fila['mutacion']}**: "
                    f"IC50 = **{fila['ic50_nM']:.2f} nM**; "
                    f"{calidad}."
                )

    except Exception as error:
        st.error(f"No se pudo realizar el análisis: {error}")


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)
