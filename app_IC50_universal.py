from io import BytesIO
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

st.set_page_config(page_title="Analizador universal de IC50", page_icon="🧪", layout="wide")

COLUMNAS_OBLIGATORIAS = {
    "mutacion", "farmaco", "concentracion_nM", "viabilidad_porcentaje"
}


def modelo_logistico_4p(concentracion, minimo, maximo, ic50, pendiente_hill):
    concentracion = np.asarray(concentracion, dtype=float)
    return minimo + (maximo - minimo) / (
        1.0 + (concentracion / ic50) ** pendiente_hill
    )


def limpiar_y_validar_datos(archivo):
    datos = pd.read_csv(archivo)
    datos.columns = datos.columns.astype(str).str.strip()

    faltantes = COLUMNAS_OBLIGATORIAS - set(datos.columns)
    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias: " + ", ".join(sorted(faltantes))
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

    antes = len(datos)
    datos = datos.dropna(
        subset=[
            "mutacion", "farmaco", "concentracion_nM", "viabilidad_porcentaje"
        ]
    )
    datos = datos[
        (datos["concentracion_nM"] >= 0)
        & (datos["viabilidad_porcentaje"] >= 0)
        & (datos["viabilidad_porcentaje"] <= 150)
    ]

    if datos.empty:
        raise ValueError("No quedan datos válidos tras la limpieza.")

    return datos, antes - len(datos)


def resumir_replicas(datos):
    resumen = (
        datos.groupby(
            ["farmaco", "mutacion", "concentracion_nM"], as_index=False
        )
        .agg(
            viabilidad_media=("viabilidad_porcentaje", "mean"),
            desviacion_estandar=("viabilidad_porcentaje", "std"),
            numero_replicas=("viabilidad_porcentaje", "size"),
        )
        .sort_values(["farmaco", "mutacion", "concentracion_nM"])
    )
    resumen["desviacion_estandar"] = resumen["desviacion_estandar"].fillna(0.0)
    return resumen


def ajustar_4p(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mascara = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[mascara]
    y = y[mascara]

    if len(np.unique(x)) < 4:
        return None, "Menos de 4 concentraciones positivas."

    p0 = [
        max(0.0, float(np.min(y))),
        min(120.0, float(np.max(y))),
        float(np.median(x)),
        1.0,
    ]
    lower = [0.0, 0.0, max(np.min(x) / 1000.0, 1e-12), 0.05]
    upper = [150.0, 150.0, np.max(x) * 1000.0, 10.0]

    try:
        parametros, _ = curve_fit(
            modelo_logistico_4p,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=100000,
        )
        return parametros, None
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return None, str(error)


def interpolar_ic50_log(x, y):
    orden = np.argsort(x)
    x = np.asarray(x, dtype=float)[orden]
    y = np.asarray(y, dtype=float)[orden]

    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        if y1 == y2:
            continue
        if (y1 - 50) * (y2 - 50) <= 0:
            log_x1 = np.log10(x[i])
            log_x2 = np.log10(x[i + 1])
            fraccion = (50 - y1) / (y2 - y1)
            return float(10 ** (log_x1 + fraccion * (log_x2 - log_x1)))
    return np.nan


def analizar_grupo(grupo):
    grupo = grupo.sort_values("concentracion_nM")
    x = grupo["concentracion_nM"].to_numpy(dtype=float)
    y = grupo["viabilidad_media"].to_numpy(dtype=float)

    mascara = x > 0
    x_pos = x[mascara]
    y_pos = y[mascara]
    n = len(np.unique(x_pos))

    base = {
        "numero_concentraciones_positivas": n,
        "ic50_nM": np.nan,
        "respuesta_minima": np.nan,
        "respuesta_maxima": np.nan,
        "pendiente_hill": np.nan,
        "r2": np.nan,
        "metodo": "No calculable",
        "fiabilidad": "Insuficiente",
        "estado_ajuste": "",
    }

    if n == 0:
        base["estado_ajuste"] = "No hay concentraciones positivas."
        return base, None
    if n == 1:
        base["estado_ajuste"] = "Solo hay una concentración positiva."
        return base, None

    if n >= 4:
        parametros, error = ajustar_4p(x_pos, y_pos)
        if parametros is not None:
            minimo, maximo, ic50, pendiente = parametros
            pred = modelo_logistico_4p(x_pos, *parametros)
            ss_res = np.sum((y_pos - pred) ** 2)
            ss_tot = np.sum((y_pos - np.mean(y_pos)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            fiabilidad = (
                "Alta" if np.isfinite(r2) and r2 >= 0.90
                else "Moderada" if np.isfinite(r2) and r2 >= 0.75
                else "Baja"
            )
            return {
                **base,
                "ic50_nM": ic50,
                "respuesta_minima": minimo,
                "respuesta_maxima": maximo,
                "pendiente_hill": pendiente,
                "r2": r2,
                "metodo": "Curva logística 4P",
                "fiabilidad": fiabilidad,
                "estado_ajuste": "Correcto",
            }, parametros
        base["estado_ajuste"] = f"Falló el ajuste 4P: {error}"

    ic50 = interpolar_ic50_log(x_pos, y_pos)
    if np.isfinite(ic50):
        return {
            **base,
            "ic50_nM": ic50,
            "metodo": "Interpolación logarítmica",
            "fiabilidad": "Aproximada",
            "estado_ajuste": (
                "IC50 estimada entre dos concentraciones que rodean el 50%."
            ),
        }, None

    if np.all(y_pos > 50):
        estado = (
            "No se alcanzó el 50% de viabilidad. La IC50 es mayor que la "
            "concentración máxima ensayada."
        )
    elif np.all(y_pos < 50):
        estado = (
            "Todas las respuestas están por debajo del 50%. La IC50 es menor "
            "que la concentración mínima ensayada."
        )
    else:
        estado = "No se pudo identificar un cruce fiable del 50%."

    return {**base, "estado_ajuste": estado}, None


def calcular_resultados(resumen):
    resultados = []
    parametros = {}
    for (farmaco, mutacion), grupo in resumen.groupby(["farmaco", "mutacion"]):
        resultado, params = analizar_grupo(grupo)
        resultados.append({"farmaco": farmaco, "mutacion": mutacion, **resultado})
        parametros[(farmaco, mutacion)] = params
    return pd.DataFrame(resultados), parametros


def dibujar_grupo(eje, grupo, resultado, parametros, etiqueta):
    grupo = grupo.sort_values("concentracion_nM")
    x = grupo["concentracion_nM"].to_numpy(dtype=float)
    y = grupo["viabilidad_media"].to_numpy(dtype=float)
    err = grupo["desviacion_estandar"].to_numpy(dtype=float)

    mascara = x > 0
    x, y, err = x[mascara], y[mascara], err[mascara]
    if len(x) == 0:
        return

    eje.errorbar(x, y, yerr=err, fmt="o", capsize=3, alpha=0.85)

    if parametros is not None:
        x_curva = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 300)
        y_curva = modelo_logistico_4p(x_curva, *parametros)
        eje.plot(
            x_curva,
            y_curva,
            label=f"{etiqueta} | IC50={resultado['ic50_nM']:.2f} nM | 4P",
        )
    else:
        texto = (
            f"IC50≈{resultado['ic50_nM']:.2f} nM | interpolación"
            if np.isfinite(resultado["ic50_nM"])
            else "IC50 no estimable"
        )
        eje.plot(x, y, linestyle="--", alpha=0.8, label=f"{etiqueta} | {texto}")


def crear_figura_farmaco(farmaco, resumen, resultados, parametros_por_grupo):
    datos_farmaco = resumen[resumen["farmaco"] == farmaco]
    figura, eje = plt.subplots(figsize=(10, 7))

    for mutacion, grupo in datos_farmaco.groupby("mutacion"):
        fila = resultados[
            (resultados["farmaco"] == farmaco)
            & (resultados["mutacion"] == mutacion)
        ].iloc[0]
        dibujar_grupo(
            eje,
            grupo,
            fila,
            parametros_por_grupo.get((farmaco, mutacion)),
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


def crear_figura_conjunta(resumen, resultados, parametros_por_grupo):
    farmacos = sorted(resumen["farmaco"].unique())
    columnas = 2
    filas = max(1, int(np.ceil(len(farmacos) / columnas)))
    figura, ejes = plt.subplots(
        filas, columnas, figsize=(14, 5.5 * filas), squeeze=False
    )
    ejes = ejes.flatten()

    for i, farmaco in enumerate(farmacos):
        eje = ejes[i]
        datos_farmaco = resumen[resumen["farmaco"] == farmaco]
        for mutacion, grupo in datos_farmaco.groupby("mutacion"):
            fila = resultados[
                (resultados["farmaco"] == farmaco)
                & (resultados["mutacion"] == mutacion)
            ].iloc[0]
            dibujar_grupo(
                eje,
                grupo,
                fila,
                parametros_por_grupo.get((farmaco, mutacion)),
                mutacion,
            )
        eje.set_xscale("log")
        eje.set_xlabel("Concentración (nM)")
        eje.set_ylabel("Viabilidad (%)")
        eje.set_title(farmaco)
        eje.set_ylim(-5, 115)
        eje.grid(True, which="both", alpha=0.25)
        eje.legend(fontsize=7)

    for i in range(len(farmacos), len(ejes)):
        ejes[i].axis("off")

    figura.suptitle("Curvas dosis-respuesta", fontsize=16, y=1.01)
    figura.tight_layout()
    return figura


def crear_heatmap(resultados):
    tabla = resultados.pivot(index="mutacion", columns="farmaco", values="ic50_nM")
    if tabla.empty:
        return None

    tabla_valida = tabla.where(tabla > 0)
    figura, eje = plt.subplots(
        figsize=(max(8, 2 * len(tabla.columns)), max(5, 0.8 * len(tabla.index)))
    )
    imagen = eje.imshow(np.log10(tabla_valida.to_numpy(dtype=float)), aspect="auto")
    eje.set_xticks(range(len(tabla.columns)))
    eje.set_xticklabels(tabla.columns, rotation=45, ha="right")
    eje.set_yticks(range(len(tabla.index)))
    eje.set_yticklabels(tabla.index)
    eje.set_title("IC50 por mutación y fármaco")
    eje.set_xlabel("Fármaco")
    eje.set_ylabel("Mutación")
    barra = figura.colorbar(imagen, ax=eje)
    barra.set_label("log10(IC50 en nM)")

    for fila in range(len(tabla.index)):
        for columna in range(len(tabla.columns)):
            valor = tabla.iloc[fila, columna]
            if np.isfinite(valor):
                eje.text(columna, fila, f"{valor:.1f}", ha="center", va="center", fontsize=8)

    figura.tight_layout()
    return figura


def figura_a_png(figura):
    buffer = BytesIO()
    figura.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer


st.title("🧪 Analizador universal de curvas dosis-respuesta e IC50")
st.write(
    "Acepta distintos fármacos, mutaciones, números de réplicas y números de concentraciones."
)
st.info(
    "Con 4 o más concentraciones positivas intenta una curva logística 4P. "
    "Con 2 o 3 concentraciones estima la IC50 por interpolación solo si los datos "
    "cruzan el 50% de viabilidad."
)

with st.expander("Formato obligatorio del CSV"):
    st.write(
        "Columnas obligatorias: `mutacion`, `farmaco`, `concentracion_nM`, "
        "`viabilidad_porcentaje`. La columna `replica` es opcional."
    )

archivo = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

if archivo is not None:
    try:
        datos, eliminadas = limpiar_y_validar_datos(archivo)
        st.success("Archivo cargado correctamente.")
        if eliminadas > 0:
            st.warning(f"Se eliminaron {eliminadas} filas inválidas o incompletas.")

        st.subheader("Datos limpios")
        st.dataframe(datos, width="stretch")
        st.write(
            f"Filas válidas: **{len(datos)}** · "
            f"Fármacos: **{datos['farmaco'].nunique()}** · "
            f"Mutaciones/condiciones: **{datos['mutacion'].nunique()}**"
        )

        farmacos = sorted(datos["farmaco"].unique())
        seleccion = st.multiselect(
            "Selecciona los fármacos que quieres analizar",
            farmacos,
            default=farmacos,
        )

        if st.button("Analizar datos", type="primary"):
            if not seleccion:
                st.warning("Selecciona al menos un fármaco.")
                st.stop()

            datos_filtrados = datos[datos["farmaco"].isin(seleccion)]
            with st.spinner("Analizando datos..."):
                resumen = resumir_replicas(datos_filtrados)
                resultados, parametros = calcular_resultados(resumen)

            st.success("Análisis terminado.")
            st.subheader("Tabla de resultados")

            tabla_mostrada = resultados.copy()
            numericas = [
                "ic50_nM", "respuesta_minima", "respuesta_maxima", "pendiente_hill", "r2"
            ]
            tabla_mostrada[numericas] = tabla_mostrada[numericas].round(3)
            st.dataframe(tabla_mostrada, width="stretch")

            st.download_button(
                "Descargar tabla de resultados",
                resultados.to_csv(index=False).encode("utf-8"),
                file_name="resultados_IC50_adaptativos.csv",
                mime="text/csv",
            )
            st.download_button(
                "Descargar datos resumidos",
                resumen.to_csv(index=False).encode("utf-8"),
                file_name="datos_resumidos.csv",
                mime="text/csv",
            )

            st.header("Curvas por fármaco")
            for farmaco in seleccion:
                figura = crear_figura_farmaco(farmaco, resumen, resultados, parametros)
                st.subheader(farmaco)
                st.pyplot(figura)
                nombre = str(farmaco).replace("/", "_").replace("\\", "_").replace(" ", "_")
                st.download_button(
                    f"Descargar gráfica de {farmaco}",
                    data=figura_a_png(figura),
                    file_name=f"curvas_{nombre}.png",
                    mime="image/png",
                    key=f"descarga_{nombre}",
                )
                plt.close(figura)

            st.header("Figura conjunta")
            figura_conjunta = crear_figura_conjunta(resumen, resultados, parametros)
            st.pyplot(figura_conjunta)
            st.download_button(
                "Descargar figura conjunta",
                data=figura_a_png(figura_conjunta),
                file_name="todas_las_curvas.png",
                mime="image/png",
            )
            plt.close(figura_conjunta)

            figura_heatmap = crear_heatmap(resultados)
            if figura_heatmap is not None:
                st.header("Mapa de calor de IC50")
                st.pyplot(figura_heatmap)
                st.download_button(
                    "Descargar mapa de calor",
                    data=figura_a_png(figura_heatmap),
                    file_name="mapa_calor_IC50.png",
                    mime="image/png",
                )
                plt.close(figura_heatmap)

            st.header("Interpretación automática")
            for _, fila in resultados.iterrows():
                ic50_texto = (
                    f"{fila['ic50_nM']:.2f} nM"
                    if np.isfinite(fila["ic50_nM"])
                    else "no estimable"
                )
                st.write(
                    f"**{fila['farmaco']} – {fila['mutacion']}**: "
                    f"IC50 {ic50_texto}; método: **{fila['metodo']}**; "
                    f"fiabilidad: **{fila['fiabilidad']}**. {fila['estado_ajuste']}"
                )

    except Exception as error:
        st.error(f"No se pudo realizar el análisis: {error}")

warnings.filterwarnings("ignore", category=RuntimeWarning)
