"""
INTERFAZ STREAMLIT PARA EL ANOTADOR UNIVERSAL DE ALPHAMISSENSE

Coloca este archivo en la misma carpeta que:
- anotar_alphamissense_universal.py
- requirements.txt

Ejecuta localmente con:
python -m streamlit run ".\Alphamissense\app_alphamissense.py"
"""

from io import BytesIO
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from anotar_alphamissense_universal import (
    anotar_dataframe,
    crear_graficas,
)


st.set_page_config(
    page_title="Anotador universal AlphaMissense",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Anotador universal AlphaMissense")

st.write(
    "Sube un archivo CSV con variantes de cualquier gen humano. "
    "La aplicación consultará Ensembl VEP y añadirá la predicción "
    "de AlphaMissense cuando esté disponible."
)

st.warning(
    "AlphaMissense es una predicción computacional. "
    "No sustituye la interpretación clínica, ClinVar, la evidencia "
    "funcional ni la valoración de un especialista."
)


with st.expander("Formato recomendado del archivo CSV"):
    st.markdown(
        """
        El formato recomendado incluye estas columnas:

        - `gen`
        - `transcrito`
        - `hgvs_c`
        - `hgvs_p` *(opcional)*
        - `mutacion` *(opcional)*

        Ejemplo:

        ```csv
        gen,transcrito,hgvs_c,hgvs_p,mutacion
        JAK2,ENST00000381652,c.1849G>T,p.Val617Phe,V617F
        NBN,ENST00000265433,c.505C>T,p.Arg169Cys,R169C
        ```

        También puedes usar directamente una columna llamada:

        - `variante_hgvs_consulta`

        Ejemplo:

        ```csv
        variante_hgvs_consulta
        ENST00000381652:c.1849G>T
        ENST00000265433:c.505C>T
        ```
        """
    )


archivo = st.file_uploader(
    "Selecciona el archivo CSV",
    type=["csv"],
)


def convertir_csv_descargable(df: pd.DataFrame) -> bytes:
    return df.to_csv(
        index=False,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")


if archivo is not None:
    try:
        datos = pd.read_csv(archivo)
        datos.columns = datos.columns.astype(str).str.strip()

        st.success("Archivo cargado correctamente.")

        st.subheader("Vista previa de las variantes")
        st.dataframe(datos, use_container_width=True)

        st.write(
            f"El archivo contiene **{len(datos)} variantes** "
            f"y **{len(datos.columns)} columnas**."
        )

        columnas_utiles = {
            "variante_hgvs_consulta",
            "transcrito",
            "hgvs_c",
            "hgvs_p",
            "gen",
        }

        if not columnas_utiles.intersection(datos.columns):
            st.error(
                "El archivo no contiene columnas suficientes para construir "
                "las consultas HGVS. Usa `variante_hgvs_consulta` o, "
                "preferiblemente, `transcrito` y `hgvs_c`."
            )
            st.stop()

        if st.button("Anotar variantes", type="primary"):
            with st.spinner(
                "Consultando Ensembl VEP y recuperando AlphaMissense..."
            ):
                resultado = anotar_dataframe(datos)

            st.success("Anotación terminada.")

            st.subheader("Resultados")
            st.dataframe(
                resultado,
                use_container_width=True,
            )

            nombre_original = Path(archivo.name).stem
            nombre_salida = f"{nombre_original}_AlphaMissense.csv"

            st.download_button(
                label="Descargar CSV anotado",
                data=convertir_csv_descargable(resultado),
                file_name=nombre_salida,
                mime="text/csv",
            )

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
                st.subheader("Resumen")
                st.dataframe(
                    resultado[columnas_resumen],
                    use_container_width=True,
                )

            st.subheader("Gráficas")

            with tempfile.TemporaryDirectory() as carpeta_temporal:
                carpeta = Path(carpeta_temporal)
                crear_graficas(resultado, carpeta)

                ruta_scores = carpeta / "alphamissense_scores.png"
                ruta_clases = carpeta / "alphamissense_clasificaciones.png"

                if ruta_scores.exists():
                    st.image(
                        str(ruta_scores),
                        caption="Puntuaciones AlphaMissense",
                        use_container_width=True,
                    )

                    st.download_button(
                        "Descargar gráfica de puntuaciones",
                        data=ruta_scores.read_bytes(),
                        file_name="alphamissense_scores.png",
                        mime="image/png",
                    )

                if ruta_clases.exists():
                    st.image(
                        str(ruta_clases),
                        caption="Clasificaciones AlphaMissense",
                        use_container_width=True,
                    )

                    st.download_button(
                        "Descargar gráfica de clasificaciones",
                        data=ruta_clases.read_bytes(),
                        file_name="alphamissense_clasificaciones.png",
                        mime="image/png",
                    )

                if not ruta_scores.exists() and not ruta_clases.exists():
                    st.info(
                        "No se generaron gráficas porque no se obtuvieron "
                        "predicciones suficientes."
                    )

            st.subheader("Control de calidad")

            if "estado_anotacion" in resultado.columns:
                conteo_estados = (
                    resultado["estado_anotacion"]
                    .fillna("Sin estado")
                    .value_counts()
                    .rename_axis("estado")
                    .reset_index(name="numero_variantes")
                )

                st.dataframe(
                    conteo_estados,
                    use_container_width=True,
                )

    except Exception as error:
        st.error(f"No se pudo completar la anotación: {error}")
