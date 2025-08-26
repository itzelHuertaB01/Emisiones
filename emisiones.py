import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# ------------- Ruta base (tu carpeta) -------------
BASE = r'C:\Users\mikel\OneDrive\Documentos\Emisiones'

# ============================================================
# 1. Generar un data frame con los datos de los 4 archivos
# ============================================================
emisiones_2016 = pd.read_csv(fr'{BASE}\emisiones-2016.csv', sep=';')
emisiones_2017 = pd.read_csv(fr'{BASE}\emisiones-2017.csv', sep=';')
emisiones_2018 = pd.read_csv(fr'{BASE}\emisiones-2018.csv', sep=';')
emisiones_2019 = pd.read_csv(fr'{BASE}\emisiones-2019.csv', sep=';')

emisiones = pd.concat([emisiones_2016, emisiones_2017, emisiones_2018, emisiones_2019])

print('\n' + '=' * 80)
print('EJERCICIO 1: DataFrame combinado (4 archivos)')
print('=' * 80)
print(emisiones)

# =======================================================================================
# 2. Filtrar las columnas del data frame y que solo muestre las columnas "Estación, Magnitud, Año y Mes" y las correspondientes a los días de 1,2,3,4 etc.
# =======================================================================================
columnas = ['ESTACION', 'MAGNITUD', 'ANO', 'MES']


def _es_dia(col: str) -> bool:
    c = col.strip().upper()
    if c.startswith('D'):
        c = c[1:]
    return c.isdigit() and 1 <= int(c) <= 31


cols_dias = [c for c in emisiones.columns if _es_dia(c)]
columnas.extend(cols_dias)
emisiones = emisiones[columnas]

print('\n' + '=' * 80)
print('EJERCICIO 2: DataFrame filtrado (solo ESTACION, MAGNITUD, ANO, MES y D01..D31)')
print('=' * 80)
print(emisiones)

# ==========================================================================================
# 3. Reestructurar el data frame para que los valores contaminantes de las columnas "Dias" se muestres en una sola columna
# ==========================================================================================
emisiones = emisiones.melt(
    id_vars=['ESTACION', 'MAGNITUD', 'ANO', 'MES'],
    var_name='DIA',
    value_name='VALOR'
)

print('\n' + '=' * 80)
print('EJERCICIO 3: DataFrame reestructurado (melt)')
print('=' * 80)
print(emisiones)

# ==========================================================================
# 4. Agregar una columna, con la fecha a partir del "año el mes y el dia" utilizar datatime
# ==========================================================================
emisiones['DIA'] = emisiones['DIA'].str.strip('D')

# Formar cadena YYYY/MM/DD y convertir
emisiones['FECHA'] = (
        emisiones['ANO'].astype(str) + '/' +
        emisiones['MES'].astype(int).map(lambda m: f'{m:02d}') + '/' +
        emisiones['DIA'].astype(str)
)

emisiones['FECHA'] = pd.to_datetime(
    emisiones['FECHA'],
    format='%Y/%m/%d',
    errors='coerce'
)

print('\n' + '=' * 80)
print('EJERCICIO 4: DataFrame con columna FECHA')
print('=' * 80)
print(emisiones)

# ============================================================================================
# 5. Eliminar las filas con fechas no validas y una vez eliminadas mostrar en el data frame por estaciones contaminantes y fechas
# ============================================================================================
emisiones = emisiones.drop(emisiones[np.isnat(emisiones['FECHA'])].index)
emisiones = emisiones.sort_values(['ESTACION', 'MAGNITUD', 'FECHA'])

print('\n' + '=' * 80)
print('EJERCICIO 5: DataFrame limpio (sin fechas inválidas) y ordenado')
print('=' * 80)
print(emisiones)

# ==================================================================================
# 6. Mostrar en pantalla todas las estaciones y los contaminates disponibles en el data frame
# ==================================================================================
estaciones = sorted(int(x) for x in emisiones['ESTACION'].unique())
contaminantes = sorted(int(x) for x in emisiones['MAGNITUD'].unique())

print('\n' + '=' * 80)
print('EJERCICIO 6: Estaciones disponibles')
print('=' * 80)
print(estaciones)

print('\n' + '=' * 80)
print('EJERCICIO 6: Contaminantes disponibles (MAGNITUD)')
print('=' * 80)
print(contaminantes)

# =======================================================================
# 7. Muestra un resumen descriptivo por cada contaminante utiliza describe y otro por cada estado
# =======================================================================
print('\n' + '=' * 80)
print('EJERCICIO 7: Resumen descriptivo por contaminante (MAGNITUD)')
print('=' * 80)
print(emisiones.groupby('MAGNITUD')['VALOR'].describe())

# ============================================================================================
# 7. Muestra un resumen descriptivo por cada contaminante utiliza describe y otro por cada estado
# ============================================================================================
print('\n' + '=' * 80)
print('EJERCICIO 9: Resumen descriptivo por contaminante y por estación')
print('=' * 80)
print(emisiones.groupby(['ESTACION', 'MAGNITUD'])['VALOR'].describe())


# ======================================================================================================
# EJERCICIO 10: Función resumen(estacion, contaminante) -> describe() de la serie de esa estación/contaminante
# ======================================================================================================
def resumen(estacion: int, contaminante: int) -> pd.Series:
    return emisiones[
        (emisiones['ESTACION'] == estacion) &
        (emisiones['MAGNITUD'] == contaminante)
        ]['VALOR'].describe()


print('\n' + '=' * 80)
print('EJERCICIO 10: Resúmenes por estación/contaminante (ejemplos)')
print('=' * 80)
print('Resumen Dióxido de Nitrógeno en Plaza Elíptica:\n', resumen(56, 8), '\n', sep='')
print('Resumen Dióxido de Nitrógeno en Plaza del Carmen:\n', resumen(35, 8), sep='')


# ===========================================================================================================
# 8. Desarolla una función que devuelva las emiciones medias mensuales y de un contaminate y un año dado para todas las estaciones
# ===========================================================================================================
def evolucion_mensual(contaminante: int, anio: int) -> pd.DataFrame:
    return (emisiones[
                (emisiones['MAGNITUD'] == contaminante) &
                (emisiones['ANO'] == anio)
                ].groupby(['ESTACION', 'MES'])['VALOR']
            .mean()
            .unstack('MES'))


print('\n' + '=' * 80)
print('EJERCICIO 8: Medias mensuales por estación (ejemplo: contaminante=8, año=2019)')
print('=' * 80)
print(evolucion_mensual(8, 2019))


# ===========================================================================================================
# 8. Desarolla una función que devuelva las emiciones medias mensuales y de un contaminate y un año dado para todas las estaciones
# ===========================================================================================================
def medias_mensuales_por_estacion(estacion: int) -> pd.DataFrame:
    df = emisiones[emisiones['ESTACION'] == estacion]
    return df.pivot_table(
        index='MES', columns='MAGNITUD', values='VALOR', aggfunc='mean'
    ).sort_index()


# ===========================================================================================================
# 9. Desarolla una función que reciba una estación de medición y muestre un data frame con las medidas mensuales de todos los tipos de contaminantes
# ===============================================================================
def medias_mensuales_por_estacion(estacion: int) -> pd.DataFrame:
    df = emisiones[emisiones['ESTACION'] == estacion]
    # Tabla pivote: medias por mes y contaminante
    tabla = df.pivot_table(
        index='MES',  # filas = mes
        columns='MAGNITUD',  # columnas = contaminante
        values='VALOR',  # valores = mediciones
        aggfunc='mean'  # media mensual
    )
    # Ordenar meses y columnas
    tabla = tabla.sort_index()
    tabla = tabla.reindex(sorted(tabla.columns), axis=1)
    return tabla


# (Opcional) Ejemplo de impresión en consola con encabezado:
print('\n' + '=' * 80)
print('EJERCICIO 9: Medias mensuales por contaminante en una estación (ejemplo: estación 56)')
print('=' * 80)
print(medias_mensuales_por_estacion(56))

# ============================================================
# SECCIÓN DE GRÁFICAS (matplotlib)
# ============================================================


# Paleta sencilla para rotar colores "bonitos"
_PALETA = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"
]


def _apply_style():
    """Aplica un estilo amigable. Si no está disponible, sigue sin fallar."""
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        pass


# -------------------------------------------------------------
# EJERCICIO 8 (a): Medias por contaminante (barras)
# -------------------------------------------------------------
def plot_medias_por_contaminante():
    """
    Dibuja barras con la media de VALOR por contaminante (MAGNITUD).
    Requiere el DataFrame global `emisiones`.
    """
    _apply_style()
    medias = emisiones.groupby('MAGNITUD')['VALOR'].mean().sort_index()

    plt.figure(figsize=(10, 4.5))
    colors = (_PALETA * ((len(medias) // len(_PALETA)) + 1))[:len(medias)]
    plt.bar(
        medias.index.astype(str),
        medias.values,
        color=colors,
        edgecolor="black",
        linewidth=0.6
    )
    plt.title("Media de emisiones por contaminante (MAGNITUD) - EJ8")
    plt.xlabel("Contaminante (MAGNITUD)")
    plt.ylabel("Media")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------
# EJERCICIO 8 (b): Distribución por contaminante (boxplots)
# ----------------------------------------------------------------
def plot_boxplot_por_contaminante():
    """
    Dibuja boxplots de VALOR agrupado por MAGNITUD.
    Compatibilidad con Matplotlib 3.9+: usa `tick_labels` (antes `labels`).
    """
    _apply_style()
    # Orden de categorías por MAGNITUD
    orden = sorted(emisiones['MAGNITUD'].unique())
    datos = [emisiones.loc[emisiones['MAGNITUD'] == m, 'VALOR'].dropna().values for m in orden]

    plt.figure(figsize=(10, 5))
    bp = plt.boxplot(
        datos,
        patch_artist=True,
        tick_labels=[str(m) for m in orden],  # <-- fix para Matplotlib >= 3.9
        showfliers=False
    )

    # Colorear cajas
    colors = (_PALETA * ((len(orden) // len(_PALETA)) + 1))[:len(orden)]
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)

    plt.title("Distribución de emisiones por contaminante (boxplot) - EJ8")
    plt.xlabel("Contaminante (MAGNITUD)")
    plt.ylabel("Valor")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------
# EJERCICIO 11: Medias mensuales por estación y año (para un contaminante)
# ---------------------------------------------------------------------------------------
def plot_evolucion_mensual(contaminante: int, anio: int, estaciones: list[int] | None = None):
    """
    Dibuja líneas de medias mensuales (mes 1..12) por estación, para un contaminante y año dados.
    - `contaminante`: MAGNITUD a graficar
    - `anio`: año (int)
    - `estaciones`: lista opcional para filtrar qué estaciones mostrar
    Requiere la función `evolucion_mensual(contaminante, anio)`.
    """
    _apply_style()
    tabla = evolucion_mensual(contaminante, anio)  # index: ESTACION ; cols: MES (1..12)

    # Asegurar orden de meses (1..12) por si llegan desordenados o como str
    tabla.columns = [int(c) for c in tabla.columns]
    tabla = tabla.reindex(sorted(tabla.columns), axis=1)

    # Filtrar estaciones si se especifica
    if estaciones:
        tabla = tabla.loc[tabla.index.isin(estaciones)]

    plt.figure(figsize=(11, 6))
    for i, (est, fila) in enumerate(tabla.iterrows()):
        color = _PALETA[i % len(_PALETA)]
        plt.plot(
            tabla.columns,
            fila.values,
            marker="o",
            linewidth=1.8,
            label=f"Est. {est}",
            color=color
        )

    plt.xticks(tabla.columns)
    plt.title(f"Medias mensuales {anio} - Contaminante {contaminante} (EJ11)")
    plt.xlabel("Mes")
    plt.ylabel("Media")
    plt.legend(ncol=2, fontsize=8, frameon=True)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------
# EJERCICIO 12: Medias mensuales por contaminante en una estación
# --------------------------------------------------------------------------------
def plot_medias_mensuales_por_estacion(estacion: int, max_contaminantes: int = 8):
    """
    Dibuja varias líneas (una por MAGNITUD) con la media mensual (mes 1..12)
    en una estación dada. Para evitar saturación, muestra hasta `max_contaminantes`
    ordenados por mayor media anual.
    Requiere `medias_mensuales_por_estacion(estacion)`.
    """
    _apply_style()
    tabla = medias_mensuales_por_estacion(estacion)  # index: MES (1..12) ; cols: MAGNITUD
    tabla = tabla.sort_index()

    # Elegir los contaminantes con mayor media global
    ranking = tabla.mean().sort_values(ascending=False)
    cols = ranking.index[:max_contaminantes]
    tabla = tabla[cols]

    plt.figure(figsize=(11, 6))
    for i, col in enumerate(tabla.columns):
        color = _PALETA[i % len(_PALETA)]
        plt.plot(
            tabla.index,
            tabla[col].values,
            marker="o",
            linewidth=1.8,
            label=f"Mag. {col}",
            color=color
        )

    plt.xticks(tabla.index)
    plt.title(f"Estación {estacion}: medias mensuales por contaminante (EJ12)")
    plt.xlabel("Mes")
    plt.ylabel("Media")
    plt.legend(ncol=2, fontsize=8, frameon=True)
    plt.tight_layout()
    plt.show()


# ============================================================
# EJEMPLOS DE USO (opcional). Borra o comenta si no los necesitas.
# ============================================================
if __name__ == "__main__":
    print('\n' + '=' * 80)
    print('GRÁFICA EJ. 8: Medias por contaminante (barras)')
    print('=' * 80)
    plot_medias_por_contaminante()

    print('\n' + '=' * 80)
    print('GRÁFICA EJ. 8: Distribución por contaminante (boxplot)')
    print('=' * 80)
    plot_boxplot_por_contaminante()

    print('\n' + '=' * 80)
    print('GRÁFICA EJ. 11: Medias mensuales por estación (Mag 8, Año 2019)')
    print('=' * 80)
    # Puedes limitar estaciones si hay muchas, por ejemplo: estaciones=[4,8,11,16,17]
    plot_evolucion_mensual(8, 2019)

    print('\n' + '=' * 80)
    print('GRÁFICA EJ. 12: Medias mensuales por contaminante en estación 56 (hasta 8 líneas)')
    print('=' * 80)
    plot_medias_mensuales_por_estacion(56, max_contaminantes=8)
