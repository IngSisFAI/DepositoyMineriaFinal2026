import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import clickhouse_connect
    import os

    _password = os.environ.get("CLICKHOUSE_PASSWORD", "admin")
    engine = clickhouse_connect.get_client(
      host="host.docker.internal",
      user="defaul",
      secure=False,
      port=8123,
      password=_password,
    )
    return (engine,)


@app.cell
def _(engine, mo):
    modelo_df = mo.sql(
        f"""
        SELECT * FROM vista_fraude_unificada
        """,
        engine=engine
    )
    return (modelo_df,)


@app.cell
def _(modelo_df):
    modelo_df.info()
    return


@app.cell
def _(mo, modelo_df):
    #Crearemos una columna adicional llamada id-flag
    #esto se debe a que habia una aclaracion en los logs de que las filas empezadas con ese ID deberian considerarse fraude. 
    import numpy as np

    df3 = modelo_df.copy()

    df3["id_flag"] = np.where(
        df3["transaction_id"].str.startswith("TX-2024-4"),
        1,
        0
    )

    print(df3["transaction_id"].str.startswith("TX-2024-4").sum())
    print((df3["id_flag"]== 1).sum())

    mo.vstack([
        mo.md("## Dataset Preparado para el Árbol"),
        mo.md(f"Columnas finales: `{', '.join(df3.columns.tolist())}`"),
        mo.ui.table(df3.head())
    ])
    return df3, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Antes de empezar con los graficos debemos hacer unas mejoras antes:

        1.  **Limpieza de Ruido**:
            * Eliminamos transaction_id. Al ser un identificador único, no posee valor predictivo y podría causar *overfitting* (que el modelo memorice IDs en lugar de aprender patrones).

        2. **Codificación de Categorías**:
            * Los modelos matemáticos no entienden texto. Transformamos la columna merchant_category en múltiples columnas binarias (0 o 1).
            * *Ejemplo:* Si una transacción es de 'Food', la columna `merchant_category_food` será **1** y las demás **0**.

        3.  **Casteo Final**:
            * Convertimos todas las variables a formatos numéricos compatibles para evitar errores durante el entrenamiento del modelo.

        ---
        > **Nota:** Este proceso garantiza que el dataset esté "limpio" y sea 100% numérico, cumpliendo con los requisitos de la librería `scikit-learn`.
    """)
    return


@app.cell
def _(df3, mo):
    import pandas as pd

    # Copia del dataframe original
    df_proc = df3.copy()

    # Eliminamos el ID de transacción (no aporta al modelo)
    df_proc = df_proc.drop(columns=['transaction_id'], errors='ignore')

    # One-Hot Encoding de merchant_category
    df_proc = pd.get_dummies(df_proc, columns=['merchant_category'], dtype=int)

    # Convertimos todo a numérico
    df_proc = df_proc.apply(pd.to_numeric, errors='coerce')

    # Eliminamos posibles NaN generados en la conversión
    df_proc = df_proc.dropna()

    print("Shape final:", df_proc.shape)

    #RESULTADO EN MARIMO
    mo.vstack([
        mo.md("## Dataset Preparado para el Árbol"),
        mo.md(f"Columnas finales: `{', '.join(df_proc.columns.tolist())}`"),
        mo.ui.table(df_proc.head(16053))
    ]) 

    return df_proc, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diagrama de Cajas
    """)
    return


@app.cell
def _(df_proc, np):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. Seleccionamos las columnas numéricas originales (sin las categorías dummy para no saturar)
    # Usamos df_proc que ya tiene los tipos corregidos
    cols_para_boxplot = ['amount', 'transaction_hour', 'device_trust_score', 'velocity_last_24h', 'cardholder_age']

    # 2. Configuración de la figura
    n_cols = 3
    n_rows = int(np.ceil(len(cols_para_boxplot) / n_cols))

    # Creamos el objeto figure de matplotlib
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten() # Aplanamos para iterar fácil

    # 3. Iteración para crear los gráficos
    for i, col in enumerate(cols_para_boxplot):
        sns.boxplot(data=df_proc, y=col, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribución de {col}', fontsize=12, fontweight='bold')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Limpieza de subplots vacíos (si sobran espacios en la cuadrícula)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0)

    # 5. En marimo, simplemente devolvemos la figura al final de la celda
    fig
    return plt, sns


@app.cell
def _(df_proc, plt, sns):
    # 1. Usamos el dataframe que ya tiene las categorías convertidas (df_proc)
    corr_matrix = df_proc.corr()

    # Cerramos figuras anteriores
    plt.close('all')

    # 2. Creamos el mapa de calor
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": .8}
    )

    plt.title("Mapa de Calor: Correlación de Variables (Incluyendo Categorías)", fontsize=15)

    # 3. Mostrar en Marimo
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fig_corr = plt.gcf()
    fig_corr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Al observar el Mapa de Calor, hemos identificado las variables que tienen mayor influencia en la probabilidad de que una transacción sea catalogada como fraude (is_fraud):

    - transaction_hour: tiene una correlación negativa de -0.16.
        - Las transacciones realizadas en horas de la madrugada (valores bajos de hora) tienen una mayor probabilidad de ser fraudulentas.

    - is_foreign_transaction y location_mismatch muestran una correlación positiva de 0.16 y 0.15.

        - Las transacciones internacionales o aquellas donde la ubicación del comercio no coincide con la del cliente son señales de alerta importantes.

    - merchant_category_groceries: Con un 0.12, es la categoría con mayor peso positivo en este dataset
        - Las compras en supermercados están más asociadas al fraude que otras categorías como farmacias o gasolineras.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Arboles de decision
    """)
    return


@app.cell
def _(df_proc):
    # 1. Reseteamos df_Arbol1 usando el DataFrame original que tiene las transacciones
    # Asegurate de usar el nombre del dataframe que viene de ClickHouse (df o df_proc)
    df_Arbol1 = df_proc.copy()

    # 2. Volvemos a borrar el ID para que no moleste al modelo
    df_Arbol1 = df_Arbol1.drop(columns=['transaction_id'], errors='ignore')

    # 3. Verificamos la proporción (ahora sí debería darte 0 y 1)
    proporcion_real = df_Arbol1['is_fraud'].value_counts(normalize=True)

    print(proporcion_real)
    return (df_Arbol1,)


@app.cell
def _(df_Arbol1):
    # 1. Definimos las variables independientes (X) y la dependiente (y)
    # Quitamos 'is_fraud' de X porque es lo que el modelo tiene que aprender a predecir
    x_df = df_Arbol1.drop('is_fraud', axis=1)

    # Nuestra variable objetivo es is_fraud (0 = Legal, 1 = Fraude)
    y_df = df_Arbol1.is_fraud
    return x_df, y_df


@app.cell
def _(x_df, y_df):
    print(x_df)
    print(y_df)
    return


@app.cell
def _(mo, x_df, y_df):
    from sklearn.model_selection import train_test_split

    # 1. Dividimos el dataset (80% train, 20% test)
    # 'stratify' es vital aquí para mantener ese 6.9% de fraude en ambos lados
    x_train, x_test, y_train, y_test = train_test_split(
        x_df, 
        y_df, 
        test_size=0.20, 
        random_state=42,
        stratify=y_df
    )

    # 2. Creamos el resumen con los valores reales que me pasaste
    resumen_split = {
        "Conjunto": ["Entrenamiento (Train)", "Prueba (Test)"],
        "Registros": [len(x_train), len(x_test)],
        "Proporción de Fraude (%)": [
            f"{y_train.mean() * 100:.2f}%", 
            f"{y_test.mean() * 100:.2f}%"
        ]
    }

    # 3. Visualización en Marimo
    mo.vstack([
        mo.md("### Dataset Dividido"),
        mo.md(f"El modelo tiene **{len(x_train)}** ejemplos para aprender."),
        mo.md(f"Confirmamos que el fraude se mantiene en un **{y_train.mean()*100:.2f}%** en ambos conjuntos."),
        mo.ui.table(resumen_split)
    ])
    return x_test, x_train, y_test, y_train


@app.cell
def _(mo, x_train, y_train):
    from sklearn.tree import DecisionTreeClassifier 
    from sklearn.model_selection import GridSearchCV 
    # 1. Definimos el clasificador y la rejilla de parámetros 
    clf = DecisionTreeClassifier(random_state=42) 
    param_grid = {'criterion': ['gini', 'entropy'], 
    'max_depth': [1, 2, 3, 4, 5, 6]} 

    # 2. Configuramos la búsqueda (Cross-Validation de 5 pliegues) 
    grid_search = GridSearchCV(clf,
    param_grid=param_grid, cv=5, 
    return_train_score=True) 

    # 3. Entrenamos para encontrar la combinación ganadora 
    grid_search.fit(x_train, y_train)

    # 4. Resultados principales
    best_params = grid_search.best_params_ 
    best_score = grid_search.best_score_ 

    # 5. Formato visual para Marimo 
    mo.vstack([ mo.md(f"## Optimización del Árbol Terminada"), mo.md(f"El mejor modelo encontrado utiliza:"), mo.stat(label="Criterio", value=best_params['criterion']), mo.stat(label="Profundidad Máxima", value=str(best_params['max_depth'])), mo.stat(label="Puntuación (CV Score)", value=f"{best_score:.4f}"), mo.md("---"), mo.md("💡 *Este modelo es el que usaremos para predecir si una transacción es fraude o no.*") ])
    return (grid_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ejecutando el clasificador de Arboles de Decision

    ### Vamos a realizar el fit() y predecit() con los parámetros encontrados
    """)
    return


@app.cell
def _(grid_search, mo, x_train, y_train):
    # 1. Extraemos el mejor modelo encontrado por GridSearchCV
    best_clf = grid_search.best_estimator_

    # 2. Lo entrenamos con el set de entrenamiento
    best_clf.fit(x_train, y_train)

    # 3. Mostramos las características del modelo ganador usando Markdown estándar
    mo.vstack([
        mo.md("##Modelo Final Configurado"),
        mo.md(f"El árbol de decisión ha sido entrenado con los siguientes parámetros óptimos:"),
        mo.md(f"```python\n{best_clf}\n```"), 
        mo.md("---"),
        mo.md("**¿Qué significa esto?** \n"
              "Ya tenemos el 'cerebro' del detector de fraudes listo para ser puesto a prueba con los datos que nunca ha visto.")
    ])
    return (best_clf,)


@app.cell
def _(best_clf, mo, pd, x_test, x_train, y_test):
    # 1. Generamos las predicciones
    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)

    # 2. Creamos un pequeño resumen visual para Marimo
    # Comparamos cuántos fraudes predijo vs cuántos hubo realmente en el test
    comparativa = pd.DataFrame({
        "Realidad (y_test)": y_test.value_counts(),
        "Predicción (y_test_pred)": pd.Series(y_test_pred).value_counts().values
    })

    mo.vstack([
        mo.md("###Predicciones Generadas"),
        mo.md("El modelo ya ha etiquetado todas las transacciones del set de prueba."),
        mo.ui.table(comparativa),
        mo.md(f"💡 **Dato clave:** Si los números de 'Realidad' y 'Predicción' son parecidos, el modelo va por muy buen camino.")
    ])
    return (y_test_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Aplicando Metricas

    ### Analizamos métricas como la matriz de confusión, exactitud, precision promedio, etc.
    """)
    return


@app.cell
def _(best_clf, mo, pd, plt, y_test, y_test_pred):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

    # 1. Calculamos la matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred, labels=best_clf.classes_)

    # 2. Creamos el gráfico
    fig_cm, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legal', 'Fraude'])
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Matriz de Confusión: Detección de Fraude')

    # 3. Generamos el reporte de métricas como un DataFrame para que se vea lindo
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # 4. Mostramos todo en Marimo
    mo.vstack([
        mo.md("## Evaluación Final del Modelo"),
        mo.as_html(fig_cm),
        mo.md("### Reporte de Clasificación"),
        mo.ui.table(report_df.round(2)),
        mo.md("💡 **Guía rápida:**\n"
              "* **Precision (Fraude):** Si digo que es fraude, ¿qué tan probable es que lo sea?\n"
              "* **Recall (Fraude):** De todos los fraudes reales, ¿qué porcentaje logré atrapar?")
    ])
    return


@app.cell
def _(best_clf, pd, plt, sns, x_train):
    # 1. Obtener la importancia de las variables del mejor modelo
    feature_scores = pd.DataFrame(
        pd.Series(best_clf.feature_importances_, index=x_train.columns)
        .sort_values(ascending=False), columns=["Importancia"]
    )

    # 2. Seleccionar las top 10
    top_10_features = feature_scores.head(10).reset_index()
    top_10_features.columns = ["Variable", "Importancia"]

    # 3. Crear el gráfico
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=top_10_features,
        x="Variable", y="Importancia", hue="Variable", palette="viridis", legend=False
    )

    # 4. Anotaciones y estilo
    for index, value in enumerate(top_10_features["Importancia"]):
        plt.annotate(f'{value:.2f}', xy=(index, value), ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.title("Variables clave en la detección de Fraude", fontsize=14)
    plt.ylabel("Nivel de Importancia", fontsize=12)
    plt.tight_layout()

    # 5. Capturamos la figura para Marimo
    fig_imp = plt.gcf()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dibujando el Árbol

    ### Ahora finalmente vamos a graficar el arbol obtenido
    """)
    return


@app.cell
def _(best_clf, mo, plt, x_train):
    from sklearn import tree

    # 1. Definimos los nombres de clase correctos para Fraude
    # 0 suele ser Legal y 1 suele ser Fraude
    class_names_fraude = ['Legal', 'Fraude']

    # 2. Configuramos el gráfico
    plt.figure(figsize=(30, 10))

    # Dibujamos el árbol (usando el modelo que ya entrenamos)
    tree.plot_tree(
        best_clf, 
        feature_names=x_train.columns, 
        filled=True, 
        class_names=class_names_fraude,
        rounded=True,
        fontsize=12
    )

    # 3. Guardamos la imagen
    plt.savefig('arbol_decision_fraude.png', dpi=300)

    # 4. Lo mostramos en marimo
    mo.as_html(plt.gcf())
    return


if __name__ == "__main__":
    app.run()
