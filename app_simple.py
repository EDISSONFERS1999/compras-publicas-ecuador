import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# =====================================================

st.set_page_config(
    page_title="Análisis de Compras Públicas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CONSTANTES
# =====================================================

PROVINCIAS_ECUADOR = [
    "AZUAY", "BOLÍVAR", "CAÑAR", "CARCHI", "CHIMBORAZO", "COTOPAXI", 
    "EL ORO", "ESMERALDAS", "GALÁPAGOS", "GUAYAS", "IMBABURA", "LOJA", 
    "LOS RÍOS", "MANABÍ", "MORONA-SANTIAGO", "NAPO", "ORELLANA", 
    "PASTAZA", "PICHINCHA", "SANTA ELENA", "SANTO DOMINGO DE LOS TSÁCHILAS", 
    "SUCUMBÍOS", "TUNGURAHUA", "ZAMORA-CHINCHIPE"
]

TIPOS_CONTRATACION = [
    "Licitación",
    "Todas",
    "Subasta Inversa Electrónica",
    "Obra artística, científica o literaria",
    "Menor Cuantía",
    "Cotización",
    "Contratos entre Entidades Públicas o sus subsidiarias",
    "Contratación directa",
    "Catálogo electrónico - Mejor oferta",
    "Catálogo electrónico - Compra directa",
    "Bienes y Servicios únicos",
    "Asesoría y Patrocinio Jurídico",
    "Repuestos o Accesorios",
    "Lista corta",
    "Comunicación Social – Contratación Directa",
    "Transporte de correo interno o internacional",
    "Licitación de Seguros",
    "Comunicación Social – Proceso de Selección",
    "Catálogo electrónico - Gran compra puja",
    "Catálogo electrónico - Gran compra mejor oferta"
]

TIPOS_GRAFICAS = {
    "📊 Gráfica de Barras - Total por Tipo": "bar_chart",
    "📈 Evolución Mensual - Líneas": "line_chart", 
    "📊 Barras Apiladas - Mensual por Tipo": "stacked_bar_chart",
    "🥧 Gráfica de Pastel - Proporción Contratos": "pie_chart",
    "🎯 Dispersión - Total vs Contratos": "scatter_chart",
    "📈 Comparativa Líneas - Tipos por Año": "comparative_line_chart"
}

# =====================================================
# FUNCIONES AUXILIARES - API Y PROCESAMIENTO
# =====================================================

def get_combined_data(year, regions, contract_types):
    """
    Obtiene y combina datos de múltiples provincias y tipos de contratación
    
    Args:
        year (int): Año para la consulta
        regions (list): Lista de provincias
        contract_types (list): Lista de tipos de contratación
        
    Returns:
        pd.DataFrame: DataFrame combinado de todas las consultas
    """
    all_data = []
    successful_queries = []
    failed_queries = []
    
    total_queries = len(regions) * len(contract_types)
    
    # Crear barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    query_count = 0
    
    for region in regions:
        for contract_type in contract_types:
            query_count += 1
            
            # Actualizar progreso
            progress = query_count / total_queries
            progress_bar.progress(progress)
            status_text.text(f"Consultando {region} - {contract_type}... ({query_count}/{total_queries})")
            
            # Construir URL para esta combinación
            url = f"https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/get_analysis?year={year}&region={region}&type={contract_type}"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:  # Verificar que hay datos
                        # Agregar columnas de identificación a los datos
                        df_query = pd.DataFrame(data)
                        df_query['provincia'] = region
                        df_query['tipo_consulta'] = contract_type
                        all_data.append(df_query)
                        successful_queries.append(f"{region} - {contract_type}")
                    else:
                        failed_queries.append(f"{region} - {contract_type} (sin datos)")
                else:
                    failed_queries.append(f"{region} - {contract_type} (error {response.status_code})")
            except Exception as e:
                failed_queries.append(f"{region} - {contract_type} (error conexión)")
    
    # Limpiar barra de progreso
    progress_bar.empty()
    status_text.empty()
    
    # Mostrar resumen de consultas
    if successful_queries:
        st.success(f"✅ Datos obtenidos de {len(successful_queries)} consulta(s) exitosa(s)")
    
    if failed_queries:
        st.warning(f"⚠️ Sin datos o con errores en {len(failed_queries)} consulta(s)")
    
    # Combinar todos los datos
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return None

def get_data(url):
    """
    Realiza la consulta a la API de compras públicas (función mantenida para compatibilidad)
    
    Args:
        url (str): URL de la API a consultar
        
    Returns:
        dict: Datos obtenidos de la API o None si hay error
    """
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        return None
    """
    Obtiene y combina datos de múltiples provincias
    
    Args:
        year (int): Año para la consulta
        provinces (list): Lista de provincias
        contract_type (str): Tipo de contratación
        
    Returns:
        pd.DataFrame: DataFrame combinado de todas las provincias
    """
    all_data = []
    successful_provinces = []
    failed_provinces = []
    
    # Crear barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, province in enumerate(provinces):
        # Actualizar progreso
        progress = (i + 1) / len(provinces)
        progress_bar.progress(progress)
        status_text.text(f"Consultando {province}... ({i+1}/{len(provinces)})")
        
        # Construir URL para esta provincia
        url = f"https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/get_analysis?year={year}&region={province}&type={contract_type}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:  # Verificar que hay datos
                    # Agregar columna de provincia a los datos
                    df_province = pd.DataFrame(data)
                    df_province['provincia'] = province
                    all_data.append(df_province)
                    successful_provinces.append(province)
                else:
                    failed_provinces.append(f"{province} (sin datos)")
            else:
                failed_provinces.append(f"{province} (error {response.status_code})")
        except Exception as e:
            failed_provinces.append(f"{province} (error conexión)")
    
    # Limpiar barra de progreso
    progress_bar.empty()
    status_text.empty()
    
    # Mostrar resumen de consultas
    if successful_provinces:
        st.success(f"✅ Datos obtenidos de {len(successful_provinces)} provincia(s): {', '.join(successful_provinces)}")
    
    if failed_provinces:
        st.warning(f"⚠️ Sin datos o con errores en {len(failed_provinces)} provincia(s): {', '.join(failed_provinces)}")
    
    # Combinar todos los datos
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return None
    """
    Realiza la consulta a la API de compras públicas
    
    Args:
        url (str): URL de la API a consultar
        
    Returns:
        dict: Datos obtenidos de la API o None si hay error
    """
    try:
        response = requests.get(url)
        
        # Mostrar información de depuración
        with st.expander("🔍 Información de Depuración"):
            st.code(f"URL: {url}")
            st.code(f"Status Code: {response.status_code}")
            st.code(f"Respuesta completa: {response.text}")
        
        if response.status_code == 200:
            st.success("✅ Consulta exitosa, procesando los datos...")
            try:
                json_data = response.json()
                st.write("📋 Estructura de datos recibidos:")
                st.json(json_data)
                return json_data
            except ValueError as e:
                st.error(f"❌ Error al convertir respuesta a JSON: {e}")
                return None
        else:
            st.error(f"❌ Error en la consulta, código de estado: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error en la consulta: {e}")
        return None

def process_data(data):
    """
    Procesa y limpia los datos obtenidos de la API
    
    Args:
        data (dict o DataFrame): Datos crudos de la API o DataFrame ya procesado
        
    Returns:
        pd.DataFrame: DataFrame procesado y limpio
    """
    try:
        # Mostrar información sobre los datos recibidos
        st.write("🔍 **Analizando estructura de datos:**")
        st.write(f"Tipo de datos: {type(data)}")
        
        # Si ya es un DataFrame (datos combinados), procesarlo directamente
        if isinstance(data, pd.DataFrame):
            st.success("✅ Datos ya procesados como DataFrame")
            df = data.copy()
            
            st.write(f"📊 DataFrame con {len(df)} filas y {len(df.columns)} columnas")
            st.write("📋 Columnas disponibles:")
            st.write(list(df.columns))
            
            if len(df) > 0:
                st.write("📄 Muestra de datos (primeras 3 filas):")
                st.dataframe(df.head(3))
        else:
            # Procesar datos crudos de API (código original)
            # Verificar si los datos están vacíos
            if isinstance(data, list) and len(data) == 0:
                st.warning("📭 **La API devolvió una lista vacía**")
                st.info("💡 **Posibles causas:**\n"
                       "- No hay registros para esa combinación específica de filtros\n"
                       "- Los datos pueden no estar disponibles para ese año/provincia/tipo\n"
                       "- Intenta con diferentes parámetros (otro año, provincia o tipo de contratación)")
                return None
            
            if isinstance(data, dict):
                st.write(f"Claves disponibles: {list(data.keys())}")
                # Verificar si el diccionario está vacío o tiene datos vacíos
                if not data or all(not v for v in data.values()):
                    st.warning("📭 **La API devolvió datos vacíos**")
                    st.info("💡 **Posibles causas:**\n"
                           "- No hay registros para esa combinación específica de filtros\n"
                           "- Los datos pueden no estar disponibles para ese año/provincia/tipo\n"
                           "- Intenta con diferentes parámetros")
                    return None
                
            # Intentar diferentes formas de convertir a DataFrame
            if isinstance(data, list):
                if len(data) == 0:
                    return None
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Si es un diccionario, puede estar en una clave específica
                if 'data' in data:
                    if not data['data'] or len(data['data']) == 0:
                        st.warning("📭 **No hay datos en 'data'**")
                        return None
                    df = pd.DataFrame(data['data'])
                elif 'results' in data:
                    if not data['results'] or len(data['results']) == 0:
                        st.warning("📭 **No hay datos en 'results'**")
                        return None
                    df = pd.DataFrame(data['results'])
                elif 'records' in data:
                    if not data['records'] or len(data['records']) == 0:
                        st.warning("📭 **No hay datos en 'records'**")
                        return None
                    df = pd.DataFrame(data['records'])
                else:
                    # Intentar usar los datos directamente
                    df = pd.DataFrame([data])
            else:
                st.error("❌ Formato de datos no reconocido")
                return None
            
            st.success(f"✅ DataFrame creado con {len(df)} filas y {len(df.columns)} columnas")
            st.write("📋 Columnas disponibles:")
            st.write(list(df.columns))
            
            if len(df) > 0:
                st.write("📄 Muestra de datos (primeras 3 filas):")
                st.dataframe(df.head(3))
        
        # Verificar columnas necesarias
        required_columns = ['internal_type', 'total', 'month', 'contracts']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"⚠️ Columnas faltantes: {missing_columns}")
            st.write("💡 Intentando mapear columnas similares...")
            
            # Mapeo alternativo de nombres de columnas
            column_mapping = {
                'tipo': 'internal_type',
                'type': 'internal_type',
                'tipo_contratacion': 'internal_type',
                'monto': 'total',
                'valor': 'total',
                'amount': 'total',
                'mes': 'month',
                'contratos': 'contracts',
                'num_contratos': 'contracts',
                'cantidad': 'contracts'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df[new_name] = df[old_name]
                    st.success(f"✅ Mapeado: '{old_name}' → '{new_name}'")
        
        # Verificar nuevamente después del mapeo
        still_missing = [col for col in required_columns if col not in df.columns]
        if still_missing:
            st.error(f"❌ No se pudieron encontrar las columnas: {still_missing}")
            st.info("💡 Datos disponibles podrían estar en formato diferente")
            return None
        
        # Conversión de tipos de datos
        df['total'] = pd.to_numeric(df['total'], errors='coerce')
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
        df['contracts'] = pd.to_numeric(df['contracts'], errors='coerce')
        
        # Manejo de valores nulos
        initial_rows = len(df)
        df = df.dropna(subset=['internal_type', 'total'])
        final_rows = len(df)
        
        if initial_rows != final_rows:
            st.info(f"ℹ️ Se eliminaron {initial_rows - final_rows} filas con valores nulos")
        
        if len(df) == 0:
            st.warning("⚠️ No quedan datos válidos después de la limpieza")
            st.info("💡 **Sugerencias:**\n"
                   "- Prueba con otros filtros (año, provincia, tipo de contratación)\n"
                   "- Algunos años/provincias pueden tener pocos registros\n"
                   "- Verifica que la combinación de filtros sea válida")
            return None
        
        # Mapear números de mes a nombres en español
        month_names = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        
        # Convertir month a numérico y luego mapear a nombres
        df['month_name'] = df['month'].map(month_names)
        df['month_order'] = df['month']  # Para ordenar correctamente
        
        st.success(f"✅ Datos procesados correctamente: {len(df)} registros válidos")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error procesando datos: {str(e)}")
        st.error(f"Tipo de error: {type(e).__name__}")
        return None

# =====================================================
# FUNCIONES DE VISUALIZACIÓN
# =====================================================

def create_bar_chart(df):
    """Crea gráfica de barras: Total por Tipo de Contratación"""
    df_grouped = df.groupby(['month_name', 'month_order', 'internal_type']).agg({'total': 'sum'}).reset_index()
    df_grouped = df_grouped.sort_values('month_order')  # Ordenar por número de mes
    
    fig = px.bar(
        df_grouped, 
        x='month_name', 
        y='total', 
        title='📊 Total por Tipo de Contratación', 
        labels={'month_name': 'Mes', 'total': 'Monto Total', 'internal_type': 'Tipo de Contratación'},
        color='internal_type',
        text='total',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Monto Total (USD)",
        showlegend=True,
        height=600
    )
    
    return fig

def create_line_chart(df):
    """Crea gráfica de líneas: Evolución Mensual de Montos Totales"""
    df_monthly = df.groupby(['month_name', 'month_order']).agg({'total': 'sum'}).reset_index()
    df_monthly = df_monthly.sort_values('month_order')  # Ordenar por número de mes
    
    fig = px.line(
        df_monthly, 
        x='month_name', 
        y='total', 
        title='📈 Evolución Mensual de Montos Totales',
        labels={'month_name': 'Mes', 'total': 'Monto Total'},
        markers=True,
        line_shape='linear',
        range_y=[0, df_monthly['total'].max() + 5000000],
        template='plotly_white'
    )
    
    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Monto Total (USD)",
        height=600
    )
    
    return fig

def create_stacked_bar_chart(df):
    """Crea gráfica de barras apiladas: Total por Tipo de Contratación por Mes"""
    df_grouped = df.groupby(['month_name', 'month_order', 'internal_type']).agg({'total': 'sum'}).reset_index()
    df_grouped = df_grouped.sort_values('month_order')  # Ordenar por número de mes
    
    fig = px.bar(
        df_grouped, 
        x='month_name', 
        y='total', 
        color='internal_type', 
        title='📊 Total por Tipo de Contratación por Mes', 
        labels={'month_name': 'Mes', 'total': 'Monto Total'},
        text='total',
        color_discrete_sequence=px.colors.qualitative.Set1,
        barmode='stack',
        template='plotly_white'
    )
    
    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Monto Total (USD)",
        height=600
    )
    
    return fig

def create_pie_chart(df):
    """Crea gráfica de pastel: Proporción de Contratos por Tipo de Contratación"""
    df_contracts = df.groupby('internal_type').agg({'contracts': 'sum'}).reset_index()
    
    fig = px.pie(
        df_contracts, 
        names='internal_type', 
        values='contracts', 
        title='🥧 Proporción de Contratos por Tipo de Contratación',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(height=600)
    
    return fig

def create_scatter_chart(df):
    """Crea gráfica de dispersión: Total vs. Cantidad de Contratos"""
    fig = px.scatter(
        df, 
        x='contracts', 
        y='total', 
        title='🎯 Total vs. Cantidad de Contratos', 
        labels={'contracts': 'Cantidad de Contratos', 'total': 'Monto Total'},
        color='internal_type',
        trendline='ols',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        xaxis_title="Cantidad de Contratos",
        yaxis_title="Monto Total (USD)",
        height=600
    )
    
    return fig

def create_comparative_line_chart(df):
    """Crea gráfica de líneas comparativa: Tipos de Contratación a lo Largo del Año"""
    df_comparative = df.groupby(['month_name', 'month_order', 'internal_type']).agg({'total': 'sum'}).reset_index()
    df_comparative = df_comparative.sort_values('month_order')  # Ordenar por número de mes
    
    fig = px.line(
        df_comparative, 
        x='month_name', 
        y='total', 
        color='internal_type', 
        title='📈 Comparativa de Tipos de Contratación a lo Largo del Año',
        labels={'month_name': 'Mes', 'total': 'Monto Total'},
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Monto Total (USD)",
        height=600
    )
    
    return fig

def get_chart_description(chart_type):
    """Retorna la descripción de cada tipo de gráfica"""
    descriptions = {
        "bar_chart": "**Descripción:** Muestra el total de montos por cada tipo de contratación agrupados por mes. Permite identificar cuáles son los tipos de contratación más utilizados y con mayores montos.",
        
        "line_chart": "**Descripción:** Representa la evolución del monto total de contratos a lo largo de los meses. Ayuda a visualizar tendencias a lo largo del año, como picos en ciertos meses.",
        
        "stacked_bar_chart": "**Descripción:** Muestra la distribución del total de montos por tipo de contratación para cada mes usando barras apiladas. Permite ver cómo se distribuyen los diferentes tipos de contratación mes a mes.",
        
        "pie_chart": "**Descripción:** Representa la proporción de contratos por cada tipo de contratación en formato circular. Muestra qué tipos de contratación son más frecuentes en términos de cantidad.",
        
        "scatter_chart": "**Descripción:** Muestra la relación entre el total de montos y la cantidad de contratos mediante puntos dispersos. Permite identificar si hay una correlación entre el número de contratos y el monto total.",
        
        "comparative_line_chart": "**Descripción:** Compara diferentes tipos de contratación a lo largo de los meses usando múltiples líneas. Permite observar patrones y comportamientos en diferentes tipos de contratación a lo largo del tiempo."
    }
    return descriptions.get(chart_type, "")

# =====================================================
# FUNCIONES DE MODELOS DE PREDICCIÓN
# =====================================================

def apply_clustering(df):
    """Aplica clustering para agrupar provincias/contratos"""
    try:
        # Preparar datos para clustering
        features = df.groupby('internal_type').agg({
            'total': ['sum', 'mean', 'count'],
            'contracts': ['sum', 'mean']
        }).reset_index()
        
        features.columns = ['internal_type', 'total_sum', 'total_mean', 'total_count', 'contracts_sum', 'contracts_mean']
        
        # Verificar si hay suficientes datos
        if len(features) < 3:
            st.warning(f"⚠️ Se necesitan al menos 3 tipos de contratación para clustering, pero solo hay {len(features)}")
            st.info("💡 **Sugerencias:**\n- Prueba con un filtro más amplio (ej: 'Todas' en tipo de contratación)\n- Selecciona una provincia con más actividad\n- Cambia a un año con más datos")
            return None
        
        # Normalizar datos
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features[['total_sum', 'total_mean', 'contracts_sum', 'contracts_mean']])
        
        # Determinar número óptimo de clusters (máximo 3, pero puede ser menos)
        n_clusters = min(3, len(features))
        
        # Aplicar KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        features['cluster'] = kmeans.fit_predict(features_scaled)
        
        return features
    except Exception as e:
        st.error(f"Error en clustering: {e}")
        return None

def predict_amounts(df):
    """Predice montos usando regresión simple"""
    try:
        # Preparar datos
        df_pred = df.copy()
        df_pred['month_num'] = df_pred['month_order']
        
        # Agrupar por mes
        monthly_data = df_pred.groupby('month_num').agg({
            'total': 'sum',
            'contracts': 'sum'
        }).reset_index()
        
        if len(monthly_data) < 3:
            st.warning(f"⚠️ Se necesitan al menos 3 meses de datos para predicción, pero solo hay {len(monthly_data)}")
            st.info("💡 **Sugerencias:**\n- Selecciona un año completo con más meses\n- Prueba con una provincia más grande\n- Usa 'Todas' en tipo de contratación para obtener más datos")
            return None
            
        # Predicción simple usando tendencia lineal
        X = monthly_data['month_num'].values.reshape(-1, 1)
        y = monthly_data['total'].values
        
        # Verificar que hay variación en los datos
        if len(set(y)) == 1:
            st.warning("⚠️ Todos los valores son iguales, no se puede calcular tendencia")
            return None
        
        # Calcular tendencia
        trend = np.polyfit(monthly_data['month_num'], monthly_data['total'], 1)
        
        # Predecir próximos 3 meses
        next_months = [13, 14, 15]  # Enero, Febrero, Marzo del siguiente año
        predictions = []
        
        for month in next_months:
            pred_value = trend[0] * month + trend[1]
            predictions.append({
                'month': month,
                'predicted_total': max(0, pred_value)  # No valores negativos
            })
        
        return pd.DataFrame(predictions)
    except Exception as e:
        st.error(f"Error en predicción: {e}")
        return None

def classify_contracts(df):
    """Clasifica contratos por valor (Alto, Medio, Bajo)"""
    try:
        # Verificar que hay suficientes datos
        if len(df) < 3:
            st.warning(f"⚠️ Se necesitan al menos 3 registros para clasificación, pero solo hay {len(df)}")
            st.info("💡 **Sugerencias:**\n- Selecciona una provincia con más actividad\n- Usa 'Todas' en tipo de contratación\n- Prueba con un año diferente")
            return None
        
        # Crear categorías basadas en percentiles
        df_class = df.copy()
        df_class['total_per_contract'] = df_class['total'] / df_class['contracts']
        
        # Verificar que hay variación en los valores
        unique_values = df_class['total_per_contract'].nunique()
        if unique_values < 3:
            st.warning(f"⚠️ Hay muy poca variación en los datos ({unique_values} valores únicos)")
            st.info("💡 La clasificación funciona mejor con más variedad de montos")
            # Continuar con clasificación simple
            if unique_values == 1:
                df_class['classification'] = 'Medio'
                return df_class
        
        # Definir umbrales
        high_threshold = df_class['total_per_contract'].quantile(0.75)
        low_threshold = df_class['total_per_contract'].quantile(0.25)
        
        # Clasificar
        def classify_value(value):
            if value >= high_threshold:
                return 'Alto'
            elif value <= low_threshold:
                return 'Bajo'
            else:
                return 'Medio'
        
        df_class['classification'] = df_class['total_per_contract'].apply(classify_value)
        
        return df_class
    except Exception as e:
        st.error(f"Error en clasificación: {e}")
        return None

def apply_pca(df):
    """Aplica PCA para reducción dimensional"""
    try:
        # Preparar datos
        features = df.groupby('internal_type').agg({
            'total': ['sum', 'mean', 'std'],
            'contracts': ['sum', 'mean', 'std'],
            'month_order': ['min', 'max', 'mean']
        }).reset_index()
        
        # Limpiar nombres de columnas
        features.columns = ['internal_type', 'total_sum', 'total_mean', 'total_std', 
                           'contracts_sum', 'contracts_mean', 'contracts_std',
                           'month_min', 'month_max', 'month_mean']
        
        # Rellenar NaN
        features = features.fillna(0)
        
        # Aplicar PCA
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Verificar dimensiones mínimas
        n_samples, n_features = numeric_features.shape
        
        if n_samples < 2:
            st.warning(f"⚠️ PCA necesita al menos 2 muestras, pero solo hay {n_samples}")
            st.info("💡 **Sugerencias:**\n- Selecciona una provincia con más tipos de contratación\n- Usa 'Todas' en tipo de contratación\n- Prueba con un filtro más amplio")
            return None, None
        
        if n_features < 2:
            st.warning(f"⚠️ PCA necesita al menos 2 características, pero solo hay {n_features}")
            st.info("💡 Los datos no tienen suficiente variabilidad para PCA")
            return None, None
        
        # Determinar número de componentes (máximo 2, pero puede ser menos)
        n_components = min(2, n_samples, n_features)
        
        if n_components < 2:
            st.warning(f"⚠️ Solo se puede calcular {n_components} componente principal")
            st.info("💡 Se necesitan más datos para PCA bidimensional")
            return None, None
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_features)
        
        features['pca1'] = pca_result[:, 0]
        if n_components > 1:
            features['pca2'] = pca_result[:, 1]
        else:
            features['pca2'] = 0  # Si solo hay 1 componente
        
        return features, pca.explained_variance_ratio_
        
    except Exception as e:
        st.error(f"Error en PCA: {e}")
        return None, None

def detect_anomalies(df):
    """Detecta anomalías usando Isolation Forest"""
    try:
        # Verificar que hay suficientes datos
        if len(df) < 5:
            st.warning(f"⚠️ Se necesitan al menos 5 registros para detección de anomalías, pero solo hay {len(df)}")
            st.info("💡 **Sugerencias:**\n- Selecciona una provincia con más actividad\n- Usa 'Todas' en tipo de contratación\n- Prueba con un año con más datos")
            return None
        
        # Preparar datos
        features = df[['total', 'contracts', 'month_order']].copy()
        features = features.fillna(0)
        
        # Verificar que hay variación en los datos
        if features.std().sum() == 0:
            st.warning("⚠️ No hay variación en los datos para detectar anomalías")
            st.info("💡 Todos los valores son muy similares")
            return None
        
        # Ajustar contaminación basada en el tamaño de los datos
        contamination = min(0.2, max(0.05, 1.0 / len(df)))  # Entre 5% y 20%
        
        # Aplicar Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(features)
        
        df_anomalies = df.copy()
        df_anomalies['anomaly'] = anomalies
        df_anomalies['is_anomaly'] = df_anomalies['anomaly'] == -1
        
        return df_anomalies
    except Exception as e:
        st.error(f"Error en detección de anomalías: {e}")
        return None

def show_prediction_models(df):
    """Muestra la sección de modelos de predicción"""
    st.markdown("---")
    st.header("🤖 Aplicación de Modelos de Predicción")
    
    # Crear pestañas para diferentes modelos
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Clustering", 
        "📈 Predicción", 
        "📊 Clasificación", 
        "🔍 PCA", 
        "⚠️ Anomalías"
    ])
    
    with tab1:
        st.subheader("🎯 Clustering - Agrupar provincias/contratos")
        
        cluster_result = apply_clustering(df)
        if cluster_result is not None:
            st.success("✅ Clustering aplicado exitosamente")
            
            # Mostrar resultados
            fig_cluster = px.scatter(
                cluster_result, 
                x='total_sum', 
                y='contracts_sum',
                color='cluster',
                hover_data=['internal_type'],
                title='Agrupación de Tipos de Contratación',
                labels={'total_sum': 'Total Montos', 'contracts_sum': 'Total Contratos'}
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Mostrar tabla de clusters
            st.write("📋 **Resultados del Clustering:**")
            st.dataframe(cluster_result)
            
            st.info("**Resultado:** Los tipos de contratación han sido agrupados en clusters basados en montos y cantidad de contratos.")
        else:
            st.error(f"❌ No se pudo aplicar clustering a los datos")
            st.info("💡 **Para que funcione el clustering necesitas:**\n- Datos de al menos 3 tipos de contratación diferentes\n- Selecciona 'Todas' en tipo de contratación\n- Prueba con provincias más grandes como GUAYAS o PICHINCHA")
    
    with tab2:
        st.subheader("📈 Predicción de Montos")
        
        predictions = predict_amounts(df)
        if predictions is not None:
            st.success("✅ Predicciones generadas exitosamente")
            
            # Mostrar predicciones
            month_names = {13: 'Enero (Próximo)', 14: 'Febrero (Próximo)', 15: 'Marzo (Próximo)'}
            predictions['month_name'] = predictions['month'].map(month_names)
            
            fig_pred = px.bar(
                predictions,
                x='month_name',
                y='predicted_total',
                title='Predicción de Montos - Próximos 3 Meses',
                labels={'predicted_total': 'Monto Predicho (USD)', 'month_name': 'Mes'}
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Mostrar tabla de predicciones
            st.write("📋 **Predicciones:**")
            predictions_display = predictions.copy()
            predictions_display['predicted_total'] = predictions_display['predicted_total'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(predictions_display[['month_name', 'predicted_total']])
            
            st.info("**Resultado:** Predicciones basadas en tendencia histórica de los datos.")
        else:
            st.error("❌ No se pudieron generar predicciones")
            st.info("💡 **Para que funcionen las predicciones necesitas:**\n- Datos de al menos 3 meses diferentes\n- Prueba con 'Todas' en tipo de contratación\n- Selecciona un año completo con más actividad")
    
    with tab3:
        st.subheader("📊 Clasificación de Contratos por Valor")
        
        classified = classify_contracts(df)
        if classified is not None:
            st.success("✅ Clasificación completada exitosamente")
            
            # Gráfica de clasificación
            classification_count = classified['classification'].value_counts().reset_index()
            classification_count.columns = ['classification', 'count']
            
            fig_class = px.pie(
                classification_count,
                names='classification',
                values='count',
                title='Clasificación de Contratos por Valor',
                color_discrete_map={'Alto': 'red', 'Medio': 'yellow', 'Bajo': 'green'}
            )
            st.plotly_chart(fig_class, use_container_width=True)
            
            # Estadísticas de clasificación
            st.write("📋 **Estadísticas de Clasificación:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                alto_count = len(classified[classified['classification'] == 'Alto'])
                st.metric("🔴 Contratos Alto Valor", alto_count)
            
            with col2:
                medio_count = len(classified[classified['classification'] == 'Medio'])
                st.metric("🟡 Contratos Valor Medio", medio_count)
            
            with col3:
                bajo_count = len(classified[classified['classification'] == 'Bajo'])
                st.metric("🟢 Contratos Bajo Valor", bajo_count)
            
            st.info("**Resultado:** Los contratos han sido clasificados en Alto, Medio y Bajo valor basado en percentiles.")
        else:
            st.error("❌ No se pudo realizar la clasificación")
            st.info("💡 **Para que funcione la clasificación necesitas:**\n- Al menos 3 registros de contratos\n- Datos con variación en los montos\n- Prueba con filtros más amplios")
    
    with tab4:
        st.subheader("🔍 PCA - Visualizar agrupaciones ocultas")
        
        pca_result, variance_ratio = apply_pca(df)
        if pca_result is not None:
            st.success("✅ PCA aplicado exitosamente")
            
            # Gráfica PCA
            fig_pca = px.scatter(
                pca_result,
                x='pca1',
                y='pca2',
                hover_data=['internal_type'],
                title='Análisis de Componentes Principales (PCA)',
                labels={'pca1': f'PC1 ({variance_ratio[0]:.2%} varianza)', 
                       'pca2': f'PC2 ({variance_ratio[1]:.2%} varianza)'}
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # Información sobre varianza explicada
            st.write("📊 **Varianza Explicada:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("PC1", f"{variance_ratio[0]:.2%}")
            
            with col2:
                st.metric("PC2", f"{variance_ratio[1]:.2%}")
            
            st.info("**Resultado:** PCA reduce la dimensionalidad y muestra patrones ocultos en los datos.")
        else:
            st.error("❌ No se pudo aplicar PCA")
            st.info("💡 **Para que funcione PCA necesitas:**\n- Al menos 2 tipos de contratación diferentes\n- Datos con suficiente variabilidad\n- Selecciona 'Todas' en tipo de contratación")
    
    with tab5:
        st.subheader("⚠️ Detección de Anomalías")
        
        anomalies_result = detect_anomalies(df)
        if anomalies_result is not None:
            anomaly_count = len(anomalies_result[anomalies_result['is_anomaly'] == True])
            st.success(f"✅ Detección completada: {anomaly_count} anomalías encontradas")
            
            # Gráfica de anomalías
            fig_anomaly = px.scatter(
                anomalies_result,
                x='contracts',
                y='total',
                color='is_anomaly',
                title='Detección de Contratos Inusuales',
                labels={'contracts': 'Cantidad de Contratos', 'total': 'Monto Total'},
                color_discrete_map={True: 'red', False: 'blue'}
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Mostrar anomalías detectadas
            if anomaly_count > 0:
                st.write("🚨 **Contratos Anómalos Detectados:**")
                anomalous_contracts = anomalies_result[anomalies_result['is_anomaly'] == True]
                st.dataframe(anomalous_contracts[['internal_type', 'total', 'contracts', 'month_name']])
            
            # Métricas
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🚨 Anomalías", anomaly_count)
            
            with col2:
                normal_count = len(anomalies_result[anomalies_result['is_anomaly'] == False])
                st.metric("✅ Contratos Normales", normal_count)
            
            st.info("**Resultado:** Se identificaron contratos con patrones inusuales que podrían requerir revisión.")
        else:
            st.error("❌ No se pudo realizar la detección de anomalías")
            st.info("💡 **Para que funcione la detección de anomalías necesitas:**\n- Al menos 5 registros de contratos\n- Datos con variación en montos y cantidades\n- Prueba con filtros más amplios")

# =====================================================
# INTERFAZ DE USUARIO
# =====================================================

# Título principal
st.title('🏛️ Análisis de Compras Públicas del Ecuador')
st.markdown("---")

# Sidebar para parámetros
st.sidebar.header("⚙️ Parámetros de Consulta")

with st.sidebar:
    st.markdown("### 📅 Filtros de Datos:")
    
    year = st.selectbox(
        '🗓️ Año', 
        [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        index=8,  # Por defecto 2023
        help="Selecciona el año para el análisis"
    )
    
    region = st.multiselect(
        '🏞️ Provincias', 
        options=PROVINCIAS_ECUADOR,
        default=['GUAYAS'],  # Por defecto GUAYAS
        help="Selecciona una o más provincias para filtrar los datos"
    )
    
    st.markdown("### 📝 Selección de Tipos de Contratación:")
    
    # Opción para seleccionar todos los tipos
    select_all_types = st.checkbox(
        "📋 Seleccionar todos los tipos de contratación", 
        value=False,
        help="Marca esta casilla para incluir todos los tipos de contratación"
    )
    
    if select_all_types:
        selected_contract_types = TIPOS_CONTRATACION.copy()
        st.success(f"✅ Seleccionados todos los tipos ({len(TIPOS_CONTRATACION)})")
        # Mostrar algunos tipos como ejemplo
        st.info(f"🔸 Incluye: {', '.join(TIPOS_CONTRATACION[:3])}... y {len(TIPOS_CONTRATACION)-3} más")
    else:
        # Multiselect para tipos de contratación
        selected_contract_types = st.multiselect(
            '📝 Selecciona los Tipos de Contratación:', 
            options=TIPOS_CONTRATACION,
            default=['Licitación'],  # Por defecto Licitación
            help="Puedes seleccionar uno o más tipos. Usa Ctrl+Click para seleccionar múltiples opciones.",
            key="contract_types_multiselect"
        )
        
        if len(selected_contract_types) == 0:
            st.warning("⚠️ Debes seleccionar al menos un tipo de contratación")
        elif len(selected_contract_types) == 1:
            st.info(f"📋 Seleccionado: {selected_contract_types[0]}")
        else:
            st.success(f"✅ Seleccionados {len(selected_contract_types)} tipos:")
            # Mostrar los tipos seleccionados
            for i, tipo in enumerate(selected_contract_types[:5]):  # Mostrar máximo 5
                st.write(f"   🔸 {tipo}")
            if len(selected_contract_types) > 5:
                st.write(f"   📋 ... y {len(selected_contract_types) - 5} más")
    
    st.markdown("---")
    
    # Botón para ejecutar consulta
    execute_query = st.button('🔍 Ejecutar Consulta', type='primary', use_container_width=True)
    
    # Validar selección antes de permitir consulta
    if not select_all_types and len(selected_contract_types) == 0:
        st.error("❌ Selecciona al menos un tipo de contratación para continuar")

# =====================================================
# ÁREA PRINCIPAL
# =====================================================

if not execute_query:
    # Mensaje inicial
    st.info("👈 **Instrucciones:**\n1. Selecciona los parámetros en la barra lateral\n2. Presiona '🔍 Ejecutar Consulta' para ver todas las visualizaciones")
    
    # Mostrar información adicional
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("ℹ️ Sobre la aplicación"):
            st.markdown("""
            ### 🎯 Objetivo
            Analizar datos de compras públicas del Ecuador mediante visualizaciones interactivas y modelos de predicción.
            
            ### 📊 Características
            - Consulta datos de la API oficial (SERCOP)
            - 6 visualizaciones automáticas
            - 5 modelos de Machine Learning
            - Filtros por año, provincia y tipo de contratación
            - Descarga de datos en formato CSV
            """)
    
    with col2:
        with st.expander("📋 Funcionalidades"):
            st.markdown("""
            ### 📊 Visualizaciones:
            - **Barras**: Total por tipo de contratación
            - **Líneas**: Evolución mensual de montos
            - **Barras Apiladas**: Distribución mensual por tipo
            - **Pastel**: Proporción de contratos
            - **Dispersión**: Relación montos vs contratos
            - **Comparativa**: Tendencias por tipo de contratación
            
            ### 🤖 Modelos de Predicción:
            - **Clustering**: Agrupación automática
            - **Predicción**: Montos futuros
            - **Clasificación**: Contratos por valor
            - **PCA**: Patrones ocultos
            - **Anomalías**: Detección de irregularidades
            """)

else:
    # =====================================================
    # PROCESAMIENTO DE DATOS
    # =====================================================
    
    # Validar que se hayan seleccionado provincias y tipos
    if len(region) == 0:
        st.error("❌ Debes seleccionar al menos una provincia para continuar")
        st.stop()
    
    if not select_all_types and len(selected_contract_types) == 0:
        st.error("❌ Debes seleccionar al menos un tipo de contratación para continuar")
        st.stop()
    
    # Determinar tipos de contratación a consultar
    types_to_query = selected_contract_types if not select_all_types else TIPOS_CONTRATACION
    
    # Mostrar información de la consulta
    with st.container():
        st.info(f"📡 Consultando datos para: **{year}** | **{len(region)} provincia(s)** | **{len(types_to_query)} tipo(s) de contratación**")
        
        # Realizar consulta combinada
        with st.spinner(f'Obteniendo datos de {len(region)} provincia(s) y {len(types_to_query)} tipo(s)...'):
            combined_data = get_combined_data(year, region, types_to_query)
    
    # =====================================================
    # VISUALIZACIONES
    # =====================================================
    
    if combined_data is not None:
        # Procesar datos combinados
        df = process_data(combined_data)
        
        if df is not None and len(df) > 0:
            # Mostrar resumen de datos
            st.success(f"✅ Datos procesados exitosamente: **{len(df)}** registros")
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Registros", len(df))
            
            with col2:
                st.metric("💰 Monto Total", f"${df['total'].sum():,.2f}")
            
            with col3:
                st.metric("📋 Contratos Totales", f"{df['contracts'].sum():,}")
            
            with col4:
                st.metric("🏢 Tipos de Contratación", df['internal_type'].nunique())
            
            # Mostrar distribución por provincias y tipos si hay múltiples
            if len(region) > 1:
                st.markdown("### 🗺️ Distribución por Provincias")
                
                # Gráfica de distribución por provincias
                province_summary = df.groupby('provincia').agg({
                    'total': 'sum',
                    'contracts': 'sum'
                }).reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_prov_total = px.bar(
                        province_summary,
                        x='provincia',
                        y='total',
                        title='💰 Total de Montos por Provincia',
                        labels={'provincia': 'Provincia', 'total': 'Monto Total (USD)'}
                    )
                    fig_prov_total.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_prov_total, use_container_width=True)
                
                with col2:
                    fig_prov_contracts = px.bar(
                        province_summary,
                        x='provincia',
                        y='contracts',
                        title='📋 Total de Contratos por Provincia',
                        labels={'provincia': 'Provincia', 'contracts': 'Cantidad de Contratos'}
                    )
                    fig_prov_contracts.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_prov_contracts, use_container_width=True)
                
                st.markdown("---")
            
            if len(types_to_query) > 1:
                st.markdown("### 📋 Distribución por Tipos de Contratación")
                
                # Gráfica de distribución por tipos
                type_summary = df.groupby('tipo_consulta').agg({
                    'total': 'sum',
                    'contracts': 'sum'
                }).reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_type_total = px.bar(
                        type_summary,
                        x='tipo_consulta',
                        y='total',
                        title='💰 Total de Montos por Tipo de Contratación',
                        labels={'tipo_consulta': 'Tipo de Contratación', 'total': 'Monto Total (USD)'}
                    )
                    fig_type_total.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_type_total, use_container_width=True)
                
                with col2:
                    fig_type_contracts = px.bar(
                        type_summary,
                        x='tipo_consulta',
                        y='contracts',
                        title='📋 Total de Contratos por Tipo',
                        labels={'tipo_consulta': 'Tipo de Contratación', 'contracts': 'Cantidad de Contratos'}
                    )
                    fig_type_contracts.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_type_contracts, use_container_width=True)
                
                st.markdown("---")
            
            st.subheader("📊 Visualizaciones")
            
            # Mostrar todas las gráficas
            chart_functions = {
                "📊 Gráfica de Barras - Total por Tipo": create_bar_chart,
                "📈 Evolución Mensual - Líneas": create_line_chart,
                "📊 Barras Apiladas - Mensual por Tipo": create_stacked_bar_chart,
                "🥧 Gráfica de Pastel - Proporción Contratos": create_pie_chart,
                "🎯 Dispersión - Total vs Contratos": create_scatter_chart,
                "📈 Comparativa Líneas - Tipos por Año": create_comparative_line_chart
            }
            
            for chart_title, chart_function in chart_functions.items():
                chart_type = TIPOS_GRAFICAS[chart_title]
                
                # Crear contenedor para cada gráfica
                with st.container():
                    st.markdown(f"### {chart_title}")
                    
                    # Crear la gráfica
                    fig = chart_function(df)
                    
                    # Mostrar la gráfica
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar descripción
                    st.info(get_chart_description(chart_type))
                    
                    st.markdown("---")
            
            # Agregar sección de modelos de predicción
            show_prediction_models(df)
            
            # Sección de datos detallados
            with st.expander("📋 Ver Datos Detallados"):
                st.dataframe(df, use_container_width=True)
                
                # Estadísticas adicionales
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Estadísticas Básicas")
                    st.write(df[['total', 'contracts']].describe())
                
                with col2:
                    st.markdown("### 🏢 Distribución por Tipo")
                    type_summary = df.groupby('internal_type').agg({
                        'total': ['sum', 'mean'],
                        'contracts': ['sum', 'mean']
                    }).round(2)
                    st.write(type_summary)
                
                # Opción para descargar datos
                csv = df.to_csv(index=False)
                filename = f"compras_publicas_{year}_{region}_{len(types_to_query)}_tipos.csv"
                st.download_button(
                    label="💾 Descargar datos como CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.warning("⚠️ No se pudieron procesar los datos correctamente.")
    
    else:
        st.warning('⚠️ No se encontraron datos para esta combinación de filtros.')
        
        # Sugerencias más específicas
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("💡 **Sugerencias para encontrar datos:**\n"
                   f"- Selecciona **múltiples tipos de contratación** o marca 'Todos los tipos'\n"
                   f"- Prueba con **otro año** (2020-2023 suelen tener más datos)\n"
                   f"- Cambia la **provincia** (ej: GUAYAS, PICHINCHA)\n"
                   f"- Los tipos como 'Todas', 'Menor Cuantía' suelen tener más datos")
        
        with col2:
            st.info("🔍 **Recomendaciones específicas:**\n"
                   f"- **Año actual:** {year} - Prueba 2022 o 2023\n"
                   f"- **Provincia actual:** {region}\n"
                   f"- **Tipos actuales:** {len(types_to_query)} seleccionado(s)\n"
                   f"- Combinar múltiples tipos aumenta las posibilidades de datos")
        
        # Mostrar combinaciones sugeridas
        with st.expander("📋 Combinaciones sugeridas que suelen tener datos"):
            st.markdown("""
            **Combinaciones con alta probabilidad de datos:**
            - **2023 | GUAYAS | Múltiples tipos**
            - **2022 | PICHINCHA | Todos los tipos**  
            - **2021 | GUAYAS | Licitación + Menor Cuantía**
            - **2020 | PICHINCHA | Todos los tipos**
            - **2023 | Provincias grandes | Cotización + Licitación**
            
            *Seleccionar múltiples tipos de contratación aumenta significativamente las posibilidades de obtener datos*
            """)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666; padding: 20px;'>
        📊 <strong>Aplicación de Análisis de Compras Públicas con IA</strong><br>
        🏛️ Datos oficiales del Sistema Nacional de Contratación Pública (SERCOP) - Ecuador<br>
        🤖 Modelos de Machine Learning integrados para análisis predictivo<br>
        🔗 <a href="https://datosabiertos.compraspublicas.gob.ec" target="_blank">Portal de Datos Abiertos</a>
    </div>
    """, 
    unsafe_allow_html=True
)