from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importar los datos
df = pd.read_csv('housing_time_series_by_madrid_neighbourhood.csv', header=0, sep=',')

# Asegúrate de que las columnas sean del tipo correcto
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Gráfico 1: Crecimiento Anual del Precio por m² por Neighbourhood Group
fig1 = px.line(
    df, 
    x='date', 
    y='m2_price', 
    color='neighbourhood_group', 
    markers=True,
    title='Crecimiento Anual del Precio por m² por Neighbourhood Group',
    labels={
        'date': 'Fecha',
        'm2_price': 'Precio Promedio por m² (€)',
        'neighbourhood_group': 'Grupo de Vecindarios'
    }
)
fig1.update_traces(marker=dict(size=4), line=dict(width=2))
fig1.update_layout(
    legend_title=dict(font=dict(size=20)),
    legend=dict(orientation='v', x=1.05, y=1),
    margin=dict(l=40, r=40, t=40, b=40),
    xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
    yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
    title=dict(font=dict(size=18))
)

# Gráfico 2: Patrones Estacionales
monthly_data = df.groupby('month').agg({
    'reviews_per_month': 'mean',
    'listings_count': 'mean',
    'nigth_price': 'mean',
}).reset_index()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=monthly_data['month'],
    y=monthly_data['listings_count'],
    mode='lines+markers',
    name='Propiedades en Airbnb',
    line=dict(color='black'),
    marker=dict(size=8),
    yaxis='y'
))
fig2.add_trace(go.Scatter(
    x=monthly_data['month'],
    y=monthly_data['nigth_price'],
    mode='lines+markers',
    name='Precio medio de las viviendas',
    line=dict(color='blue'),
    marker=dict(size=8),
    yaxis='y2'
))
fig2.update_layout(
    title='Patrones Estacionales: Disponibilidad y Precio',
    xaxis=dict(
        title='Mes',
        tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                  'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    ),
    yaxis=dict(
        title='Propiedades en Airbnb',
        titlefont=dict(color='black'),
        tickfont=dict(color='black')
    ),
    yaxis2=dict(
        title='Precio medio de las viviendas (€)',
        titlefont=dict(color='blue'),
        tickfont=dict(color='blue'),
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0)', bordercolor='rgba(0,0,0,0)'),
    template='plotly_white',
    margin=dict(l=40, r=40, t=60, b=40)
)

# Gráfico 3: Número de Anfitriones vs Precio por m² por Neighbourhood Group
grouped = df.groupby('neighbourhood_group').agg({
    'hosts_count': 'mean',
    'm2_price': 'mean'
}).reset_index()
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=grouped['hosts_count'],
    y=grouped['m2_price'],
    mode='markers',
    marker=dict(
        size=grouped['hosts_count'],
        sizemode='area',
        sizeref=2.*max(grouped['hosts_count'])/(100.**2),
        sizemin=10,
        color=grouped['m2_price'],
        colorscale='Viridis',
        colorbar=dict(title='Precio por m² (€)'),
        showscale=True
    ),
    text=grouped['neighbourhood_group'],
    hovertemplate=
        '<b>%{text}</b><br>' +
        'Anfitriones: %{x}<br>' +
        'Precio por m²: %{y}€<br>' +
        '<extra></extra>',
    name='Neighbourhood Groups'
))
fig3.update_layout(
    title='Número de Anfitriones vs Precio por m² por Neighbourhood Group',
    xaxis_title='Número de Anfitriones',
    yaxis_title='Precio por m² (€)',
    template='plotly_white',
    hovermode='closest'
)

# Gráfico 4: Importancia de las Variables en la Regresión Lineal
# Preparar datos para el modelo
df = pd.read_csv('housing_time_series_by_madrid_neighbourhood.csv', header=0, sep=',')
features = df.groupby('date').mean().drop(['m2_price'], axis=1)
target = df.groupby('date').mean()['m2_price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
coef_data = pd.DataFrame({
    'Característica': features.columns,
    'Coeficiente': model.coef_
})
fig4 = px.bar(
    coef_data,
    x='Coeficiente',
    y='Característica',
    orientation='h',
    title='Importancia de las Variables en la Regresión Lineal',
    labels={
        'Coeficiente': 'Coeficientes',
        'Característica': 'Características'
    }
)
fig4.update_layout(
    xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
    yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
    title=dict(font=dict(size=18)),
    margin=dict(l=40, r=40, t=40, b=40)
)

# Crear la aplicación Dash
app = Dash(__name__)

# Layout organizado en 2x2
app.layout = html.Div([
    html.H1("Dashboard de Análisis de Datos de Vivienda en Madrid", style={'textAlign': 'center'}),
    html.Div([
        html.Div([dcc.Graph(figure=fig1)], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=fig2)], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    html.Div([
        html.Div([dcc.Graph(figure=fig3)], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=fig4)], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justify-content': 'space-between'})
])

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
