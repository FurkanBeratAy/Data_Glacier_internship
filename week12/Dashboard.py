import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

file_path = 'Test.csv'
data = pd.read_csv(file_path)
data['gross_income'] = pd.to_numeric(data['gross_income'], errors='coerce')
data = data.dropna(subset=['prov_name', 'segmentation', 'gross_income'])

app = dash.Dash(__name__)
app.title = "Client Data Dashboard"

app.layout = html.Div([
    html.H1("Client Data Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Select Province:"),
        dcc.Dropdown(
            id='province-dropdown',
            options=[{'label': prov, 'value': prov} for prov in data['prov_name'].unique()],
            value=None,
            placeholder="Select a province (Optional)"
        ),
    ], style={'width': '50%', 'margin': '0 auto'}),
    dcc.Graph(id='age-distribution'),
    dcc.Graph(id='income-distribution'),
    dcc.Graph(id='segmentation-analysis')
])

@app.callback(
    [Output('age-distribution', 'figure'),
     Output('income-distribution', 'figure'),
     Output('segmentation-analysis', 'figure')],
    [Input('province-dropdown', 'value')]
)
def update_graphs(selected_province):
    filtered_data = data if not selected_province else data[data['prov_name'] == selected_province]
    age_fig = px.histogram(filtered_data, x='age', nbins=30, title='Age Distribution')
    income_fig = px.histogram(filtered_data, x='gross_income', nbins=30, title='Gross Income Distribution')
    segmentation_fig = px.pie(filtered_data, names='segmentation', title='Segmentation Analysis')
    return age_fig, income_fig, segmentation_fig

if __name__ == '__main__':
    app.run_server(debug=True)
