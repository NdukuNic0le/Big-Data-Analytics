import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

results_df = spark.read.parquet("churn_predictions").toPandas()

with open('summary_stats.txt', 'r') as f:
    summary_stats = dict(line.strip().split(': ') for line in f)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Bank Customer Churn Dashboard'),
    
    html.Div([
        html.H3('Summary Statistics'),
        html.P(f"Average Age: {summary_stats['avg_age']}"),
        html.P(f"Average Balance: ${float(summary_stats['avg_balance']):,.2f}"),
        html.P(f"Overall Churn Rate: {float(summary_stats['churn_rate'])*100:.2f}%")
    ]),
    
    dcc.Graph(
        id='churn-prediction-chart',
        figure=px.histogram(results_df, x='prediction', title='Churn Prediction Distribution')
    ),
    
    dcc.Graph(
        id='churn-by-age',
        figure=px.scatter(results_df, x='age', y='prediction', title='Churn Probability vs Age')
    ),
    
    dcc.Graph(
        id='churn-by-balance',
        figure=px.scatter(results_df, x='balance', y='prediction', title='Churn Probability vs Account Balance')
    )
])

if __name__ == '__main__':
    app.run_server(debug=True) # when not in producion