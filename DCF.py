import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
from dcf_calculator import FinancialDataFetcher, WACCCalculator, DCFModel

app = dash.Dash(__name__)

colors = {
    'background': "#FFFFFF",
    'card': "#FFFFFF",
    'primary': '#007bff',
    'secondary': '#6c757d',
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'text': "#000000"
}

custom_styles = {
    'card': {
        'backgroundColor': colors['card'],
        'padding': '20px',
        'margin': '10px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'border': '1px solid #e9ecef'
    },
    'header': {
        'backgroundColor': colors['primary'],
        'color': 'white',
        'padding': '20px',
        'marginBottom': '20px',
        'borderRadius': '8px',
        'textAlign': 'center'
    },
    'metric_card': {
        'backgroundColor': colors['card'],
        'padding': '15px',
        'margin': '5px',
        'borderRadius': '6px',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
        'textAlign': 'center',
        'minHeight': '100px'
    }
}

app.layout = html.Div([
    html.Div([
        html.H1("DCF Valuation Dashboard", style={'margin': '0', 'fontSize': '2.5rem'}),
        html.P("Interactive Financial Modeling and Analysis", style={'margin': '5px 0 0 0', 'fontSize': '1.1rem'})
    ], style=custom_styles['header']),
    
    html.Div([
        html.Div([
            html.H3("Input Parameters", style={'marginBottom': '20px', 'color': colors['text']}),
            
            html.Div([
                html.Label("Stock Ticker:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(
                    id='ticker-input',
                    type='text',
                    value='AAPL',
                    placeholder='Enter ticker symbol (e.g., AAPL)',
                    style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}
                )
            ], style={'marginBottom': '15px'}),
            
            html.H4("Financial Assumptions", style={'marginTop': '25px', 'marginBottom': '15px', 'color': colors['secondary']}),
            
            html.Div([
                html.Div([
                    html.Label("Risk-Free Rate (%):", style={'fontWeight': 'bold'}),
                    dcc.Input(id='risk-free-rate', type='number', value=4.5, step=0.1, 
                             style={'width': '100%', 'padding': '5px', 'marginTop': '5px'})
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                
                html.Div([
                    html.Label("Market Risk Premium (%):", style={'fontWeight': 'bold'}),
                    dcc.Input(id='market-risk-premium', type='number', value=6.5, step=0.1,
                             style={'width': '100%', 'padding': '5px', 'marginTop': '5px'})
                ], style={'width': '48%', 'display': 'inline-block'})
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Div([
                    html.Label("Terminal Growth Rate (%):", style={'fontWeight': 'bold'}),
                    dcc.Input(id='terminal-growth-rate', type='number', value=2.5, step=0.1,
                             style={'width': '100%', 'padding': '5px', 'marginTop': '5px'})
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                
                html.Div([
                    html.Label("Projection Years:", style={'fontWeight': 'bold'}),
                    dcc.Input(id='projection-years', type='number', value=5, min=3, max=10,
                             style={'width': '100%', 'padding': '5px', 'marginTop': '5px'})
                ], style={'width': '48%', 'display': 'inline-block'})
            ], style={'marginBottom': '20px'}),
            
            html.Button(
                'Calculate DCF Valuation',
                id='calculate-btn',
                n_clicks=0,
                style={
                    'width': '100%',
                    'padding': '12px',
                    'backgroundColor': colors['primary'],
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '6px',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'cursor': 'pointer'
                }
            )
        ], style=custom_styles['card'])
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    
    html.Div([
        html.Div(id='key-metrics-cards', style={'marginBottom': '20px'}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='revenue-fcf-chart')
            ], style={**custom_styles['card'], 'marginBottom': '20px'}),
            
            html.Div([
                dcc.Graph(id='valuation-breakdown-chart')
            ], style={**custom_styles['card'], 'marginBottom': '20px'}),
            
            html.Div([
                dcc.Graph(id='wacc-components-chart')
            ], style={**custom_styles['card'], 'marginBottom': '20px'}),
            
            html.Div([
                dcc.Graph(id='sensitivity-chart')
            ], style={**custom_styles['card'], 'marginBottom': '20px'}),
            
            html.Div([
                html.H4("Key Assumptions", style={'marginBottom': '15px', 'color': colors['text']}),
                html.Div(id='assumptions-table')
            ], style=custom_styles['card'])
        ])
    ], style={'width': '68%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
    
    dcc.Store(id='dcf-results-store')
    
], style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})

@app.callback(
    Output('dcf-results-store', 'data'),
    [Input('calculate-btn', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('risk-free-rate', 'value'),
     State('market-risk-premium', 'value'),
     State('terminal-growth-rate', 'value'),
     State('projection-years', 'value')]
)
def calculate_dcf(n_clicks, ticker, risk_free_rate, market_risk_premium, terminal_growth_rate, projection_years):
    if n_clicks == 0 or not ticker:
        return {}
    
    try:
        dcf = DCFModel(ticker, projection_years, terminal_growth_rate/100)
        dcf.wacc_calculator.risk_free_rate = risk_free_rate/100
        dcf.wacc_calculator.market_risk_premium = market_risk_premium/100
        
        valuation = dcf.calculate_dcf_valuation()
        
        if valuation:
            return {
                'ticker': ticker.upper(),
                'valuation': valuation,
                'success': True
            }
        else:
            return {'success': False, 'error': 'Failed to fetch data or calculate valuation'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.callback(
    Output('key-metrics-cards', 'children'),
    [Input('dcf-results-store', 'data')]
)
def update_key_metrics(data):
    if not data or not data.get('success'):
        return html.Div("Enter a ticker symbol and click Calculate to see results.", 
                       style={'textAlign': 'center', 'padding': '20px', 'color': colors['secondary']})
    
    valuation = data['valuation']
    ticker = data['ticker']
    
    upside_downside = valuation['upside_downside']
    upside_color = colors['success'] if upside_downside > 0 else colors['danger']
    
    metrics_cards = [
        html.Div([
            html.H4(f"${valuation['current_price']:.2f}", style={'margin': '0', 'color': colors['text']}),
            html.P("Current Price", style={'margin': '5px 0 0 0', 'color': colors['secondary'], 'fontSize': '0.9rem'})
        ], style=custom_styles['metric_card']),
        
        html.Div([
            html.H4(f"${valuation['value_per_share']:.2f}", style={'margin': '0', 'color': colors['primary']}),
            html.P("DCF Fair Value", style={'margin': '5px 0 0 0', 'color': colors['secondary'], 'fontSize': '0.9rem'})
        ], style=custom_styles['metric_card']),
        
        html.Div([
            html.H4(f"{upside_downside:.1%}", style={'margin': '0', 'color': upside_color}),
            html.P("Upside/Downside", style={'margin': '5px 0 0 0', 'color': colors['secondary'], 'fontSize': '0.9rem'})
        ], style=custom_styles['metric_card']),
        
        html.Div([
            html.H4(f"{valuation['wacc']:.1%}", style={'margin': '0', 'color': colors['text']}),
            html.P("WACC", style={'margin': '5px 0 0 0', 'color': colors['secondary'], 'fontSize': '0.9rem'})
        ], style=custom_styles['metric_card']),
        
        html.Div([
            html.H4(f"${valuation['enterprise_value']/1e9:.1f}B", style={'margin': '0', 'color': colors['text']}),
            html.P("Enterprise Value", style={'margin': '5px 0 0 0', 'color': colors['secondary'], 'fontSize': '0.9rem'})
        ], style=custom_styles['metric_card'])
    ]
    
    return html.Div([
        html.H3(f"{ticker} - Key Valuation Metrics", style={'marginBottom': '15px', 'color': colors['text']}),
        html.Div(metrics_cards, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'})
    ])

@app.callback(
    Output('revenue-fcf-chart', 'figure'),
    [Input('dcf-results-store', 'data')]
)
def update_revenue_fcf_chart(data):
    if not data or not data.get('success'):
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    projections = data['valuation']['projections']
    
    years = [f"Year {p['year']}" for p in projections]
    revenues = [p['revenue']/1e9 for p in projections]
    fcfs = [p['fcf']/1e9 for p in projections]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=years, y=revenues, name="Revenue", marker_color=colors['primary'], opacity=0.7),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=years, y=fcfs, name="Free Cash Flow", line=dict(color=colors['success'], width=3),
                  mode='lines+markers', marker=dict(size=8)),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Projection Years")
    fig.update_yaxes(title_text="Revenue ($ Billions)", secondary_y=False)
    fig.update_yaxes(title_text="Free Cash Flow ($ Billions)", secondary_y=True)
    
    fig.update_layout(
        title="Revenue and Free Cash Flow Projections",
        hovermode="x unified",
        template="plotly_white",
        height=400
    )
    
    return fig

@app.callback(
    Output('valuation-breakdown-chart', 'figure'),
    [Input('dcf-results-store', 'data')]
)
def update_valuation_breakdown_chart(data):
    if not data or not data.get('success'):
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    valuation = data['valuation']
    
    pv_cash_flows = valuation['pv_cash_flows']
    pv_terminal_value = valuation['pv_terminal_value']
    
    fig = go.Figure(go.Waterfall(
        name="Valuation Components",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["PV of Cash Flows", "PV of Terminal Value", "Enterprise Value"],
        y=[pv_cash_flows/1e9, pv_terminal_value/1e9, 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": colors['success']}},
        decreasing={"marker": {"color": colors['danger']}},
        totals={"marker": {"color": colors['primary']}}
    ))
    
    fig.update_layout(
        title="Enterprise Value Breakdown",
        yaxis_title="Value ($ Billions)",
        template="plotly_white",
        height=400
    )
    
    return fig

@app.callback(
    Output('wacc-components-chart', 'figure'),
    [Input('dcf-results-store', 'data')]
)
def update_wacc_components_chart(data):
    if not data or not data.get('success'):
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    wacc_breakdown = data['valuation']['wacc_breakdown']
    
    labels = ['Equity', 'Debt']
    values = [wacc_breakdown['weight_equity'], wacc_breakdown['weight_debt']]
    colors_pie = [colors['primary'], colors['warning']]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Capital Structure', 'Cost Components'),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Pie(labels=labels, values=values, marker_colors=colors_pie, name="Weights"),
        row=1, col=1
    )
    
    costs = ['Cost of Equity', 'Cost of Debt', 'WACC']
    cost_values = [
        wacc_breakdown['cost_of_equity'] * 100,
        wacc_breakdown['cost_of_debt'] * 100,
        wacc_breakdown['wacc'] * 100
    ]
    
    fig.add_trace(
        go.Bar(x=costs, y=cost_values, marker_color=[colors['primary'], colors['warning'], colors['success']], name="Costs"),
        row=1, col=2
    )
    
    fig.update_layout(
        title="WACC Analysis",
        template="plotly_white",
        height=400
    )
    
    fig.update_yaxes(title_text="Rate (%)", row=1, col=2)
    
    return fig

@app.callback(
    Output('sensitivity-chart', 'figure'),
    [Input('dcf-results-store', 'data')]
)
def update_sensitivity_chart(data):
    if not data or not data.get('success'):
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    valuation = data['valuation']
    base_value = valuation['value_per_share']
    
    wacc_range = np.arange(0.06, 0.16, 0.01)  # 6% to 15%
    terminal_growth_range = np.arange(0.01, 0.05, 0.005)  # 1% to 4.5%
    
    sensitivity_data = []
    
    for wacc in wacc_range:
        row = []
        for terminal_growth in terminal_growth_range:
            wacc_impact = (valuation['wacc'] - wacc) / valuation['wacc'] * base_value * 0.5
            terminal_impact = (terminal_growth - 0.025) / 0.025 * base_value * 0.3
            adjusted_value = base_value + wacc_impact + terminal_impact
            row.append(adjusted_value)
        sensitivity_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=sensitivity_data,
        x=[f"{tg:.1%}" for tg in terminal_growth_range],
        y=[f"{w:.1%}" for w in wacc_range],
        colorscale='RdYlGn',
        zmid=base_value
    ))
    
    fig.update_layout(
        title="Sensitivity Analysis: Value per Share",
        xaxis_title="Terminal Growth Rate",
        yaxis_title="WACC",
        template="plotly_white",
        height=400
    )
    
    return fig

@app.callback(
    Output('assumptions-table', 'children'),
    [Input('dcf-results-store', 'data')]
)
def update_assumptions_table(data):
    if not data or not data.get('success'):
        return html.Div("No assumptions to display.")
    
    assumptions = data['valuation']['assumptions']
    wacc_breakdown = data['valuation']['wacc_breakdown']
    
    table_data = [
        {'Assumption': 'Revenue Growth Rate', 'Value': f"{assumptions['revenue_growth']:.1%}"},
        {'Assumption': 'EBIT Margin', 'Value': f"{assumptions['ebit_margin']:.1%}"},
        {'Assumption': 'Tax Rate', 'Value': f"{assumptions['tax_rate']:.1%}"},
        {'Assumption': 'Capex as % of Revenue', 'Value': f"{assumptions['capex_ratio']:.1%}"},
        {'Assumption': 'Beta', 'Value': f"{wacc_breakdown['beta']:.2f}"},
        {'Assumption': 'Risk-Free Rate', 'Value': f"{wacc_breakdown['cost_of_equity'] - wacc_breakdown['beta'] * 0.065:.1%}"},
        {'Assumption': 'Market Risk Premium', 'Value': "6.5%"}
    ]
    
    return dash_table.DataTable(
        data=table_data,
        columns=[{"name": "Assumption", "id": "Assumption"}, {"name": "Value", "id": "Value"}],
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'},
        style_data={'backgroundColor': colors['background']},
        style_table={'overflowX': 'auto'}
    )

if __name__ == '__main__':
    app.run (debug=True, port=8050)