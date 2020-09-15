import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np

import plotly
import plotly.graph_objects as go
import etl
from scipy.integrate import odeint
from scipy.optimize import minimize,curve_fit

app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True
app.title = "Anuraag's Dashboard"

data = pd.read_csv('data/dashboard_data.csv')
data['date'] = pd.to_datetime(data['date'])
# selects the "data last updated" date
update = data['date'].dt.strftime('%B %d, %Y').iloc[-1]

dash_colors = {
    'background': '#FFFFFF',
    'text': '#111111',
    'grid': '#333333',
    'red': '#BF0000',
    'blue': '#466fc2',
    'green': '#5bc246'
}

available_countries = sorted(data['Country/Region'].unique())
region_options = {'Worldwide': available_countries}

df_worldwide = pd.read_csv('data/df_worldwide.csv')
df_worldwide['percentage'] = df_worldwide['percentage'].astype(str)



@app.callback(
    Output('confirmed_ind', 'figure'),
    [Input('global_format', 'value')])
def confirmed(view):
    '''
    creates the CUMULATIVE CONFIRMED indicator
    '''
    df=data
    value = df[df['date'] == df['date'].iloc[-1]]['Confirmed'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Confirmed'].sum()
    return {
            'data': [{'type': 'indicator',
                    'mode': 'number+delta',
                    'value': value,
                    'delta': {'reference': delta,
                              'valueformat': ',g',
                              'relative': False,
                              'increasing': {'color': dash_colors['blue']},
                              'decreasing': {'color': dash_colors['green']},
                              'font': {'size': 25}},
                    'number': {'valueformat': ',',
                              'font': {'size': 50}},
                    'domain': {'y': [0, 1], 'x': [0, 1]}}],
            'layout': go.Layout(
                title={'text': "CUMULATIVE CONFIRMED"},
                font=dict(color=dash_colors['red']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                height=200
                )
            }

    

@app.callback(
    Output('active_ind', 'figure'),
    [Input('global_format', 'value')])
def active(view):
    '''
    creates the CURRENTLY ACTIVE indicator
    '''
    df=data

    value = df[df['date'] == df['date'].iloc[-1]]['Active'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Active'].sum()
    return {
            'data': [{'type': 'indicator',
                    'mode': 'number+delta',
                    'value': value,
                    'delta': {'reference': delta,
                              'valueformat': ',g',
                              'relative': False,
                              'increasing': {'color': dash_colors['blue']},
                              'decreasing': {'color': dash_colors['green']},
                              'font': {'size': 25}},
                    'number': {'valueformat': ',',
                              'font': {'size': 50}},
                    'domain': {'y': [0, 1], 'x': [0, 1]}}],
            'layout': go.Layout(
                title={'text': "CURRENTLY ACTIVE"},
                font=dict(color=dash_colors['red']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                height=200
                )
            }

@app.callback(
    Output('recovered_ind', 'figure'),
    [Input('global_format', 'value')])
def recovered(view):
    '''
    creates the RECOVERED CASES indicator
    '''
    df = data

    value = df[df['date'] == df['date'].iloc[-1]]['Recovered'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Recovered'].sum()
    return {
            'data': [{'type': 'indicator',
                    'mode': 'number+delta',
                    'value': value,
                    'delta': {'reference': delta,
                              'valueformat': ',g',
                              'relative': False,
                              'increasing': {'color': dash_colors['blue']},
                              'decreasing': {'color': dash_colors['green']},
                              'font': {'size': 25}},
                    'number': {'valueformat': ',',
                              'font': {'size': 50}},
                    'domain': {'y': [0, 1], 'x': [0, 1]}}],
            'layout': go.Layout(
                title={'text': "RECOVERED CASES"},
                font=dict(color=dash_colors['red']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                height=200
                )
            }

@app.callback(
    Output('deaths_ind', 'figure'),
    [Input('global_format', 'value')])
def deaths(view):
    '''
    creates the DEATHS TO DATE indicator
    '''
    df = data

    value = df[df['date'] == df['date'].iloc[-1]]['Deaths'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Deaths'].sum()
    return {
            'data': [{'type': 'indicator',
                    'mode': 'number+delta',
                    'value': value,
                    'delta': {'reference': delta,
                              'valueformat': ',g',
                              'relative': False,
                              'increasing': {'color': dash_colors['blue']},
                              'decreasing': {'color': dash_colors['green']},
                              'font': {'size': 25}},
                    'number': {'valueformat': ',',
                              'font': {'size': 50}},
                    'domain': {'y': [0, 1], 'x': [0, 1]}}],
            'layout': go.Layout(
                title={'text': "DEATHS TO DATE"},
                font=dict(color=dash_colors['red']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                height=200
                )
            }

@app.callback(
    Output('worldwide_trend', 'figure'),
    [Input('global_format', 'value'),
     Input('population_select', 'value')])
def worldwide_trend(view, population):
    '''
    creates the upper-left chart (aggregated stats for the view)
    '''
    df = df_worldwide

    if population == 'absolute':
        confirmed = df.groupby('date')['Confirmed'].sum()
        active = df.groupby('date')['Active'].sum()
        recovered = df.groupby('date')['Recovered'].sum()
        deaths = df.groupby('date')['Deaths'].sum()
        title_suffix = ''
        hover = '%{y:,g}'
    elif population == 'percent':
        df = df.dropna(subset=['population'])
        confirmed = df.groupby('date')['Confirmed'].sum() / df.groupby('date')['population'].sum()
        active = df.groupby('date')['Active'].sum() / df.groupby('date')['population'].sum()
        recovered = df.groupby('date')['Recovered'].sum() / df.groupby('date')['population'].sum()
        deaths = df.groupby('date')['Deaths'].sum() / df.groupby('date')['population'].sum()
        title_suffix = ' per 100,000 people'
        hover = '%{y:,.2f}'
    else:
        confirmed = df.groupby('date')['Confirmed'].sum()
        active = df.groupby('date')['Active'].sum()
        recovered = df.groupby('date')['Recovered'].sum()
        deaths = df.groupby('date')['Deaths'].sum()
        title_suffix = ''
        hover = '%{y:,g}'

    traces = [go.Scatter(
                    x=df.groupby('date')['date'].first(),
                    y=confirmed,
                    hovertemplate=hover,
                    name="Confirmed",
                    mode='lines'),
                go.Scatter(
                    x=df.groupby('date')['date'].first(),
                    y=active,
                    hovertemplate=hover,
                    name="Active",
                    mode='lines'),
                go.Scatter(
                    x=df.groupby('date')['date'].first(),
                    y=recovered,
                    hovertemplate=hover,
                    name="Recovered",
                    mode='lines'),
                go.Scatter(
                    x=df.groupby('date')['date'].first(),
                    y=deaths,
                    hovertemplate=hover,
                    name="Deaths",
                    mode='lines')]
    return {
            'data': traces,
            'layout': go.Layout(
                title="{} Infections{}".format(view, title_suffix),
                xaxis_title="Date",
                yaxis_title="Number of Cases",
                font=dict(color=dash_colors['text']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                xaxis=dict(gridcolor=dash_colors['grid']),
                yaxis=dict(gridcolor=dash_colors['grid'])
                )
            }

@app.callback(
    Output('country_select', 'options'),
    [Input('global_format', 'value')])
def set_active_options(selected_view):
    '''
    sets allowable options for regions in the upper-right chart drop-down
    '''
    return [{'label': i, 'value': i} for i in region_options[selected_view]]

@app.callback(
    Output('country_select', 'value'),
    [Input('global_format', 'value'),
     Input('country_select', 'options')])
def set_countries_value(view, available_options):

    return ['India', 'Italy','Germany','United Kingdom', 'Spain', 'France', 'Russia', 'Brazil', 'Sweden', 'Belgium', 'Peru']

@app.callback(
    Output('active_countries', 'figure'),
    [Input('global_format', 'value'),
     Input('country_select', 'value'),
     Input('column_select', 'value'),
     Input('population_select', 'value')])
def active_countries(view, countries, column, population):
    '''
    creates the upper-right chart (sub-region analysis)
    '''
    df = df_worldwide

    if population == 'absolute':
        column_label = column
        hover = '%{y:,g}<br>%{x}'
    elif population == 'percent':
        column_label = '{} per 100,000'.format(column)
        df = df.dropna(subset=['population'])
        hover = '%{y:,.2f}<br>%{x}'
    else:
        column_label = column
        hover = '%{y:,g}<br>%{x}'

    traces = []
    countries = df[(df['Country/Region'].isin(countries)) &
                   (df['date'] == df['date'].max())].groupby('Country/Region')['Active'].sum().sort_values(ascending=False).index.to_list()
    for country in countries:
        if population == 'absolute':
            y_data = df[df['Country/Region'] == country].groupby('date')[column].sum()
            recovered = df[df['Country/Region'] == 'Recovered'].groupby('date')[column].sum()
        elif population == 'percent':
            y_data = df[df['Country/Region'] == country].groupby('date')[column].sum() / df[df['Country/Region'] == country].groupby('date')['population'].first()
            recovered = df[df['Country/Region'] == 'Recovered'].groupby('date')[column].sum() / df[df['Country/Region'] == country].groupby('date')['population'].first()
        else:
            y_data = df[df['Country/Region'] == country].groupby('date')[column].sum()
            recovered = df[df['Country/Region'] == 'Recovered'].groupby('date')[column].sum()

        traces.append(go.Scatter(
                    x=df[df['Country/Region'] == country].groupby('date')['date'].first(),
                    y=y_data,
                    hovertemplate=hover,
                    name=country,
                    mode='lines'))
    if column == 'Recovered':
        traces.append(go.Scatter(
                    x=df[df['Country/Region'] == 'Recovered'].groupby('date')['date'].first(),
                    y=recovered,
                    hovertemplate=hover,
                    name='Unidentified',
                    mode='lines'))
    return {
            'data': traces,
            'layout': go.Layout(
                    title="{} by Region".format(column_label),
                    xaxis_title="Date",
                    yaxis_title="Number of Cases",
                    font=dict(color=dash_colors['text']),
                    paper_bgcolor=dash_colors['background'],
                    plot_bgcolor=dash_colors['background'],
                    xaxis=dict(gridcolor=dash_colors['grid']),
                    yaxis=dict(gridcolor=dash_colors['grid']),
                    hovermode='closest'
                )
            }

@app.callback(
    Output('world_map', 'figure'),
    [Input('global_format', 'value'),
     Input('date_slider', 'value')])
def world_map(view, date_index):
    '''
    creates the lower-left chart (map)
    '''
    df = df_worldwide
    scope = 'world'
    projection_type = 'natural earth'
    sizeref = 35

    df = df[(df['date'] == df['date'].unique()[date_index]) & (df['Confirmed'] > 0)]
    return {
            'data': [
                go.Scattergeo(
                    lon = df['Longitude'],
                    lat = df['Latitude'],
                    text = df['Country/Region'] + ': ' +\
                        ['{:,}'.format(i) for i in df['Confirmed']] +\
                        ' total cases, ' + df['percentage'] +\
                        '% from previous week',
                    hoverinfo = 'text',
                    mode = 'markers',
                    marker = dict(reversescale = False,
                        autocolorscale = False,
                        symbol = 'circle',
                        size = np.sqrt(df['Confirmed']),
                        sizeref = sizeref,
                        sizemin = 0,
                        line = dict(width=.5, color='rgba(0, 0, 0)'),
                        colorscale = 'Reds',
                        cmin = 0,
                        color = df['share_of_last_week'],
                        cmax = 100,
                        colorbar = dict(
                            title = "Percentage of<br>cases occurring in<br>the previous week",
                            thickness = 30)
                        )
                    )
            ],
            'layout': go.Layout(
                title ='Number of Cumulative Confirmed Cases (size of marker)<br>and Share of New Cases from the Previous Week (color)',
                geo=dict(scope=scope,
                        projection_type=projection_type,
                        showland = True,
                        landcolor = "rgb(100, 125, 100)",
                        showocean = True,
                        oceancolor = "rgb(80, 150, 250)",
                        showcountries=True,
                        showlakes=True),
                font=dict(color=dash_colors['text']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background']
            )
        }

def hex_to_rgba(h, alpha=1):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])

@app.callback(
    Output('trajectory', 'figure'),
    [Input('global_format', 'value'),
     Input('date_slider', 'value')])
def trajectory(view, date_index):
    '''
    creates the lower-right chart (trajectory)
    '''
    df = data
    scope = 'countries'
    threshold = 1000
    
    date = data['date'].unique()[date_index]

    df = df.groupby(['date', 'Country/Region'], as_index=False)['Confirmed'].sum()
    df['previous_week'] = df.groupby(['Country/Region'])['Confirmed'].shift(7, fill_value=0)
    df['new_cases'] = df['Confirmed'] - df['previous_week']
    df['new_cases'] = df['new_cases'].clip(lower=0)

    xmax = np.log(1.25 * df['Confirmed'].max()) / np.log(10)
    xmin = np.log(threshold) / np.log(10)
    ymax = np.log(1.25 * df['new_cases'].max()) / np.log(10)
    ymin = np.log(10)

    countries_full = df.groupby(by='Country/Region', as_index=False)['Confirmed'].max().sort_values(by='Confirmed', ascending=False)['Country/Region'].to_list()
    
    df = df[df['date'] <= date]

    countries = df.groupby(by='Country/Region', as_index=False)['Confirmed'].max().sort_values(by='Confirmed', ascending=False)
    countries = countries[countries['Confirmed'] > threshold]['Country/Region'].to_list()
    countries = [country for country in countries_full if country in countries]

    traces = []
    trace_colors = plotly.colors.qualitative.D3
    color_idx = 0

    for country in countries:
        filtered_df = df[df['Country/Region'] == country].reset_index()
        idx = filtered_df['Confirmed'].sub(threshold).gt(0).idxmax()
        trace_data = filtered_df[idx:].copy()
        trace_data['date'] = pd.to_datetime(trace_data['date'])
        trace_data['date'] = trace_data['date'].dt.strftime('%b %d, %Y')

        marker_size = [0] * (len(trace_data) - 1) + [10]
        color = trace_colors[color_idx % len(trace_colors)]
        marker_color = 'rgba' + str(hex_to_rgba(color, 1))
        line_color = 'rgba' + str(hex_to_rgba(color, .5))

        traces.append(
            go.Scatter(
                    x=trace_data['Confirmed'],
                    y=trace_data['new_cases'],
                    mode='lines+markers',
                    marker=dict(color=marker_color,
                                size=marker_size,
                                line=dict(width=0)),
                    line=dict(color=line_color, width=2),
                    name=country,
                    text = ['{}: {:,} confirmed; {:,} from previous week'.format(country,
                                                                                trace_data['Confirmed'].iloc[i],
                                                                                trace_data['new_cases'].iloc[i]) \
                                                                                    for i in range(len(trace_data))],
                    hoverinfo='text')
        )

        color_idx += 1

    return {
        'data': traces,
        'layout': go.Layout(
                title='Trajectory of Cases<br>'.format(scope, threshold),
                xaxis_type="log",
                yaxis_type="log",
                xaxis_title='Total Confirmed Cases',
                yaxis_title='New Confirmed Cases (in the past week)',
                font=dict(color=dash_colors['text']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                xaxis=dict(gridcolor=dash_colors['grid'],
                           range=[xmin, xmax]),
                yaxis=dict(gridcolor=dash_colors['grid'],
                           range=[ymin, ymax]),
                hovermode='closest',
                showlegend=True
            )
        }
                    

@app.callback(Output('sir_simulations', 'figure'),
[Input('sir_list', 'value')])                    
def sir_simulations(country):

    N = 1000000
    betas = []
    gammas = []
    simulations = []
    countrydata=df_worldwide[df_worldwide['Country/Region']==country]
    infections, recoveries = countrydata.Confirmed.T, countrydata.Recovered.T
    def SIR(y, t, beta, gamma):    
        S = y[0]
        I = y[1]
        R = y[2]
        return -beta*S*I/N, (beta*S*I)/N-(gamma*I), gamma*I

    def fit_odeint(t,beta, gamma):
        return odeint(SIR,(s_0,i_0,r_0), t, args = (beta,gamma))[:,1]
    
    def loss(point, data, s_0, i_0, r_0):
        predict = fit_odeint(t, *point)
        l1 = np.sqrt(np.mean((predict - data)**2))
        return l1
  
    for index in range(len(infections)):
        if index % 7== 0:
            if index+1 <= len(data)-7:
                train = infections.values[index:index+7]
            else:
                train =  infections.values[index:]
            i_0 = train[0]
            r_0 = recoveries.values[index]
            s_0 = N - i_0 - r_0

            if len(train) > 2:
                t = np.arange(len(train))
                params, cerr = curve_fit(fit_odeint,t, train)
                optimal = minimize(loss, params, args=(train, s_0, i_0, r_0))
                beta,gamma = optimal.x
                betas.append(beta)
                gammas.append(gamma)
                
            predict = list(fit_odeint(np.arange(7),beta,gamma))
            simulations.extend(predict)
    i_0 =  infections.values[-1]
    r_0 =  recoveries.values[-1]
    s_0 = N - i_0 - r_0 
    future_simulations = list(fit_odeint(np.arange(7), beta, gamma ))
    start_date = np.array('2020-01-22', dtype=np.datetime64)
    dates = start_date + np.arange(len(countrydata.Confirmed))
    future_dates = dates[-1] + np.arange(7)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = dates, y=simulations,
                        mode='lines+markers',
                        name='Simulated'))
    fig.add_bar(x = dates, y= infections.values, name = "Actual",marker_color='red')
    fig.add_bar(x = future_dates, y= future_simulations, name = "Expected",marker_color='yellow')
    fig.update_layout(height = 800,  xaxis_title="Date",
    yaxis_title="Infections",  hovermode='x unified',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
 )
    return fig



    
app.layout = html.Div(style={'backgroundColor': dash_colors['background']}, children=[
    html.H1(children='COVID-19 Dashboard by Anuraag Sharma Thotapally',
        style={
            'textAlign': 'center',
            'color': dash_colors['text']
            }
        ),

    html.Div(children='Data last updated {} end-of-day'.format(update), style={
        'textAlign': 'center',
        'color': dash_colors['text']
        }),
    

    html.Div(dcc.RadioItems(id='global_format',
            options=[{'label': i, 'value': i} for i in ['Worldwide']],
            value='Worldwide',
            labelStyle={'float': 'center', 'display': 'inline-block'}
            ), style={'textAlign': 'center',
                'color': dash_colors['background'],
                'width': '100%',
                'float': 'center',
                'display': 'inline-block'
            }
        ),

    html.Div(dcc.Graph(id='confirmed_ind'),
        style={
            'textAlign': 'center',
            'color': dash_colors['red'],
            'width': '25%',
            'float': 'left',
            'display': 'inline-block'
            }
        ),

    html.Div(dcc.Graph(id='active_ind'),
        style={
            'textAlign': 'center',
            'color': dash_colors['red'],
            'width': '25%',
            'float': 'left',
            'display': 'inline-block'
            }
        ),

    html.Div(dcc.Graph(id='deaths_ind'),
        style={
            'textAlign': 'center',
            'color': dash_colors['red'],
            'width': '25%',
            'float': 'left',
            'display': 'inline-block'
            }
        ),

    html.Div(dcc.Graph(id='recovered_ind'),
        style={
            'textAlign': 'center',
            'color': dash_colors['red'],
            'width': '25%',
            'float': 'left',
            'display': 'inline-block'
            }
        ),

    html.Div(dcc.Markdown('Display data in the below two charts as total values or as values relative to population:'),
        style={
            'textAlign': 'center',
            'color': dash_colors['text'],
            'width': '100%',
            'float': 'center',
            'display': 'inline-block'}),

    html.Div(dcc.RadioItems(id='population_select',
            options=[{'label': 'Total values', 'value': 'absolute'},
                        {'label': 'Values per 100,000 of population', 'value': 'percent'}],
            value='absolute',
            labelStyle={'float': 'center', 'display': 'inline-block'},
            style={'textAlign': 'center',
                'color': dash_colors['text'],
                'width': '100%',
                'float': 'center',
                'display': 'inline-block'
                })
        ),

    html.Div(  # worldwide_trend and active_countries
        [
            html.Div(
                dcc.Graph(id='worldwide_trend'),
                style={'width': '50%', 'float': 'left', 'display': 'inline-block'}
                ),
            html.Div([
                dcc.Graph(id='active_countries'),
                html.Div([
                    dcc.RadioItems(
                        id='column_select',
                        options=[{'label': i, 'value': i} for i in ['Confirmed', 'Active', 'Recovered', 'Deaths']],
                        value='Active',
                        labelStyle={'float': 'center', 'display': 'inline-block'},
                        style={'textAlign': 'center',
                            'color': dash_colors['text'],
                            'width': '100%',
                            'float': 'center',
                            'display': 'inline-block'
                            }),
                    dcc.Dropdown(
                        id='country_select',
                        multi=True,
                        style={'width': '95%', 'float': 'center'}
                        )],
                    style={'width': '100%', 'float': 'center', 'display': 'inline-block'})
                ],
                style={'width': '50%', 'float': 'right', 'vertical-align': 'bottom'}
            )],
        style={'width': '98%', 'float': 'center', 'vertical-align': 'bottom'}
        ),

    html.Div(dcc.Markdown(' '),
        style={
            'textAlign': 'center',
            'color': dash_colors['text'],
            'width': '100%',
            'float': 'center',
            'display': 'inline-block'}),

    html.Div(dcc.Graph(id='world_map'),
        style={'width': '50%',
            'display': 'inline-block'}
        ),

    html.Div([dcc.Graph(id='trajectory')],
        style={'width': '50%',
            'float': 'right',
            'display': 'inline-block'}),

    html.Div(html.Div(dcc.Slider(id='date_slider',
                min=list(range(len(data['date'].unique())))[0],
                max=list(range(len(data['date'].unique())))[-1],
                value=list(range(len(data['date'].unique())))[-1],
                marks={(idx): {'label': date.format(u"\u2011", u"\u2011") if
                    (idx-4)%7==0 else '', 'style':{'transform': 'rotate(30deg) translate(0px, 7px)'}} for idx, date in
                    enumerate(sorted(set([item.strftime("%m{}%d{}%Y") for
                    item in data['date']])))},  # for weekly marks,
                # marks={(idx): (date.format(u"\u2011", u"\u2011") if
                #     date[4:6] in ['01', '15'] else '') for idx, date in
                #     enumerate(sorted(set([item.strftime("%m{}%d{}%Y") for
                #     item in data['date']])))},  # for bi-monthly makrs
                step=1,
                vertical=False,
                updatemode='mouseup'),
            style={'width': '94.74%', 'float': 'left'}),  # width = 1 - (100 - x) / x
        style={'width': '95%', 'float': 'right'}),  # width = x
    
   
    
    
    
    
    
    
    html.Div([
    html.P(),
    html.Br(),
    html.Br(),
    html.P(['SIR SIMULATION (S-Susceptible, I-Infected, R-Recovered)'], style = {'fontFamily' : 'arial', 'fontSize': '40px', 'textAlign' : 'center'}),
    html.Div(
            dcc.Dropdown(id = 'sir_list',
        options=[{'label': each , 'value': each} for each in df_worldwide['Country/Region'].values ],
        value="Germany",
        style = {'width' : '400px', 'float': 'center' ,'fontSize' : '21px'}
    ))], style = {'width' : '100%','float': 'center'}),
    
    html.Div([
        html.Div([dcc.Graph(id = 'sir_simulations')],style={'width': '100%',
            'float': 'center',
            'display': 'inline-block',
            'background':'white'}),

    ], style = {'width' : '100%','background':'white'}),
    

    
    
    
    html.Div(dcc.Markdown('''
            &nbsp;  
            &nbsp;  
            Built by [Anuraag Sharma Thotapally](https://www.linkedin.com/in/anuraag-sharma-thotapally-12933489/)  
            Source data: [Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)  
            '''),
            style={
                'textAlign': 'center',
                'color': dash_colors['text'],
                'width': '100%',
                'float': 'center',
                'display': 'inline-block'}
            )
        ])

if __name__ == '__main__':
    app.run_server(debug=False)
