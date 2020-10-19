def yuzdelik(x):
    yuzde_list=[]
    for i in range(len(x)):
        if type(x.iloc[i])==np.float64 and x.iloc[i]<=1 and x.iloc[i]>=0:
            a="{:.2%}".format(x.iloc[i])
            yuzde_list.append(a)
    return pd.Series(yuzde_list)

def format_text(x,y='population_size'):
    formatted=[]
    for i in range(len(x)):
        formatted.append(y+': {}'.format(x.iloc[i]))
    return pd.Series(formatted)

screen_width = 1000
screen_height = 500
bubble_color = 'rgb(255,98,0)'

size = graph_stats.pc_of_rank
trace0 = go.Scatter(
    x = list(range(0,len(graph_stats))),
    y = graph_stats.target_ratio,
    mode = 'markers+text',
    text = yuzdelik(graph_stats.target_ratio),
    textposition='top center',
    textfont=dict(size=11),
    showlegend=False,
    hoverlabel = dict(bgcolor=bubble_color, font=dict(color='white',size=14)),
    marker=dict(symbol='circle',opacity=1,size=size,sizemode='area',sizeref=2.*max(size)/(40.**2),sizemin=8,color=bubble_color),
    hoverinfo='none'
)

dummy_trace = go.Scatter(
    x=list(range(0,len(graph_stats))),
    y=graph_stats.target_ratio,
    text=format_text(yuzdelik(size)),
    opacity=0,
    showlegend=False,
    hoverlabel=dict(bgcolor=bubble_color,font=dict(color='white',size=14)),
    hoverinfo='text'
)

dummy_trace_2 = go.Scatter(
    name='population size',
    x=list(range(0,len(graph_stats))),
    y=graph_stats.target_ratio,
    dx=5,
    showlegend=True,
    mode='markers',
    marker=dict(symbol='circle',size=5,color=bubble_color,opacity=1),
    hoverinfo='none'
)

data = [dummy_trace_2,trace0,dummy_trace]
layout = go.Layout(
    title=feat,
    autosize=False,
    width=screen_width,
    height=screen_height,
    showlegend=True,
    legend=dict(orientation='h',x=0,y=1.1),
    yaxis=dict(
        title='% Target of All',
        tickformat= ',.2%', showgrid=True,hoverformat='.2f',zeroline=False,
        mirror=True,
        ticks='outside',
        showline=True),
    xaxis=dict(
        title='Categories',
        tickmode='array',tickvals=list(range(0,len(graph_stats))),ticktext=graph_stats.index,
        showgrid=False,
        zeroline=False,
        mirror=True,
        ticks='outside',
        showline=True))

fig = go.Figure(data=data,layout=layout)