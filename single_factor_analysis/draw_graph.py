#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def yuzdelik(x):
    yuzde_list=[]
    for i in range(len(x)):
        if type(x.iloc[i])==np.float64 and x.iloc[i]<=1 and x.iloc[i]>=0:
            a="{:.2%}".format(x.iloc[i])
            yuzde_list.append(a)
    return pd.Series(yuzde_list)

def format_text(x,y):
    formatted=[]
    for i in range(len(x)):
        formatted.append(y+': {}'.format(x.iloc[i]))
    return pd.Series(formatted)

def x_coordinates(x, bar_widths):
    x_coordinates=[]
    for i in range(x.max_rank_value.count()):
        if(x.mean_rank_value.iloc[i] - x.min_rank_value.iloc[i] > x.max_rank_value.iloc[i] - x.mean_rank_value.iloc[i]):
            x_coordinates.append(x.mean_rank_value[i]+bar_widths[i]/2)
        else:
            x_coordinates.append(x.mean_rank_Value[i]-bar_widths[i]/2)
    return x_coordinates

# Class variables
bar_width = 0.02
screen_width = 1000
screen_height = 500
dot_size = 7
regression_line_color = 'rgb(0,0,0)'
bar_color = 'rgb(255,98,0)'
range_bar_color = 'rgb(255,98,0)'
pd_dots_color = 'rgb(2,0,0)'

# Range calculation
bar_widths = pd.Series(np.repeat((max(graph_stats.max_rank_value)- min(graph_stats.min_rank_value))/(graph_stats.max_rank_value.count()*3),graph_stats.max_rank_value.count()))
range_widths = pd.Series(graph_stats.max_rank_value - graph_stats.min_rank_value)
range_widths[range_widths==0] = bar_widths[0]

# Data bars
trace1 = go.Bar(
    name = 'population size',
    x = graph_stats.mean_rank_value,
    y = graph_stats.pc_of_rank,
    marker = dict(color=bar_color),
    width = bar_widths,
    opacity = 1,
    text = format_text(yuzdelik(graph_stats.pc_of_rank),'population size'),
    hoverinfo = None)

trace2 = go.Scatter(
    name = 'orders',
    x=graph_stats.mean_rank_value,
    y=graph_stats.fpd_ratio,
    yaxis='y2',
    marker=dict(color=pd_dots_color, size=dot_size),
    mode='markers+lines',
    line=dict(shape='spline', dash='solid',color='#000000',width=1)
    )

trace4 = go.Bar(
    x=pd.Series((graph_stats.max_rank_value + graph_stats.min_rank_value)/2),
    y=pd.Series(np.repeat(max(graph_stats.pc_of_rank)/10,graph_stats.max_rank_value.count())),
    name='data range',
    width=range_widths,
    marker=dict(color=range_bar_color, line=dict(color='rgb(255,255,255)',width=2)),
    opacity=0.3,
    y0=-2
)

from numpy import array,ones
from scipy import stats

# generate linear fit
x = graph_stats.mean_rank_value
y = graph_stats.fpd_ratio
slope,intercept,r_value,p_value,std_err = stats.linregress(x,y)
line = slope*x + intercept

trace3 = go.Scatter(
    x=x,
    y=line,
    mode='lines',
    marker=go.scatter.Marker(color=regression_line_color),
    name='fit',
    yaxis='y3'
)

data= [trace4,trace1,trace2,trace3]
layout= go.Layout(autosize=False,
                 width=screen_width,
                 height=screen_height,
                 barmode='overlay',
                 bargap=0.15,
                 title=feat,
                 legend=dict(orientation='h',x=0,y=1.1),
                 yaxis=dict(
                     title='% of sample size',
                     tickformat=',.0%',
                     mirror=True,
                     ticks='outside',
                     showline=True,
                     zeroline=False
                 ),yaxis2=dict(
                     title='# mean orders',
                     overlaying='y',
                     side='right',
                     showgrid=False,
                     zeroline=False
                 ),yaxis3=dict(
                     tickformat=',.1%',
                     overlaying='y',
                     side='right',
                     showticklabels=False,
                     showgrid=False,
                     zeroline=False
                 ),xaxis=dict(
                     title='Mean Rank Value',
                     tickvals=graph_stats.mean_rank_value,
                     tickformat='.2f',
                     tickangle=-90,
                     mirror=True,
                     ticks='outside',
                     showline=True
                 )
                 )
            
fig = go.Figure(data=data,layout=layout)