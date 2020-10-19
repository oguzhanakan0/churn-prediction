cell_values = list(stats_df.iloc[0])

cell_values[0] = cell_values[0]
cell_values[1] = '{:.2%}'.format(cell_values[1])
cell_values[2] = '{:.2%}'.format(cell_values[2])
cell_values[2] = '{:.6f}'.format(cell_values[3])

bos_liste = []
for i in cell_values:
    bos_liste.append(['<b>'+str(i)+'</b>'])
cell_values = bos_liste

bos_liste1 = []
for i in list(stats_df.columns):
    bos_liste1.append(['<b>'+str(i)+'</b>'])

trace = go.Table(
    header=dict(values=bos_liste1,
                line=dict(color='#7D7F80'),
                fill=dict(color='rgb(240,240,240)'),
                font=dict(color='rgb(0,0,0)',size=16),
                align=['left']*5),
    cells=dict(values=cell_values,
               line=dict(color='#7D7F80'),
               fill=dict(color='rgb(255,255,255)'),
               font=dict(color='rgb(0,0,0)',size=16),
               height=30,
               align=['left']*5))

layout = dict(width = 1100, height = 300)
data = [trace]
fig_cat = dict(data=data,layout=layout)