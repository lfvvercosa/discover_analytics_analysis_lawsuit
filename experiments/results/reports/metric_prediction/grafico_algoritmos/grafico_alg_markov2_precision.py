from cProfile import label
from matplotlib import legend_handler
from matplotlib.pyplot import legend, tick_params, title
from numpy import size
import plotly.graph_objs as go

x = ['TOP7_RF', 'TOP5_RF', 'TOP3_RF','TRSH15_RF','TOP7_XGB', 'TOP5_XGB','TOP3_XGB', 'TRSH15_XGB','Top 10 RF']

y_svr = [70, 70, 63, 63, 59, 65, 75, 75, 61]
y_lr = [59, 57, 59, 59, 52, 55, 57, 57, 55]
y_MLP = [71, 68, 47, 58, 63, 65, 67, 66, 59]
y_RFR = [83, 82, 67, 67, 86, 85, 84, 84, 85]
y_XGB = [78, 74, 59, 59, 76, 82, 81, 81, 80]


'''
y_svr_upper = [85+19, 77+9, 78+15, 77+15, 82+13, 76+10, 67+15, 76+10]
y_svr_down = [85-19, 77-9, 78-15, 77-15, 82-13, 76-10, 67-15, 76-10]

y_lr_upper = [98, 92, 93, 94, 95, 89, 82, 89]
y_lr_down = [62, 50, 51, 48, 47, 53, 40, 53]


y_MLP_upper = [94, 131, 85, 91, 95, 91, 81, 91]
y_MLP_down = [70, -49, 53, 55, 39, 63, 43, 49]

y_RFR_upper = [96, 98, 92, 89, 95, 91, 88, 91]
y_RFR_down = [60, 58, 48, 57, 61, 59, 46, 59]

y_XGB_upper = [99, 91, 93, 90, 93, 90, 83, 90]
y_XGB_down = [69, 57, 55, 56, 57, 56, 41, 56]
'''

fig = go.Figure([
    go.Scatter(
        x=x,
        y=y_svr,
        line=dict(color='rgb(0,0,0)'),
        mode='lines',
        name='SVR',
    ),
    go.Scatter(
        x=x,
        y=y_lr,
        line=dict(color='rgb(0,0,139)'),
        mode='lines',
        name='LR',
    ),

    #(0,191,255)
    go.Scatter(
        x=x,
        y=y_MLP,
        line=dict(color='rgb(0,100,0)'),
        mode='lines',
        name='MLP',
    ),go.Scatter(
        x=x,
        y=y_RFR,
        line=dict(color='rgb(255,255,0)'),
        mode='lines',
        name='RFR',
    ),
    go.Scatter(
        x=x,
        y=y_XGB,
        line=dict(color='rgb(255,0,0)'),
        mode='lines',
        name='XGB',
    ),


])

fig.update_yaxes(range=None, 
                 title_text='R² Score', 
                 tickfont_size=15,
                 title=dict(font=dict(size=15)))

fig.update_xaxes(range=[0,8], 
                 title_text='Sel_Feat', 
                 dtick=1, 
                 tickfont_size=15, 
                 ticklabelposition="outside right",
                 title=dict(font=dict(size=15)),
                 )



fig.show()