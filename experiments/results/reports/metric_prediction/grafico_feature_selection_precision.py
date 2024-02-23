from cProfile import label
from matplotlib import legend_handler
from matplotlib.pyplot import legend, tick_params, title
from numpy import size
import plotly.graph_objs as go

x = [1,2,3]


y_TOP7_RF =[49,83,82]
y_TOP5_RF =[53,82,81]
y_TOP3_RF = [30,67,70]
y_TRSH15_RF = [46,67,70]
y_TOP7_XGB = [60,86,82]
y_TOP5_XGB = [64,85,80]
y_TOP3_XGB = [41,84,79]
y_TRSH15_XGB = [60,84,79]
y_Top10RF = [71,85,78]


fig = go.Figure([
    go.Scatter(
        x=x,
        y=y_TOP7_RF,
        line=dict(color='rgb(128,0,128)'),
        mode='lines',
        name='TOP7_RF',
    ),
    go.Scatter(
        x=x,
        y=y_TOP5_RF,
        line=dict(color='rgb(0,0,255)'),
        mode='lines',
        name='TOP5_RF',
    ),

    go.Scatter(
        x=x,
        y=y_TOP3_RF,
        line=dict(color='rgb(255,0,0)'),
        mode='lines',
        name='TOP3_RF',
    ),
    go.Scatter(
        x=x,
        y=y_TRSH15_RF,
        line=dict(color='rgb(255,165,0)'),
        mode='lines',
        name='TRSH15_RF',
    ),
    go.Scatter(
        x=x,
        y=y_TOP7_XGB,
        line=dict(color='rgb(0,255,0)'),
        mode='lines',
        name='TOP7_XGB',
    ),
    go.Scatter(
        x=x,
        y=y_TOP5_XGB,
        line=dict(color='rgb(255,255,0)'),
        mode='lines',
        name='TOP5_XGB',
    ),
    go.Scatter(
        x=x,
        y=y_TOP3_XGB,
        line=dict(color='rgb(0,255,255)'),
        mode='lines',
        name='TOP3_XGB',
    ),
    go.Scatter(
        x=x,
        y=y_TRSH15_XGB,
        line=dict(color='rgb(0,0,0)'),
        mode='lines',
        name='TRSH15_XGB',
    ),
    go.Scatter(
        x=x,
        y=y_Top10RF,
        line=dict(color='rgb(160,82,45)'),
        mode='lines',
        name='Top10RF',
    ),


])

fig.update_yaxes(range=None, 
                 title_text='R² Score', 
                 tickfont_size=15,
                 title=dict(font=dict(size=15)))

fig.update_xaxes(range=[1,3], 
                 title_text='K', 
                 dtick=1, 
                 tickfont_size=15, 
                 ticklabelposition="outside right",
                 title=dict(font=dict(size=15)),
                 )



fig.show()