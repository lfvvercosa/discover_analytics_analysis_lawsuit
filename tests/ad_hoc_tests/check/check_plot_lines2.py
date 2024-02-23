import plotly.graph_objs as go

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 7, 4, 5, 6, 7, 8, 9, 10]
y_upper = [2, 3, 8, 5, 6, 7, 8, 9, 10, 11]
y_lower = [0, 1, 5, 3, 4, 5, 6, 7, 8, 9]


fig = go.Figure([
    go.Scatter(
        x=x,
        y=y,
        line=dict(color='rgb(0,100,80)'),
        mode='lines'
    ),
    go.Scatter(
        x=x+x[::-1], # x, then x reversed
        y=y_upper+y_lower[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    )
])
fig.show()