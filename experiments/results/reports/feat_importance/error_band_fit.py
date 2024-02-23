from cProfile import label
from matplotlib.pyplot import title
import plotly.graph_objs as go

x = [1, 2, 3]

y_bf = [70, 83, 85]
y_bf_upper = [91, 94, 93]
y_bf_lower = [51, 72, 77]

y_top10 = [84, 86, 85]
y_top10_upper = [84+19, 86+12, 87+10]
y_top10_lower = [84-19, 86-12, 87-10]

y_rf = [85, 88, 87]
y_rf_upper = [100, 98, 94]
y_rf_lower = [70, 78, 82]

y_xgb = [82, 86, 87]
y_xgb_upper = [82+17, 86+10, 87+5]
y_xgb_lower = [82-17, 86-10, 87-5]


fig = go.Figure([
    go.Scatter(
        x=x,
        y=y_bf,
        line=dict(color='rgb(222,12,75)'),
        mode='lines',
        name='Best feature',
    ),
    go.Scatter(
        x=x,
        y=y_top10,
        line=dict(color='rgb(218,180,77)'),
        mode='lines',
        name='Top 10 features - Pearson',
    ),
    go.Scatter(
        x=x,
        y=y_rf,
        line=dict(color='rgb(0,100,80)'),
        mode='lines',
        name='Top 10 features - RF',
    ),
    go.Scatter(
        x=x,
        y=y_xgb,
        line=dict(color='rgb(48,98,165)'),
        mode='lines',
        name='Top 10 features - XGB',
    ),
    # go.Scatter(
    #     x=x+x[::-1], # x, then x reversed
    #     y=y_rf_upper+y_rf_lower[::-1], # upper, then lower reversed
    #     fill='toself',
    #     fillcolor='rgba(0,100,80,0.2)',
    #     line=dict(color='rgba(255,255,255,0)'),
    #     hoverinfo="skip",
    #     showlegend=False
    # ),
    # go.Scatter(
    #     x=x2+x2[::-1], # x, then x reversed
    #     y=y2_upper+y2_lower[::-1], # upper, then lower reversed
    #     fill='toself',
    #     fillcolor='rgba(222,12,75,0.2)',
    #     line=dict(color='rgba(255,255,255,0)'),
    #     hoverinfo="skip",
    #     showlegend=False
    # ),
#     go.Scatter(
#         x=x+x[::-1], # x, then x reversed
#         y=y_xgb_upper+y_xgb_lower[::-1], # upper, then lower reversed
#         fill='toself',
#         fillcolor='rgba(48,98,165,0.2)',
#         line=dict(color='rgba(255,255,255,0)'),
#         hoverinfo="skip",
#         showlegend=False
#     ),
])
fig.update_yaxes(range=[40, 100], title_text='R² Score', tickfont_size=16)
fig.update_xaxes(title_text='K', dtick=1, tickfont_size=16)
fig.update_layout(title_text='R² Score for Fitness using XGBoost',
                  title_x=0.5)
fig.show()