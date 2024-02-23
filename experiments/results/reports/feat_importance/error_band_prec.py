from cProfile import label
from matplotlib.pyplot import title
import plotly.graph_objs as go

x = [1, 2, 3]

y_bf = [20, 70, 70]
y_bf_upper = [20+62, 70+16, 70+16]
y_bf_lower = [20-62, 70-16, 70-16]

y_top10 = [54, 74, 75]
y_top10_upper = [54+29, 74+17, 75+22]
y_top10_lower = [54-29, 74-17, 75-22]

y_rf = [71, 85, 79]
y_rf_upper = [71+24, 85+11, 79+29]
y_rf_lower = [71-24, 85-11, 79-29]

y_xgb = [60, 83, 81]
y_xgb_upper = [60+35, 83+13, 81+17]
y_xgb_lower = [60-35, 83-13, 81-17]


fig = go.Figure([
    # go.Scatter(
    #     x=x,
    #     y=y_bf,
    #     line=dict(color='rgb(222,12,75)'),
    #     mode='lines',
    #     name='Best feature',
    # ),
    # go.Scatter(
    #     x=x,
    #     y=y_top10,
    #     line=dict(color='rgb(218,180,77)'),
    #     mode='lines',
    #     name='Top 10 features - Pearson',
    # ),
    go.Scatter(
        x=x,
        y=y_rf,
        line=dict(color='rgb(0,100,80)'),
        mode='lines',
        name='Top 10 features - RF',
    ),
    # go.Scatter(
    #     x=x,
    #     y=y_xgb,
    #     line=dict(color='rgb(48,98,165)'),
    #     mode='lines',
    #     name='Top 10 features - XGB',
    # ),
    # go.Scatter(
    #     x=x+x[::-1], # x, then x reversed
    #     y=y_bf_upper+y_bf_lower[::-1], # upper, then lower reversed
    #     fill='toself',
    #     fillcolor='rgba(222,12,75,0.2)',
    #     line=dict(color='rgba(255,255,255,0)'),
    #     hoverinfo="skip",
    #     showlegend=False
    # ),
    # go.Scatter(
    #     x=x+x[::-1], # x, then x reversed
    #     y=y_top10_upper+y_top10_lower[::-1], # upper, then lower reversed
    #     fill='toself',
    #     fillcolor='rgba(218,180,77,0.2)',
    #     line=dict(color='rgba(255,255,255,0)'),
    #     hoverinfo="skip",
    #     showlegend=False
    # ),
    go.Scatter(
        x=x+x[::-1], # x, then x reversed
        y=y_rf_upper+y_rf_lower[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
    # go.Scatter(
    #     x=x+x[::-1], # x, then x reversed
    #     y=y_xgb_upper+y_xgb_lower[::-1], # upper, then lower reversed
    #     fill='toself',
    #     fillcolor='rgba(48,98,165,0.2)',
    #     line=dict(color='rgba(255,255,255,0)'),
    #     hoverinfo="skip",
    #     showlegend=False
    # ),
])
fig.update_yaxes(range=[20, 100], title_text='R² Score', tickfont_size=16)
fig.update_xaxes(title_text='K', dtick=1, tickfont_size=16)
fig.update_layout(title_text='R² Score for Fitness using Random Forest',
                  title_x=0.5)
fig.show()