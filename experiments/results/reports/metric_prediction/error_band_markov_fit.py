from cProfile import label
from matplotlib import legend_handler
from matplotlib.pyplot import legend, tick_params, title
from numpy import size
import plotly.graph_objs as go

x = [1, 2, 3]

y_lr = [85, 88, 87]
y_lr_upper = [85+15, 88+10, 87+7]
y_lr_lower = [85-15, 88-10, 87-7]

# y_lr = [65, 78, 81]
# y_lr_upper = [65+17, 78+11, 81+9]
# y_lr_lower = [65-17, 78-11, 81-9]

y_baseline = [57, 57, 57]
y_baseline_upper = [57+20, 57+20, 57+20]
y_baseline_lower = [57-20, 57-20, 57-20]

y_subset = [93, 93, 93]
y_subset_upper = [93+4, 93+4, 93+4]
y_subset_lower = [93-4, 93-4, 93-4]

text_size = 20

# y_rf = [85, 88, 87]
# y_rf_upper = [100, 98, 94]
# y_rf_lower = [70, 78, 82]

# y_xgb = [82, 86, 87]
# y_xgb_upper = [82+17, 86+10, 87+5]
# y_xgb_lower = [82-17, 86-10, 87-5]


fig = go.Figure([
    go.Scatter(
        x=x,
        y=y_subset,
        line=dict(color='rgb(0, 71, 171)'),
        mode='lines',
        name='Subset',
    ),
    go.Scatter(
        x=x,
        y=y_lr,
        line=dict(color='rgb(0,153,0)'),
        mode='lines',
        name='Markov',
    ),
    go.Scatter(
        x=x,
        y=y_baseline,
        line=dict(color='rgb(218,180,77)'),
        mode='lines',
        name='Footprint',
    ),
    # go.Scatter(
    #     x=x,
    #     y=y_ml,
    #     line=dict(color='rgb(0,191,255)'),
    #     mode='lines',
    #     name='Multiple Features',
    # ),
    go.Scatter(
        x=x+x[::-1], # x, then x reversed
        y=y_subset_upper + y_subset_lower[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0, 71, 171,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
    go.Scatter(
        x=x+x[::-1], # x, then x reversed
        y=y_lr_upper + y_lr_lower[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,153,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
    go.Scatter(
        x=x+x[::-1], # x, then x reversed
        y=y_baseline_upper + y_baseline_lower[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(218,180,77,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
    # go.Scatter(
    #     x=x+x[::-1], # x, then x reversed
    #     y=y_ml_upper + y_ml_lower[::-1], # upper, then lower reversed
    #     fill='toself',
    #     fillcolor='rgba(218,180,77,0.2)',
    #     line=dict(color='rgba(0,191,255,0)'),
    #     hoverinfo="skip",
    #     showlegend=False
    # ),
])
fig.update_yaxes(range=[0, 100], 
                 title_text='R² Score', 
                 tickfont_size=text_size,
                 title=dict(font=dict(size=text_size)))

fig.update_xaxes(range=[1,3], 
                 title_text='K', 
                 dtick=1, 
                 tickfont_size=text_size, 
                 ticklabelposition="outside right",
                 title=dict(font=dict(size=text_size)),
                 )

fig.update_layout(title_text='Fitness Alignment Prediction',
                  title_x=0.5,
                  legend=dict(font=dict(size=text_size)),
                  title=dict(font=dict(size=text_size)))

fig.show()