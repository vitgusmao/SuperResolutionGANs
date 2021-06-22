import numpy as np
import plotly.graph_objects as go


def plot_togheter(informations, using=None):

    data = []
    using = using if using else informations.keys()

    for information in using:
        info = informations[information]
        info_range = [i for i in range(len(info))]

        trace = go.Scatter(x=info_range,
                           y=info,
                           marker=dict(color=np.random.random_integers(
                               0, high=100, size=1), ),
                           name=information)
        data.append(trace)

    layout = go.Layout(xaxis={'title': 'Épocas'},
                       yaxis={'title': 'Valor'},
                       margin={
                           'l': 40,
                           'b': 40,
                           't': 50,
                           'r': 50
                       },
                       hovermode='closest')

    figure = go.Figure(data=data, layout=layout)
    figure.show()

    return data
