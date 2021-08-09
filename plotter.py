import numpy as np
import plotly.graph_objects as go
import pandas as pd

from utils import load_history, load_yaml

def plot_graphics(config):

    history = load_history(config)
    data = []
    

    for name, values in history.items():
        iterations = [i for i in range(len(values))]

        # .capitalize()

        trace = go.Scatter(x=iterations,
                           y=values,
                           marker=dict(color=np.random.random_integers(
                               0, high=100, size=1), ),
                           name=name)
        data.append(trace)

        layout = go.Layout(xaxis={'title': 'Iteração'},
                        yaxis={'title': name.upper()},
                        margin={
                            'l': 40,
                            'b': 40,
                            't': 50,
                            'r': 50
                        },
                        hovermode='closest')

        figure = go.Figure(data=data, layout=layout)
        figure.show()
        data = []


config = load_yaml("./configs/srcnn.yaml")
plot_graphics(config)
