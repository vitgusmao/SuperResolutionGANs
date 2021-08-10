import numpy as np
import plotly.graph_objects as go
import argparse

from utils import load_history, load_yaml

parser = argparse.ArgumentParser(description="Options for super resolution")
parser.add_argument(
    "--config", help="name of yaml config file under configs/", default=None
)

args = parser.parse_args()


def plot_graphics(config):

    history = load_history(config, True)

    for name, values in history.items():
        iterations = [i for i in range(len(values))]
        if len(values) < 0:
            raise Exception(">> missing values.")

        trace = go.Scatter(
            x=iterations,
            y=values,
            marker={"color": 'rgba(255, 0, 0, 1)'},
            name=name,
        )

        layout = go.Layout(
            xaxis={"title": "Iteração"},
            yaxis={"title": name.capitalize()},
            margin={"l": 100, "b": 100, "t": 50, "r": 50},
            hovermode="closest",
            width=960,
            height=675,
            font={"size": 30},
        )

        figure = go.Figure(data=trace, layout=layout)
        # figure.show()
        figure.write_image(f"{config['name'].lower()}_{name.lower()}.svg")


if args.config:
    config = load_yaml(f"./configs/{args.config}.yaml")
    print(f">> {config['name']} config file loaded")
    plot_graphics(config)
    print(f">> {config['name']} graphics plotted.")
else:
    raise Exception(">> missing config file.")
