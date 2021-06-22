import ipdb
import numpy as np
import plotly.graph_objects as go


def plot_losses(losses):
    discriminator_loss = losses['d_loss']
    dis_loss_fake = losses['d_fake_loss']
    dis_loss_real = losses['d_real_loss']
    dis_loss_range = [i for i in range(len(discriminator_loss))]

    gan_loss = losses['g_loss']
    gan_loss_range = [i for i in range(len(gan_loss))]

    data = []

    trace = go.Scatter(x=dis_loss_range,
                       y=discriminator_loss,
                       marker=dict(color='blue'),
                       name="Loss Discriminador")
    data.append(trace)

    trace = go.Scatter(x=dis_loss_range,
                       y=dis_loss_fake,
                       marker=dict(color='green'),
                       name="Loss Discriminador Fake")
    data.append(trace)

    trace = go.Scatter(x=dis_loss_range,
                       y=dis_loss_real,
                       marker=dict(color='purple'),
                       name="Loss Discriminador Real")
    data.append(trace)

    trace = go.Scatter(x=gan_loss_range,
                       y=gan_loss,
                       marker=dict(color='red'),
                       name="Loss Generator")
    data.append(trace)

    layout = go.Layout(xaxis={'title': 'Tempo'},
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
