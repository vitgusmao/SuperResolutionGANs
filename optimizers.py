from keras.optimizers import Adam
import tensorflow_addons as tfa


def get_adam_optimizer(lr=1e-4,
                       beta_1=0.9,
                       beta_2=0.999,
                       amsgrad=False,
                       epsilon=1e-08,
                       moving_avarage=False):
    optimizer = Adam(learning_rate=lr,
                     beta_1=beta_1,
                     beta_2=beta_2,
                     amsgrad=amsgrad,
                     epsilon=epsilon)

    if moving_avarage:
        return tfa.optimizers.MovingAverage(optimizer)
    return optimizer