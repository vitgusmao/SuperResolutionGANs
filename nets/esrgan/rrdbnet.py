from keras.engine.input_layer import Input
from keras.layers import LeakyReLU, Add
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model


def ResidualDenseBlock(block_input, num_filters=64, num_grow_ch=32, num=0):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """
    # in_channels, out_channels, kernel_size, stride=1, padding=0,
    lrelu = LeakyReLU(alpha=0.2)

    conv1 = Conv2D(filters=num_filters,
                   kernel_size=3,
                   kernel_initializer='glorot_normal',
                   strides=1,
                   activation=lrelu,
                   name='rd_block_{}_conv2d_1'.format(num))(block_input)
    conv2 = Conv2D(filters=num_filters + num_grow_ch,
                   kernel_size=3,
                   kernel_initializer='glorot_normal',
                   strides=1,
                   activation=lrelu,
                   name='rd_block_{}_conv2d_2'.format(num))(conv1)
    conv3 = Conv2D(filters=num_filters + 2 * num_grow_ch,
                   kernel_size=3,
                   kernel_initializer='glorot_normal',
                   strides=1,
                   activation=lrelu,
                   name='rd_block_{}_conv2d_3'.format(num))(conv2)
    conv4 = Conv2D(filters=num_filters + 3 * num_grow_ch,
                   kernel_size=3,
                   kernel_initializer='glorot_normal',
                   strides=1,
                   activation=lrelu,
                   name='rd_block_{}_conv2d_4'.format(num))(conv3)
    conv5 = Conv2D(filters=num_filters + 4 * num_grow_ch,
                   kernel_size=3,
                   kernel_initializer='glorot_normal',
                   strides=1,
                   name='rd_block_{}_conv2d_5'.format(num))(conv4)

    return conv5

    # initialization
    default_init_weights(
        [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)


def RRDB(block_input, num_filters, num_grow_ch=32):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_filters (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """
    rdb1 = ResidualDenseBlock(block_input, num_filters, num_grow_ch)
    rdb2 = ResidualDenseBlock(rdb1, num_filters, num_grow_ch)
    rdb3 = ResidualDenseBlock(rdb2, num_filters, num_grow_ch)

    # out = self.rdb1(x)
    # out = self.rdb2(out)
    # out = self.rdb3(out)
    # # Emperically, we use 0.2 to scale the residual for better performance
    # return out * 0.2 + x

    return rdb3


def RRDBNet(net_input,
            num_in_ch,
            num_out_ch,
            num_feat=64,
            num_block=23,
            num_grow_ch=32):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """
    lrelu = LeakyReLU(alpha=0.2)

    conv_first = Conv2D(num_in_ch, num_feat, 3, 1, 1)(net_input)

    body = RRDB(conv_first,
                num_block,
                num_feat=num_feat,
                num_grow_ch=num_grow_ch)
    conv_body = Conv2D(num_feat, num_feat, 3, 1, 1)(body)
    block = Add()([body, conv_body])

    # upsample
    conv_up1 = Conv2D(num_feat, num_feat, 3, 1, 1, activation=lrelu)(block)
    conv_up1 = UpSampling2D()(conv_up1)
    conv_up2 = Conv2D(num_feat, num_feat, 3, 1, 1, activation=lrelu)(conv_up1)
    conv_up2 = UpSampling2D()(conv_up2)

    conv_hr = Conv2D(num_feat, num_feat, 3, 1, 1, activation=lrelu)(conv_up2)
    conv_last = Conv2D(num_feat, num_out_ch, 3, 1, 1)(conv_hr)

    model = Model(name='RRDBNet')

    return model


def build_rrdbnet(input_shape, filters=64):
    """
    Net data format (batch_size, height, width, channels)
    """

    lrelu = LeakyReLU(alpha=0.2)

    net_input = Input(input_shape)

    first_conv = Conv2D(filters=filters,
                        kernel_size=3,
                        kernel_initializer='glorot_normal',
                        strides=1,
                        padding='same',
                        name='first_conv')(net_input)

    block = 1
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_1')(first_conv)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_2')(x)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_3')(x)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_4')(x)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_5')(x)

    block += 1
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_1')(x)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_2')(x)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_3')(x)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_4')(x)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_5')(x)

    block += 1
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_1')(x)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_2')(x)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_3')(x)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_4')(x)
    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name=f'rd_block_{block}_conv_5')(x)

    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name='body_conv')(x)
    x = Add()([first_conv, x])

    # upsample
    x = UpSampling2D(size=2, interpolation="nearest", name='up_sampling_1')(x)
    x = Conv2D(
        filters=filters * 2,
        kernel_size=3,
        kernel_initializer='glorot_normal',
        strides=1,
        padding='same',
        activation=lrelu,
        name='conv_up_sampling_1',
    )(x)
    x = UpSampling2D(size=2, interpolation="nearest", name='up_sampling_2')(x)
    x = Conv2D(
        filters=filters * 4,
        kernel_size=3,
        kernel_initializer='glorot_normal',
        strides=1,
        padding='same',
        activation=lrelu,
        name='conv_up_sampling_2',
    )(x)

    x = Conv2D(filters=filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation=lrelu,
               name='conv_hr')(x)
    output = Conv2D(filters=3,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               activation='tanh',
               name='last_conv')(x)

    model = Model(inputs=net_input, outputs=output, name='RRDBNet')
    print(model.summary())

    return model
