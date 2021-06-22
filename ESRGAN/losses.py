from ESRGAN.vgg_net import build_vgg
from tensorflow.keras import losses

vgg = build_vgg((256, 256, 3))


def l1_loss(y_true, y_pred):
    """[summary]

    Args:
        y_true (tf.Tensor):  Ground-truth tensor with shape (batch_size, height, width, channels).
        y_pred (tf.Tensor): Input tensor with shape (batch_size, height, width, channels).

    Returns:
        tf.Tensor: [description]
    """
    loss = losses.MeanAbsoluteError(
        reduction=losses.Reduction.SUM_OVER_BATCH_SIZE, name='l1_loss')
    return loss(y_true, y_pred)


def perceptual_loss(y_true, y_pred):
    """ O loss perceptível é baseado no loss l1 feito sobre as features extraidas do modelo
        de identificação de imagem vgg19 pré treinado sobre a base imagenet
    Args:
        y_true (tf.Tensor): Ground-truth tensor with shape (batch_size, height, width, channels).
        y_pred (tf.Tensor): Input tensor with shape (batch_size, height, width, channels).

    Returns:
        tf.Tensor: Forward results.
    """

    # Exatrai as features pelo vgg19
    y_pred_features = vgg(y_pred)
    y_true_features = vgg(y_true)

    return l1_loss(y_pred_features, y_true_features)


def _gram_mat(x):
    """Calculate Gram matrix.

    Args:
        x (torch.Tensor): Tensor with shape of (n, c, h, w).

    Returns:
        torch.Tensor: Gram matrix.
    """
    n, c, h, w = x.size()
    features = x.view(n, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)

    return gram


def style_loss(y_true, y_pred):
    """
    Args:
        y_true (Tensor): Ground-truth tensor with shape (n, c, h, w).
        y_pred (Tensor): Input tensor with shape (n, c, h, w).

    Returns:
        Tensor: Forward results.
    """
    style_weight = 1.0
    criterion = 'l1'

    # extract vgg features
    x_features = vgg(x)
    gt_features = vgg(gt.detach())

    # calculate style loss
    if style_weight > 0:
        style_loss = 0
        for k in x_features.keys():
            style_loss += criterion(_gram_mat(
                x_features[k]), _gram_mat(gt_features[k])) * layer_weights[k]

        style_loss *= style_weight

    else:
        style_loss = None

    return style_loss


def gan_loss(y_true, y_pred):
    """[summary]

    Args:
        y_true (tf.Tensor):  Ground-truth tensor with shape (batch_size, height, width, channels).
        y_pred (tf.Tensor): Input tensor with shape (batch_size, height, width, channels).

    Returns:
        tf.Tensor: [description]
    """
    # from_logits=True
    return losses.binary_crossentropy(y_true, y_pred)
