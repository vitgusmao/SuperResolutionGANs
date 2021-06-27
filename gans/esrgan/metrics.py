import ipdb
import numpy as np
import tensorflow as tf

# Tentativa de salvar o modelo completo da rede
# class PSNR(tf.keras.metrics.Metric):
#     def __init__(self, name='peak_signal_to_noise_ratio', **kwargs):
#         super(PSNR, self).__init__(name=name, **kwargs)
#         self.psnr = self.add_weight(name='psnr', initializer='glorot_normal')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         ipdb.set_trace()
#         value = tf.image.psnr(y_true, y_pred, max_val=255)

#         if sample_weight is not None:
#             sample_weight = tf.cast(sample_weight, self.dtype)
#             value = tf.multiply(value, sample_weight)

#         self.psnr.assign_add(value)

#     def result(self):
#         return self.psnr

#     def reset_states(self):
#         self.psnr.assign(0)

#     def get_config(self):
#         config = super(PSNR, self).get_config()
#         config.update({"psnr": float(self.psnr)})
#         return config


def psnr_metric(y_true, y_pred):
    tf.image.psnr(y_pred, y_true, max_val=255)

# def _ssim(img1, img2):
#     """Calculate SSIM (structural similarity) for one channel images.

#     It is called by func:`calculate_ssim`.

#     Args:
#         img1 (ndarray): Images with range [0, 255] with order 'HWC'.
#         img2 (ndarray): Images with range [0, 255] with order 'HWC'.

#     Returns:
#         float: ssim result.
#     """

#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()

# def calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
#     """Calculate SSIM (structural similarity).

#     Ref:
#     Image quality assessment: From error visibility to structural similarity

#     The results are the same as that of the official released MATLAB code in
#     https://ece.uwaterloo.ca/~z70wang/research/ssim/.

#     For three-channel images, SSIM is calculated for each channel and then
#     averaged.

#     Args:
#         img1 (ndarray): Images with range [0, 255].
#         img2 (ndarray): Images with range [0, 255].
#         crop_border (int): Cropped pixels in each edge of an image. These
#             pixels are not involved in the SSIM calculation.
#         input_order (str): Whether the input order is 'HWC' or 'CHW'.
#             Default: 'HWC'.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

#     Returns:
#         float: ssim result.
#     """

#     assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
#     img1 = reorder_image(img1, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)

#     if crop_border != 0:
#         img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

#     if test_y_channel:
#         img1 = to_y_channel(img1)
#         img2 = to_y_channel(img2)

#     ssims = []
#     for i in range(img1.shape[2]):
#         ssims.append(_ssim(img1[..., i], img2[..., i]))
#     return np.array(ssims).mean()
