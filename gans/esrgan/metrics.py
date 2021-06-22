import numpy as np
import tensorflow as tf

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def psnr_metric(y_true, y_pred):
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)

    # if test_y_channel:
    #     img1 = to_y_channel(img1)
    #     img2 = to_y_channel(img2)

    # mse = np.mean((img1 - img2)**2)
    # if mse == 0:
    #     return float('inf')
    # return 20. * np.log10(255. / np.sqrt(mse))
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
