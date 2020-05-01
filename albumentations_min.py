import cv2
from functools import wraps
import numpy as np

def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1

def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """

    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                chunk = img[:, :, index : index + 4]
                chunk = process_fn(chunk, **kwargs)
                chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn

def preserve_channel_dim(func):
    """
    Preserve dummy channel dim.
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and len(result.shape) == 2:
            result = np.expand_dims(result, axis=-1)
        return result

    return wrapped_function

@preserve_channel_dim
def resize(img, height, width, interpolation=cv2.INTER_LINEAR):
    resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(width, height), interpolation=interpolation)
    return resize_fn(img)

def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))

def _func_max_size(img, max_size, interpolation, func):
    height, width = img.shape[:2]

    scale = max_size / float(func(width, height))

    if scale != 1.0:
        new_height, new_width = tuple(py3round(dim * scale) for dim in (height, width))
        img = resize(img, height=new_height, width=new_width, interpolation=interpolation)
    return img

@preserve_channel_dim
def LongestMaxSide(img, max_size, interpolation=cv2.INTER_LINEAR):
    return _func_max_size(img, max_size, interpolation, max)

@preserve_channel_dim
def pad_with_params(
    img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode=cv2.BORDER_REFLECT_101, value=None
):
    img = cv2.copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value=value)
    return img

def PadIfNeeded(img, min_height=224, min_width=224, border_mode=0):
    rows = img.shape[0]
    cols = img.shape[1]

    if rows < min_height:
        h_pad_top = int((min_height - rows) / 2.0)
        h_pad_bottom = min_height - rows - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if cols < min_width:
        w_pad_left = int((min_width - cols) / 2.0)
        w_pad_right = min_width - cols - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    return pad_with_params(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode)

if __name__ == '__main__':
    data = cv2.imread('uploads/images.jpeg')
    img_color = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    ret = LongestMaxSide(img_color, 224)
    pad = PadIfNeeded(ret)
    import matplotlib.pyplot as plt
    fig = plt.imshow(pad)
    plt.show()