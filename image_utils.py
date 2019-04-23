from PIL import Image, ImageFont, ImageDraw
import numpy as np
from io import BytesIO
from google.cloud.vision import types
import os

def load_image(
        shape=(224, 224), bounds=(0, 1), dtype=np.float32,
        data_format='channels_last', fname='example.png', abs_path=False, fpath=None):
    """ Returns a resized image of target fname.

    Parameters
    ----------
    shape : list of integers
        The shape of the returned image.
    data_format : str
        "channels_first" or "channls_last".

    Returns
    -------
    image : array_like
        The example image in bounds (0, 255) or (0, 1)
        depending on bounds parameter
    """
    if abs_path == True:
        assert fpath is not None, "fpath has not to be None when abs_path is True."
    assert len(shape) == 2
    assert data_format in ['channels_first', 'channels_last']
    if not abs_path:
        path = os.path.join(os.path.dirname(__file__), 'images/%s' % fname)
    else:
        path = fpath
    image = Image.open(path)
    image = image.resize(shape)
    image = np.asarray(image, dtype=dtype)
    image = image[:, :, :3]
    assert image.shape == shape + (3,)
    if data_format == 'channels_first':
        image = np.transpose(image, (2, 0, 1))
    if bounds != (0, 255):
        image /= 255.
    return image

def save_image(image):
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype is not np.uint8:
        image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save("./out/test.jpg")    

def numpy_to_bytes(image, format='JPEG'):
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype is not np.uint8:
        image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    byte_io = BytesIO()
    image.save(byte_io, format=format)
    image = byte_io.getvalue()
    image = types.Image(content=image)
    return image

def to_coordinates(image, coords):
    top = coords[0] * image.size[1]
    left = coords[1] * image.size[0]
    bottom = coords[2] * image.size[1]
    right = coords[3] * image.size[0]
    return [top, left, bottom, right]

def draw_boxes(image, labels, boxes):
    thickness = (image.size[0] + image.size[1]) // 300
    '''
    draw = ImageDraw.Draw(image)
    draw.rectangle([200, 300, 400, 500], outline=(255, 0, 255))
    del draw

    # write to stdout
    image.show()
    '''
    boxes = [boxes[1]]
    labels = [labels[1]]
    for i, box in enumerate(boxes):
        draw = ImageDraw.Draw(image)
        label = labels[i]
        label_size = draw.textsize(label)
        top, left, bottom, right = to_coordinates(image, box)
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=(255, 0, 255))
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=(255, 0, 255))
        draw.text(text_origin.tolist(), label, fill=(0, 0, 0))
        del draw
    
    image.show()


