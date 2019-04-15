from PIL import Image, ImageFont, ImageDraw
import numpy as np
image = Image.open("images/0000.png")
#font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
#                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

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

#coords = [0.42, 0.26, 0.94, 0.78]


#draw_boxes(image, [coords])


