from google.cloud import vision
from google.cloud.vision import types
import io

def detect_label_numpy(image):
    client = vision.ImageAnnotatorClient()

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations
    if len(labels) > 0:
        return labels[0]
    return None

def detect_label_file(path):
    client = vision.ImageAnnotatorClient()

    # The name of the image file to annotate
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations
    return labels


def detect_objects_numpy(image):

    client = vision.ImageAnnotatorClient()
    objects = client.object_localization(
        image=image).localized_object_annotations


    if len(objects) > 0:
        return objects[0].name
    
    return None

def detect_objects_file(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations

    '''
    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))
    '''
    
    return objects


def detect_text_numpy(image):
    """Detects text in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    response = client.text_detection(image=image)
    texts = response.text_annotations
    if len(texts) > 0:
        return texts[0].description.strip()
    return None


def detect_text_file(path):
    """Detects text in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

def detect_safe_search_numpy(image):
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    
    return safe.adult

def detect_safe_search_file(path):
    """Detects unsafe features in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    #print('Safe search:')
    #print('adult: {}'.format(likelihood_name[safe.adult]))
    #print('medical: {}'.format(likelihood_name[safe.medical]))
    #print('spoofed: {}'.format(likelihood_name[safe.spoof]))
    #print('violence: {}'.format(likelihood_name[safe.violence]))
    #print('racy: {}'.format(likelihood_name[safe.racy]))
    return (safe.adult, safe.racy)

def detect_faces_numpy(image):
    """Detects faces in an image."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    if len(faces) == 0:
        return False

    return True

def detect_faces_file(path):
    """Detects faces in an image."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations
    print(len(faces))
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('Faces:')

    for face in faces:
        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])

        print('face bounds: {}'.format(','.join(vertices)))