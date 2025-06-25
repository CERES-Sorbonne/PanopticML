import io

from PIL import Image

from models import VectorType


def preprocess_image(image_data: bytes, vector_type: VectorType = VectorType.rgb):
    image = Image.open(io.BytesIO(image_data))
    if vector_type == VectorType.rgb:
        image = image.convert('RGB')
    else:
        image = image.convert('L').convert('RGB')
    return image