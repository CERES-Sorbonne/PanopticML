from PIL import Image
import base64
from io import BytesIO
from mistralai import Mistral
import os
import json
import time

model = "pixtral-12b-2409"
api_key = os.getenv("MISTRAL_KEY")
client = Mistral(api_key=api_key)


def create_batch(liste, batch_size):
    return [liste[i:i + batch_size] for i in range(0, len(liste), batch_size)]



import json

def create_labels_from_group(groups_images):
    all_labels = []
    for base64_image in groups_images:
        label = None
        while label is None:
            response = create_labels_from_image(base64_image)
            try:
                if "```json" in response:
                    response = response.split('[')[1].split(']')[0]
                    response = f"[{response}]"
                result = json.loads(response)
                if isinstance(result, list) and all(isinstance(x, str) for x in result):
                    label = result
                else:
                    print("Invalid format, retrying…")
            except json.JSONDecodeError:
                print("Invalid JSON, retrying…")
            time.sleep(1)
        all_labels.append(label)
    return all_labels


def create_labels_from_image(base64_image):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                },
                {
                    "type": "text",
                    "text": """\
                    Vous recevez jusqu'à 8 mosaïques d'images numérotées de 1 à 8. Pour chaque mosaïque, analysez le contenu et répondez UNIQUEMENT avec un JSON valide dans l'ordre exact des mosaïques (1, 2, 3, 4, 5, 6, 7, 8).

                    IMPORTANT : 
                    - Répondez EXCLUSIVEMENT en JSON valide, sans texte supplémentaire
                    - L'ordre de vos réponses doit OBLIGATOIREMENT correspondre à l'ordre des mosaïques (1→2→3→4→5→6→7→8)
                    - Utilisez exactement cette structure pour chaque mosaïque
                    
                    Format de réponse requis :
                    [
                      {
                        "mosaic_id": 1,
                        "keywords": ["mot1", "mot2", "mot3", "mot4", "mot5"],
                        "description": "Description de la mosaïque 1 en maximum 300 mots..."
                      },
                      {
                        "mosaic_id": 2,
                        "keywords": ["mot1", "mot2", "mot3", "mot4", "mot5"],
                        "description": "Description de la mosaïque 2 en maximum 300 mots..."
                      },
                      {
                        "mosaic_id": 3,
                        "keywords": ["mot1", "mot2", "mot3", "mot4", "mot5"],
                        "description": "Description de la mosaïque 3 en maximum 300 mots..."
                      },
                      {
                        "mosaic_id": 4,
                        "keywords": ["mot1", "mot2", "mot3", "mot4", "mot5"],
                        "description": "Description de la mosaïque 4 en maximum 300 mots..."
                      },
                      {
                        "mosaic_id": 5,
                        "keywords": ["mot1", "mot2", "mot3", "mot4", "mot5"],
                        "description": "Description de la mosaïque 5 en maximum 300 mots..."
                      },
                      {
                        "mosaic_id": 6,
                        "keywords": ["mot1", "mot2", "mot3", "mot4", "mot5"],
                        "description": "Description de la mosaïque 6 en maximum 300 mots..."
                      },
                      {
                        "mosaic_id": 7,
                        "keywords": ["mot1", "mot2", "mot3", "mot4", "mot5"],
                        "description": "Description de la mosaïque 7 en maximum 300 mots..."
                      },
                      {
                        "mosaic_id": 8,
                        "keywords": ["mot1", "mot2", "mot3", "mot4", "mot5"],
                        "description": "Description de la mosaïque 8 en maximum 300 mots..."
                      }
                    ]
                    
                    Analysez maintenant les 8 mosaïques dans l'ordre et répondez selon ce format exact.
                    """
                }
            ]
        }
    ]

    # Get the chat response
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )

    return chat_response.choices[0].message.content




def generate_group_image(images_paths, cluster_n):
    # Définir la taille des images dans la mosaïque
    cols, rows = 5, 4
    thumb_width, thumb_height = 200, 200  # Taille des miniatures

    # Créer une image blanche pour la mosaïque
    mosaic = Image.new('RGB', (cols * thumb_width, rows * thumb_height), (255, 255, 255))

    for index, img_path in enumerate(images_paths):
        if index >= cols * rows:
            break  # S'assure de ne traiter que 20 images

        img = Image.open(img_path)
        img = img.resize((thumb_width, thumb_height))

        # Calcul des coordonnées de placement
        x_offset = (index % cols) * thumb_width
        y_offset = (index // cols) * thumb_height

        mosaic.paste(img, (x_offset, y_offset))

    # Convertir en base64
    buffered = BytesIO()
    mosaic.save(buffered, format="JPEG")
    with open(f'clusters/cluster_{cluster_n}.jpg', 'wb') as f:
        mosaic.save(f)
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return encoded_image