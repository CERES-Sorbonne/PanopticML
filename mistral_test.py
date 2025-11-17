import json

from PIL import Image
import base64
from io import BytesIO
from mistralai import Mistral
import os

model = "pixtral-12b-2409"
api_key = os.getenv("MISTRAL_KEY")
client = Mistral(api_key=api_key)


def create_batch(liste, batch_size):
    return [liste[i:i + batch_size] for i in range(0, len(liste), batch_size)]


def create_labels_from_group(groups_images):
    all_labels = []
    for batch in create_batch(groups_images, 8):
        invalid = True
        while invalid:
            labels = create_labels_from_batch(batch)
            try:
                res = json.loads(labels)
                invalid = False
            except json.JSONDecodeError:
                print("Invalid JSON")
        all_labels.extend(res)
    return all_labels


def create_labels_from_batch(groups_images):
    messages = [
        {
            "role": "user",
            "content": []
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": """[[
                """
            }],
            "prefix": "true"
        }
    ]

    for base64_image in groups_images:
        messages[0]['content'].append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
        })
    messages[0]['content'].append({
                    "type": "text",
                    "text": """\
                    Chaque image est une mosaïque du contenu d'un cluster d'images.
                    Donne CINQ (5) mots-clés (par mosaïque) et UNIQUEMENT ces mots-clés, sans aucun autre texte 
                    sous forme de liste de listes qui devra être un JSON valide, exemple [[mot_cle1, mot_cle2, mot_cle3, mot_cle4, mot_cle5],[...]]."""  # , séparez ces mots par des virgules.
                    # Essayez d'inclure des mots-clés généraux qui pourraient décrire le contenu de nombreuses images à la fois.
                    # Veuillez séparer les mots-clés pour chaque mosaïque avec un point-virgule et NE PAS INCLURE de texte supplémentaire.
                    f"""Assure toi de ne répondre qu'en français tu me donneras EXACTEMENT {len(groups_images)} LISTES (soit une par image) et fais bien attention à respecter L'ORDRE des images !.
                    """
                })
    # Get the chat response
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )

    # Print the content of the response
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
