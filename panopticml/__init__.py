import logging

from panopticml.compute.transformer import get_transformer, type_to_class_mapping

from .panoptic_ml import PanopticML

plugin_class = PanopticML

def init_model(model_name="clip"):
    """
    utility function that allows to download a model without starting panoptic and importing images
    :param model_name: name of the model, default to clip
    """
    logging.info("Initializing model clip")
    model = type_to_class_mapping[model_name]("openai/clip-vit-base-patch32")
    return model