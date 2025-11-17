from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .utils import preprocess_image
if TYPE_CHECKING:
    from .panoptic_ml import PanopticML

import aiofiles

from panoptic.core.task.task import Task
from panoptic.models import Instance, Vector, VectorType

logger = logging.getLogger('PanopticML:VectorTask')


class ComputeVectorTask(Task):
    def __init__(self, plugin: PanopticML, vec_type: VectorType, instance: Instance,
                 data_path: str, transformer):
        super().__init__()
        self.project = plugin.project
        self.plugin: PanopticML = plugin
        self.type = vec_type
        self.instance = instance
        self.vec_type = vec_type
        self.name = f'{vec_type.params['model']} Vectors ({vec_type.id})'
        self.data_path = data_path
        self.key += f"vec_id_{vec_type.id}"
        self.transformer = transformer

    async def run(self):
        instance = self.instance
        exist = await self.project.vector_exist(self.type.id, instance.sha1)
        if exist:
            return

        image_data = await self._project.db.get_large_image(instance.sha1)
        if not image_data:
            file = instance.url
            async with aiofiles.open(file, mode='rb') as file:
                image_data = await file.read()

        vector_data = await self.run_async(self.compute_image_vector, image_data)

        if vector_data is None:
            return
        vector = Vector(self.type.id, instance.sha1, vector_data)
        res = await self.project.add_vector(vector)
        del vector
        return res

    async def run_if_last(self):
        await self.plugin.trees.rebuild_tree(self.type)


    def compute_image_vector(self, image_data: bytes):
        image = preprocess_image(image_data, self.type.params)
        vector = self.transformer.to_vector(image)

        del image
        return vector
