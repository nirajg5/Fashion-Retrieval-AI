"""
FAISS Index Manager

Loads

1. Image Index
2. Crop Index
3. Caption Index

Loads corresponding
IDs and mapping files.
"""

import faiss
import numpy as np
import pandas as pd

from config import (
    IMAGE_INDEX,
    CROP_INDEX,
    CAPTION_INDEX,
    IMAGE_IDS,
    CROP_IDS,
    CAPTION_IDS,
    IMAGE_MAPPING,
    CROP_MAPPING,
    CAPTION_MAPPING
)


class FAISSIndexManager:

    def __init__(self):

        self.image_index = None
        self.crop_index = None
        self.caption_index = None

        self.image_ids = None
        self.crop_ids = None
        self.caption_ids = None

        self.image_mapping = None
        self.crop_mapping = None
        self.caption_mapping = None

        self.load()

    def load(self):

        print("Loading FAISS Indexes...")

        self.image_index = faiss.read_index(
            str(IMAGE_INDEX)
        )

        self.crop_index = faiss.read_index(
            str(CROP_INDEX)
        )

        self.caption_index = faiss.read_index(
            str(CAPTION_INDEX)
        )

        print("✓ Indexes Loaded")

        print()

        print("Loading IDs...")

        self.image_ids = np.load(IMAGE_IDS)

        self.crop_ids = np.load(CROP_IDS)

        self.caption_ids = np.load(CAPTION_IDS)

        print("✓ IDs Loaded")

        print()

        print("Loading Mapping Files...")

        self.image_mapping = pd.read_csv(
            IMAGE_MAPPING
        )

        self.crop_mapping = pd.read_csv(
            CROP_MAPPING
        )

        self.caption_mapping = pd.read_csv(
            CAPTION_MAPPING
        )

        print("✓ Mapping Loaded")

    def search_image(

        self,

        embedding,

        top_k=100

    ):

        scores, indices = self.image_index.search(

            embedding,

            top_k

        )

        return scores[0], indices[0]
    

    def search_crop(

        self,

        embedding,

        top_k=100

    ):

        scores, indices = self.crop_index.search(

            embedding,

            top_k

        )

        return scores[0], indices[0]
    

    def search_caption(

        self,

        embedding,

        top_k=100

    ):

        scores, indices = self.caption_index.search(

            embedding,

            top_k

        )

        return scores[0], indices[0]
    

    def get_image_info(

        self,

        image_id

    ):

        row = self.image_mapping[

            self.image_mapping.image_id == image_id

        ]

        if len(row) == 0:

            return None

        return row.iloc[0].to_dict()
    

    def get_crop_info(

        self,

        crop_id

    ):

        row = self.crop_mapping[

            self.crop_mapping.crop_id == crop_id

        ]

        if len(row) == 0:

            return None

        return row.iloc[0].to_dict()
    

    def get_caption_info(

        self,

        image_id

    ):

        row = self.caption_mapping[

            self.caption_mapping.image_id == image_id

        ]

        if len(row) == 0:

            return None

        return row.iloc[0].to_dict()
    

index_manager = FAISSIndexManager()


    