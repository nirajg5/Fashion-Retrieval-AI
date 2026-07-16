from config import (
    TOP_K_IMAGE,
    TOP_K_CROP,
    TOP_K_CAPTION
)

from src.models.clip_encoder import clip_encoder
from src.models.bge_encoder import bge_encoder
from src.retrieval.faiss_index import index_manager

class SearchEngine:

    def __init__(self):

        self.index = index_manager

    def image_search(

        self,

        query,

        top_k=TOP_K_IMAGE

    ):

        embedding = clip_encoder.encode_text(
            query
        )

        scores, indices = self.index.search_image(

            embedding,

            top_k

        )

        results = []

        for score, idx in zip(

            scores,

            indices

        ):

            image_id = int(

                self.index.image_ids[idx]

            )

            results.append({

                "image_id": image_id,

                "score": float(score)

            })

        return results
    
    def crop_search(

        self,

        query,

        top_k=TOP_K_CROP

    ):

        embedding = clip_encoder.encode_text(
            query
        )

        scores, indices = self.index.search_crop(

            embedding,

            top_k

        )

        results = []

        for score, idx in zip(

            scores,

            indices

        ):

            crop_id = int(

                self.index.crop_ids[idx]

            )

            crop_info = self.index.get_crop_info(

                crop_id

            )

            if crop_info is None:
                continue

            results.append({

                "crop_id": crop_id,

                "image_id": crop_info["image_id"],

                "score": float(score)

            })

        return results
    

    def caption_search(

        self,

        query,

        top_k=TOP_K_CAPTION

    ):

        embedding = bge_encoder.encode(
            query
        )

        scores, indices = self.index.search_caption(

            embedding,

            top_k

        )

        results = []

        for score, idx in zip(

            scores,

            indices

        ):

            image_id = int(

                self.index.caption_ids[idx]

            )

            results.append({

                "image_id": image_id,

                "score": float(score)

            })

        return results
    
    def search(

        self,

        query

    ):

        return {

            "image": self.image_search(
                query
            ),

            "crop": self.crop_search(
                query
            ),

            "caption": self.caption_search(
                query
            )

        }
    
    def image_to_image_search(

    self,

    image_path,

    top_k=100

):

     embedding = clip_encoder.encode_image(
        image_path
    )

     scores, indices = self.index.search_image(

        embedding,

        top_k

    )

     results = []

     for score, idx in zip(scores, indices):

        image_id = int(
            self.index.image_ids[idx]
        )

        results.append({

            "image_id": image_id,

            "score": float(score)

        })

     return results
    
search_engine = SearchEngine()