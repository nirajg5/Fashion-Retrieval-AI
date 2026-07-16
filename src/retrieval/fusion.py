from collections import defaultdict

import numpy as np

from config import (

    IMAGE_WEIGHT,

    CROP_WEIGHT,

    CAPTION_WEIGHT,

    RERANK_TOP_K

)

class FusionEngine:

    def __init__(self):

        pass

    def normalize_scores(

        self,

        results

    ):

        if len(results) == 0:

            return results

        scores = np.array(

            [r["score"] for r in results]

        )

        minimum = scores.min()

        maximum = scores.max()

        if maximum == minimum:

            for r in results:

                r["normalized"] = 1.0

            return results

        for r in results:

            r["normalized"] = (

                r["score"] - minimum

            ) / (

                maximum - minimum

            )

        return results
    

    def add_image_scores(

        self,

        fused,

        image_results

    ):

        image_results = self.normalize_scores(

            image_results

        )

        for item in image_results:

            image_id = item["image_id"]

            fused[image_id]["image"] = (

                item["normalized"]

            )

    def add_crop_scores(

        self,

        fused,

        crop_results

    ):

        crop_results = self.normalize_scores(

            crop_results

        )

        for item in crop_results:

            image_id = item["image_id"]

            current = fused[image_id]["crop"]

            fused[image_id]["crop"] = max(

                current,

                item["normalized"]

            )

    def add_caption_scores(

        self,

        fused,

        caption_results

    ):

        caption_results = self.normalize_scores(

            caption_results

        )

        for item in caption_results:

            image_id = item["image_id"]

            fused[image_id]["caption"] = (

                item["normalized"]

            )

    def compute_scores(

        self,

        fused

    ):

        results = []

        for image_id, scores in fused.items():

            final_score = (

                IMAGE_WEIGHT * scores["image"]

                +

                CROP_WEIGHT * scores["crop"]

                +

                CAPTION_WEIGHT * scores["caption"]

            )

            results.append({

                "image_id": image_id,

                "image_score": scores["image"],

                "crop_score": scores["crop"],

                "caption_score": scores["caption"],

                "fusion_score": final_score

            })

        return results
    

    def fuse(

        self,

        image_results,

        crop_results,

        caption_results

    ):

        fused = defaultdict(

            lambda: {

                "image":0,

                "crop":0,

                "caption":0

            }

        )

        self.add_image_scores(

            fused,

            image_results

        )

        self.add_crop_scores(

            fused,

            crop_results

        )

        self.add_caption_scores(

            fused,

            caption_results

        )

        results = self.compute_scores(

            fused

        )

        results = sorted(

            results,

            key=lambda x: x["fusion_score"],

            reverse=True

        )

        return results[:RERANK_TOP_K]
    
fusion_engine = FusionEngine()