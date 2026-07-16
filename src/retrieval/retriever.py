from src.retrieval.search_engine import search_engine
from src.retrieval.fusion import fusion_engine
from src.retrieval.reranker import reranker
from src.retrieval.faiss_index import index_manager
from src.retrieval.parser import parser
from src.retrieval.query_normalizer import query_normalizer


class FashionRetriever:

    def __init__(self):

        self.search = search_engine
        self.fusion = fusion_engine
        self.reranker = reranker
        self.index = index_manager

    ############################################################
    # TEXT RETRIEVAL
    ############################################################

    def retrieve(self, query):

        # Step 1: Parse
        parsed = parser.parse(query)

        cleaned_query = parsed["cleaned"]

        # Step 2: Normalize using Qwen
        try:

            normalized_query = query_normalizer.normalize(
                cleaned_query
            )

            if (
                normalized_query is None
                or len(normalized_query.strip()) == 0
            ):
                normalized_query = cleaned_query

        except Exception as e:

            print(f"[Normalizer Error] {e}")

            normalized_query = cleaned_query

        normalized_query = normalized_query.strip()

        print("=" * 60)
        print(f"Original Query   : {query}")
        print(f"Cleaned Query    : {cleaned_query}")
        print(f"Normalized Query : {normalized_query}")
        print("=" * 60)

        # Step 3: Search
        raw_results = self.search.search(
            normalized_query
        )

        # Step 4: Fusion
        fused_results = self.fusion.fuse(
            raw_results["image"],
            raw_results["crop"],
            raw_results["caption"]
        )

        # Step 5: CrossEncoder Rerank
        reranked_results = self.reranker.rerank(
            normalized_query,
            fused_results
        )

        # Step 6: Metadata
        enriched_results = self._enrich_results(
            reranked_results
        )

        return {

            "original_query": query,

            "cleaned_query": cleaned_query,

            "normalized_query": normalized_query,

            "total_results": len(enriched_results),

            "results": enriched_results

        }

    ############################################################
    # IMAGE RETRIEVAL
    ############################################################

    def retrieve_by_image(self, image_path):

        results = self.search.image_to_image_search(
            image_path
        )

        enriched = []

        for item in results:

            image = self.index.get_image_info(
                item["image_id"]
            )

            caption = self.index.get_caption_info(
                item["image_id"]
            )

            crop = None

            try:

                crop = self.index.crop_mapping[
                    self.index.crop_mapping.image_id ==
                    item["image_id"]
                ].iloc[0].to_dict()

            except Exception:
                pass

            enriched.append({

                "image_id": item["image_id"],

                "image_path": image.get("filepath"),

                "caption": caption.get(
                    "normalized_caption",
                    caption.get("caption", "")
                ),

                "category": crop.get(
                    "category_name"
                ) if crop else None,

                "supercategory": crop.get(
                    "supercategory"
                ) if crop else None,

                "final_score": item["score"]

            })

        return {

            "total_results": len(enriched[:10]),

            "results": enriched[:10]

        }

    ############################################################
    # ENRICH RESULTS
    ############################################################

    def _enrich_results(self, results):

        enriched = []

        for item in results:

            image = self.index.get_image_info(
                item["image_id"]
            )

            caption = self.index.get_caption_info(
                item["image_id"]
            )

            crop = None

            try:

                crop = self.index.crop_mapping[
                    self.index.crop_mapping.image_id ==
                    item["image_id"]
                ].iloc[0].to_dict()

            except Exception:
                pass

            enriched.append({

                "image_id": item["image_id"],

                "image_path": image.get("filepath"),

                "caption": caption.get(
                    "normalized_caption",
                    caption.get("caption", "")
                ),

                "category": crop.get(
                    "category_name"
                ) if crop else None,

                "supercategory": crop.get(
                    "supercategory"
                ) if crop else None,

                "image_score": item.get(
                    "image_score", 0.0
                ),

                "crop_score": item.get(
                    "crop_score", 0.0
                ),

                "caption_score": item.get(
                    "caption_score", 0.0
                ),

                "fusion_score": item.get(
                    "fusion_score", 0.0
                ),

                "rerank_score": item.get(
                    "rerank_score", 0.0
                ),

                "final_score": item.get(
                    "final_score", 0.0
                )

            })

        return enriched


retriever = FashionRetriever()