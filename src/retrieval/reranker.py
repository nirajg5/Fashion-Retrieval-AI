from config import FINAL_TOP_K
from src.models.model_loader import loader
from src.retrieval.faiss_index import index_manager


class Reranker:

    def __init__(self):

        self.model = loader.load_reranker()

    def rerank(self, query, candidates):

        pairs = []

        valid_candidates = []

        for candidate in candidates:

            caption = index_manager.get_caption_info(
                candidate["image_id"]
            )

            if caption is None:
                continue

            text = caption.get(
                "normalized_caption",
                caption.get("caption", "")
            )

            pairs.append([query, text])
            valid_candidates.append(candidate)

        if len(pairs) == 0:
            return []

        scores = self.model.predict(pairs)

        for candidate, score in zip(valid_candidates, scores):

            candidate["rerank_score"] = float(score)

            candidate["final_score"] = (
                0.30 * candidate["fusion_score"]
                +
                0.70 * float(score)
            )

        valid_candidates.sort(

            key=lambda x: x["final_score"],

            reverse=True

        )

        return valid_candidates[:FINAL_TOP_K]


reranker = Reranker()