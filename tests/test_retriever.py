from src.retrieval.retriever import retriever

results = retriever.retrieve(

    "blue denim jacket"

)

print()

print("=" * 70)

print("TOP RESULTS")

print("=" * 70)

for i, result in enumerate(results, start=1):

    print()

    print(f"Rank {i}")

    print("-" * 50)

    print("Image ID:", result["image_id"])

    print("Category:", result["category"])

    print("Supercategory:", result["supercategory"])

    print("Caption:", result["caption"])

    print()

    print("Image Score:", round(result["image_score"], 4))

    print("Crop Score:", round(result["crop_score"], 4))

    print("Caption Score:", round(result["caption_score"], 4))

    print("Fusion Score:", round(result["fusion_score"], 4))

    print("Rerank Score:", round(result["rerank_score"], 4))

    print("Final Score:", round(result["final_score"], 4))

    print()

    print("Image Path:")

    print(result["image_path"])