from src.retrieval.search_engine import search_engine
from src.retrieval.fusion import fusion_engine
from src.retrieval.reranker import reranker


query = "blue denim jacket"

results = search_engine.search(query)

fusion = fusion_engine.fuse(

    results["image"],

    results["crop"],

    results["caption"]

)

reranked = reranker.rerank(

    query,

    fusion

)

print()

print("=" * 60)

print("TOP RESULTS")

print("=" * 60)

for r in reranked:

    print(r)