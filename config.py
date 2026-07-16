from pathlib import Path
import torch

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

# --------------------------
# Data
# --------------------------

DATA_DIR = PROJECT_DIR / "data"

IMAGE_DIR = DATA_DIR / "images"

METADATA_DIR = DATA_DIR / "metadata"

MANIFEST_FILE = DATA_DIR / "manifest.csv"

ANNOTATION_FILE = DATA_DIR / "image_annotations.csv"

# --------------------------
# Outputs
# --------------------------

OUTPUT_DIR = PROJECT_DIR / "outputs"

CAPTION_DIR = OUTPUT_DIR / "captions"

EMBEDDING_DIR = OUTPUT_DIR / "embeddings"

# --------------------------
# Indexes
# --------------------------

INDEX_DIR = PROJECT_DIR / "indexes"

IMAGE_INDEX = INDEX_DIR / "image_index.faiss"

CROP_INDEX = INDEX_DIR / "crop_index.faiss"

CAPTION_INDEX = INDEX_DIR / "caption_index.faiss"

# --------------------------
# Embeddings
# --------------------------

IMAGE_EMBEDDINGS = EMBEDDING_DIR / "image_embeddings.npy"

CROP_EMBEDDINGS = EMBEDDING_DIR / "crop_embeddings.npy"

CAPTION_EMBEDDINGS = EMBEDDING_DIR / "caption_embeddings.npy"

IMAGE_IDS = EMBEDDING_DIR / "image_ids.npy"

CROP_IDS = EMBEDDING_DIR / "crop_ids.npy"

CAPTION_IDS = EMBEDDING_DIR / "caption_ids.npy"

# --------------------------
# Mapping Files
# --------------------------

IMAGE_MAPPING = EMBEDDING_DIR / "image_mapping.csv"

CROP_MAPPING = EMBEDDING_DIR / "crop_mapping.csv"

CAPTION_MAPPING = EMBEDDING_DIR / "caption_mapping.csv"

CAPTION_FILE = CAPTION_DIR / "captions.csv"

# ==========================================================
# MODELS
# ==========================================================

OPENCLIP_MODEL = "ViT-B-32"

OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"

BGE_MODEL = "BAAI/bge-large-en-v1.5"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ==========================================================
# RETRIEVAL SETTINGS
# ==========================================================

TOP_K_IMAGE = 100

TOP_K_CROP = 100

TOP_K_CAPTION = 100

RERANK_TOP_K = 50

FINAL_TOP_K = 10

IMAGE_WEIGHT = 0.45

CROP_WEIGHT = 0.35

CAPTION_WEIGHT = 0.20

RERANK_WEIGHT = 0.70

FUSION_WEIGHT = 0.30

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"