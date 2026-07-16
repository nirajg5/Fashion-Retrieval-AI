# 👗 FashionLens AI
### Multimodal Fashion Retrieval System

FashionLens AI is an intelligent fashion search engine that retrieves visually and semantically similar garments using natural language queries or reference images.

The system combines **OpenCLIP**, **BGE Embeddings**, **FAISS Vector Search**, and **Cross-Encoder Reranking** to deliver accurate and fast fashion retrieval.

---

# Features

✅ Text-to-Image Retrieval

✅ Image-to-Image Retrieval

✅ Natural Language Query Normalization

✅ OpenCLIP Visual Embeddings

✅ BGE Semantic Caption Embeddings

✅ FAISS Approximate Nearest Neighbor Search

✅ Reciprocal Rank Fusion

✅ CrossEncoder Reranking

✅ FastAPI Backend

✅ Streamlit Frontend

---

# System Architecture

```
                        User
                         │
      ┌──────────────────┴──────────────────┐
      │                                     │
      ▼                                     ▼
Text Query                          Upload Image
      │                                     │
      ▼                                     ▼
 Query Parser                      OpenCLIP Encoder
      │
      ▼
 Query Normalizer
      │
      ▼
OpenCLIP Text Encoder
      │
      ▼
Image Embedding Search
      │
      ├───────────────┐
      │               │
      ▼               ▼
Crop Search     Caption Search (BGE)
      │               │
      └───────┬───────┘
              ▼
        Score Fusion
              │
              ▼
 CrossEncoder Reranker
              │
              ▼
        Top-K Results
```

---

# Retrieval Pipeline

```
Natural Language Query

        │

        ▼

Query Cleaning

        │

        ▼

Query Normalization

        │

        ▼

OpenCLIP Encoding

        │

        ▼

Image Search

        │

        ▼

Crop Search

        │

        ▼

Caption Search (BGE)

        │

        ▼

Fusion

        │

        ▼

CrossEncoder Reranking

        │

        ▼

Top Results
```

---

# Project Structure

```
fashion-retrieval/

│

├── app.py
├── main.py
├── config.py
├── requirements.txt

├── data/

│   ├── images/
│   ├── metadata/
│   ├── image_annotations.csv
│   └── manifest.csv

├── indexes/

│   ├── image_index.faiss
│   ├── crop_index.faiss
│   └── caption_index.faiss

├── outputs/

│   ├── captions/
│   ├── crops/
│   ├── embeddings/
│   └── crop_metadata.csv

├── src/

│   ├── api/
│   │
│   ├── models/
│   │
│   ├── retrieval/
│   │
│   └── utils/

└── tests/
```

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Backend | FastAPI |
| Frontend | Streamlit |
| Vision Model | OpenCLIP |
| Text Embedding | BGE |
| Vector Search | FAISS |
| Reranker | BAAI CrossEncoder |
| Image Processing | Pillow |
| Deep Learning | PyTorch |
| Data Handling | Pandas, NumPy |

---

# Dataset

Fashionpedia Dataset

Contains

- Fashion Images
- Categories
- Attributes
- Bounding Boxes
- Segmentation Masks

---

# Pipeline

## 1 Dataset Preparation

- Load annotations
- Merge metadata
- Generate garment crops

---

## 2 Image Embeddings

Generate OpenCLIP embeddings for

- Original Images
- Garment Crops

---

## 3 Caption Generation

Generate detailed garment captions.

Example

```
Black denim jacket with long sleeves, front buttons and chest pockets.
```

---

## 4 Caption Normalization

Normalize captions for better semantic retrieval.

Example

Before

```
black jacket, black denim jacket, jacket
```

After

```
black denim jacket with front buttons
```

---

## 5 Text Embeddings

Generate BGE embeddings from normalized captions.

---

## 6 Build Vector Database

Build

- Image FAISS Index

- Crop FAISS Index

- Caption FAISS Index

---

## 7 Retrieval

For every query

```
User Query

↓

OpenCLIP

↓

Image Search

↓

Crop Search

↓

Caption Search

↓

Fusion

↓

CrossEncoder

↓

Top Results
```

---

# API Endpoints

## Health

```
GET /
```

---

## Text Search

```
POST /search/text
```

Example

```json
{
  "query":"black denim jacket"
}
```

---

## Image Search

```
POST /search/image
```

Upload

```
image.jpg
```

Returns similar garments.

---

# Example Response

```json
{
    "original_query":"black denim jacket",

    "normalized_query":"black denim jacket",

    "results":[
        {
            "image_id":234,

            "category":"Jacket",

            "score":0.97
        }
    ]
}
```

---

# Running the Project

Clone

```
git clone https://github.com/your-repo/FashionLens-AI.git
```

Install

```
pip install -r requirements.txt
```

Run Backend

```
uvicorn main:app --reload
```

FastAPI Docs

```
http://127.0.0.1:8000/docs
```

Run Frontend

```
streamlit run app.py
```

---

# Models Used

## OpenCLIP

Used for

- Image Embeddings

- Text Embeddings

---

## BGE

Used for

Semantic Caption Retrieval

---

## CrossEncoder

Used for

Final reranking of retrieved garments.

---

# Future Improvements

- Outfit Recommendation

- Personalized Fashion Search

- Brand Filtering

- Color Filtering

- Price Filtering

- Hybrid Search

- Mobile Application

---

# Author

**Niraj Gahukar**

Electronics and Telecommunication Engineering

Vishwakarma Institute of Information Technology, Pune

AI • Machine Learning • Generative AI • Computer Vision