<div align="center">

# 🎬 Movie Recommendation System  
### *Smart, Genre-Aware, Rating-Optimized Movie Discovery*

<img src="https://img.shields.io/badge/Machine%20Learning-TF--IDF-blueviolet">
<img src="https://img.shields.io/badge/Backend-Flask-black">
<img src="https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JS-blue">

</div>

---

## 🌟 Overview

This project is a **content-based movie recommendation system** that intelligently suggests movies based on:

- 🎭 **Genre similarity**
- ⭐ **Movie ratings**
- ⚖️ **Weighted ranking (similarity + quality)**

Unlike naive keyword-based recommenders, this system:
- **Ignores misleading title similarity**
- **Scales safely without building massive similarity matrices**

It offers **two powerful recommendation modes** through a modern web UI.

---

## 🚀 Key Features

### 🔎 Recommend by Movie
- Input any movie from the dataset
- Finds **genre-similar movies**
- Ranks them using a **weighted score**
- Handles typos with **fuzzy matching suggestions**

### 🎭 Recommend by Genre(s)
- Select **one or multiple genres**
- Uses **AND-logic** (all selected genres must match)
- Returns **top-rated movies**
- Random sampling ensures **non-repetitive results**

### 🧠 Smart ML Design
- Single **TF-IDF model** trained on genres
- **On-demand similarity computation** (memory-safe)
- Weighted ranking:
-  **weighted_score = (α × similarity) + (β × normalized_rating)**

---

## 🖥️ User Interface Highlights

- ✨ Glassmorphic design
- 🎞️ IMDb quick-links
- 📝 Hoverable movie overviews
- 🔄 Smooth animations & transitions
- 📱 Fully responsive layout

<div align="center">

### 💻 Webpage Design

![alt text](sampleScreenshots/Screenshot%20(1860).png)

### 📊 Movie Recommendations

![alt text](sampleScreenshots/Screenshot%20(1861).png)

![alt text](sampleScreenshots/Screenshot%20(1863).png)

![alt text](sampleScreenshots/Screenshot%20(1862).png)

</div>

---

## 🏗️ System Architecture

<pre>
┌────────────┐
│ Frontend │ ← Glassmorphic HTML UI
└─────┬──────┘
│ HTTP (JSON)
┌─────▼──────┐
│ Flask API │ ← app.py
└─────┬──────┘
│ Python Import
┌─────▼──────┐
│ ML Model │ ← model.py (TF-IDF + Logic)
└─────┬──────┘
│     |
┌─────▼──────┐
│ Dataset │ ← CSV
└────────────┘
</pre>


✔ Clean separation of concerns  
✔ One model, multiple recommendation paths

---

## 🧪 Recommendation Logic (Deep Dive)

### 📌 Genre Vectorization
- Genres are normalized:
  - `Science Fiction → Sci-Fi`
- Converted to vectors using **TF-IDF**

### 📌 Similarity
- Computed using **cosine similarity**
- Implemented via `linear_kernel` (efficient on sparse matrices)

### 📌 Ranking Strategy
Instead of ranking only by similarity:
- **weighted_score = (0.7 × genre_similarity) + (0.3 × normalized_rating)**

This ensures:
- Highly similar movies are preferred
- Poorly rated movies don’t dominate results

---

## 📊 Visualizations (Jupyter Notebook)

The notebook includes **meaningful, explainable plots**:

- ⭐ Rating distribution (justifies mean imputation)
- 🎭 Genre frequency analysis
- 📊 Recommendation ranking (weighted score)
- ⚖️ Similarity vs Rating trade-off plots

These plots **prove** the model works — not just that it runs.

---

## 🔌 API Endpoints

### 🎬 Recommend by Movie

- **POST /recommend/movie**

```json
{
  "movie_name": "The Avengers"
}
```

- **Recommend by Genre**

```json
{
  "genres": ["Sci-Fi", "Thriller"]
}
```

---

## 🛠️ Tech Stack

| Layer     | Technology                |
| --------- | ------------------------- |
| ML Model  | TF-IDF, Cosine Similarity |
| Backend   | Flask                     |
| Frontend  | HTML, CSS, JavaScript     |
| Utilities | Pandas, NumPy, RapidFuzz  |

---

## ▶️ How to Run Locally

### Clone the repository
``` bash
git clone https://github.com/SACHIN-S-2004/Movie-Recommendation-System.git
```

### Navigate to project directory
```bash
cd Movie-Recommendation-System
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Flask app
```bash
python app.py
```

Open in browser:

    http://127.0.0.1:5000

---
