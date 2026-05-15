# Music-Genre-Classification-Project

**Authors:** Kristine Vergara, Richard Oh

```
Music-Genre-Classification-Project/
├── Data/
│   └── genres_original/
├── data/
├── models/
├── results/
├── data_preparation.py
├── training_module.py
├── evaluation.py
├── prediction.py
├── main.py
├── test.py
└── requirements.txt
```

**data_preparation.py** - Extracts audio features (MFCCs, chroma, spectral centroid, rolloff, ZCR, tempo) from the GTZAN dataset and saves train/test splits to `/data/`  
**training_module.py** - Trains SVM, Random Forest, KNN, and a feedforward neural network, saving all models to `/models/`  
**evaluation.py** - Loads saved models, computes accuracy and F1-score, and generates confusion matrices and comparison charts to `/results/`  
**prediction.py** - Predicts the genre of a single `.wav` file using a specified trained model  
**main.py** - Runs the full pipeline: feature extraction → training → evaluation  
**test.py** - Unit tests for feature extraction and the neural network architecture  

## Steps
1. `pip3 install -r requirements.txt`
2. Download the GTZAN dataset from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and extract to `./Data/genres_original/`
3. `python3 main.py`

To predict a single file:  
`python3 prediction.py --file path/to/song.wav --model all`
