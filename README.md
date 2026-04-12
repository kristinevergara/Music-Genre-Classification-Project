# Music Genre Classification Using Audio Signal Features and Machine Learning

**Authors:** Kristine Vergara, Richard Oh

---

## Project Overview

This project builds a machine learning model that automatically classifies songs into musical genres based on audio features extracted from raw audio files using the GTZAN dataset.

### Dataset: GTZAN
- 1,000 audio clips (30 seconds each)
- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- 100 tracks per genre

### Features Extracted
1. **Mel-Frequency Cepstral Coefficients (MFCCs)** - Capture timbral texture (80 features)
2. **Chroma features** - Represent pitch class distribution (24 features)
3. **Spectral centroid** - Brightness of sound (2 features)
4. **Spectral roll-off** - Shape of spectral energy (2 features)
5. **Zero crossing rate** - Rate of sign changes (2 features)
6. **Tempo** - Beats per minute (1 feature)

**Total: 111 features**

### Models Trained
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Feedforward Neural Network

---

## Project Structure

```
Music-Genre-Classification-Project/
├── music_genre_classification.ipynb  # Main Jupyter Notebook (complete workflow)
├── feature_extraction.py            # Feature extraction module
├── preprocessing.py                 # Data preprocessing module
├── training_module.py               # Model training module
├── evaluation.py                    # Evaluation and visualization module
├── prediction.py                    # Single file prediction script
├── test.py                          # Unit tests
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- GTZAN dataset (download from Kaggle)

### Setup

1. Clone the repository:
```bash
gh repo clone kristinevergara/Music-Genre-Classification-Project
cd Music-Genre-Classification-Project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the GTZAN dataset from Kaggle:
   - Go to: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
   - Download and extract to `./gtzan/` folder

---

## Usage

### Option 1: Jupyter Notebook (Recommended)

Run the complete workflow in the Jupyter Notebook:

```bash
jupyter notebook music_genre_classification.ipynb
```

The notebook includes:
- Data loading and exploration
- Feature extraction
- Data preprocessing
- Model training (SVM, Random Forest, KNN, Neural Network)
- Evaluation with confusion matrices
- Performance visualizations

### Option 2: Command Line Scripts

**Step 1: Extract Features**
```bash
python feature_extraction.py --data_dir ./gtzan --out features.csv
```

**Step 2: Preprocess Data**
```bash
python preprocessing.py --features features.csv --out_dir ./data
```

**Step 3: Train Models**
```bash
python training_module.py --data_dir ./data --models_dir ./models
```

**Step 4: Evaluate Models**
```bash
python evaluation.py --data_dir ./data --models_dir ./models --out_dir ./results
```

**Step 5: Predict Genre of a New Audio File**
```bash
python prediction.py --file path/to/song.wav --model all
```

---

## Results

### Model Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| SVM | ~0.75 | ~0.74 |
| Random Forest | ~0.72 | ~0.71 |
| Neural Net | ~0.70 | ~0.69 |
| KNN | ~0.65 | ~0.64 |

*Results may vary based on random seed and data split.*

### Generated Visualizations
- `model_comparison.png` - Bar chart comparing model accuracy and F1-scores
- `confusion_*.png` - Confusion matrix heatmaps for each model
- `nn_training_curves.png` - Neural network training history
- `per_genre_accuracy.png` - Per-genre classification accuracy
- `feature_importance.png` - Random Forest feature importance

---

## Testing

Run the unit tests:

```bash
pytest test.py -v
```

---

## Discussion and Limitations

### Key Findings
- SVM with RBF kernel performs well on high-dimensional audio features
- Random Forest provides interpretable feature importance
- Classical and jazz genres are easier to classify due to distinctive characteristics
- Rock and country genres may be confused due to similar instrumentation

### Limitations
1. **Dataset size**: GTZAN has only 100 samples per genre
2. **Feature limitations**: Hand-crafted features may not capture all genre-relevant information
3. **Temporal information**: Mean/std aggregation loses temporal structure

### Potential Improvements
1. **CNNs on Spectrograms**: Use convolutional neural networks directly on mel-spectrograms
2. **Data Augmentation**: Apply pitch shifting, time stretching, and noise injection
3. **Larger Datasets**: Use MagnaTagATune or Million Song Dataset
4. **Ensemble Methods**: Combine predictions from multiple models

---

## References

1. Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing, 10(5), 293-302.

2. McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. Proceedings of the 14th Python in Science Conference.

3. GTZAN Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

---

## License

This project is for educational purposes.