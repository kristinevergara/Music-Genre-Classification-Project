Project Title: Music Genre Classification Using Audio Signal Features and Machine Learning


Student Name(s): Kristine Vergara, Richard Oh


Objective (what you propose to do): 
The goal of this project is to build a machine learning model that can automatically classify songs into musical genres based on audio features extracted from raw audio files. Using the GTZAN dataset, which is the most widely used music data set, the ML model will detect and extract meaningful audio features . We will use the librosa library to train classifiers to identify genres such as classical, country, hip-hop, jazz, and pop.



Significance (why the proposed project is interesting/important):
Automatic music genre classification is an ongoing issue in music streaming platforms such as Apple Music and Spotify as recommendation systems and digital music libraries sometimes encounter errors in classification. This project demonstrates the application of signal processing on machine learning by teaching a computer to hear patterns that humans associate with genre. Other topics such as speech recognition can be also incorporated, which will give us an opportunity to learn further on the applications of ML.

Methodology (how do you plan to achieve the objective, e.g., using simulation, critique, math
analysis, etc.):
The project will proceed in the following stages:
Data Acquisition: Download the GTZAN dataset from Kaggle (1,000 audio clips, 30 seconds each, 10 genres, 100 tracks per genre).
Feature Extraction (librosa): Extract Mel-Frequency Cepstral Coefficients (MFCCs), Chroma features, Spectral centroid, Spectral roll-off, Zero crossing rate, and Tempo/beat features.
Preprocessing: Normalize features, balance each class, and split into 80/20 train-test sets.
Model Training: Train and compare Support Vector Machine (SVM), Random Forest, K-Nearest Neighbors (KNN), and a simple Feedforward Neural Network (using scikit-learn / TensorFlow / Keras).
Evaluation: Compare models using accuracy, confusion matrix, precision, recall, and F1-score.
Tools: Python, librosa, scikit-learn, pandas, numpy, matplotlib, seaborn.


Delivery (what you expect to provide in the final report, e.g. code, graphs, text writing, etc.)
The final report and submission will include:
Python code (Jupyter Notebook) for data loading, feature extraction, model training, and evaluation.
Confusion matrix heatmaps for each classifier showing per-genre classification accuracy.
Bar charts comparing model performance (accuracy, F1-score) across all classifiers.
Written report discussing methodology, results, limitations, and potential improvements such as using CNNs on spectrograms.
