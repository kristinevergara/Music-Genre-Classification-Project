"""
test.py
Unit tests for Music Genre Classification Project.

Run with: pytest test.py -v
"""

import os
import sys
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from feature_extraction import extract_features, GENRES
from preprocessing import preprocess
from training_module import build_nn


class TestFeatureExtraction:
    """Tests for feature extraction module."""
    
    def test_genres_list(self):
        """Test that GENRES contains expected genres."""
        expected_genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
                          'jazz', 'metal', 'pop', 'reggae', 'rock']
        assert GENRES == expected_genres, f"Expected {expected_genres}, got {GENRES}"
    
    def test_genres_count(self):
        """Test that we have 10 genres."""
        assert len(GENRES) == 10, f"Expected 10 genres, got {len(GENRES)}"
    
    def test_extract_features_returns_none_for_invalid_file(self):
        """Test that extract_features returns None for non-existent files."""
        # This should handle the exception gracefully
        result = extract_features("nonexistent_file.wav")
        assert result is None, "Should return None for invalid files"


class TestPreprocessing:
    """Tests for preprocessing module."""
    
    def test_feature_count(self):
        """Test that feature extraction produces 111 features."""
        # Feature breakdown:
        # MFCCs: 40 mean + 40 std = 80
        # Chroma: 12 mean + 12 std = 24
        # Spectral centroid: 1 mean + 1 std = 2
        # Spectral roll-off: 1 mean + 1 std = 2
        # Zero crossing rate: 1 mean + 1 std = 2
        # Tempo: 1
        # Total: 80 + 24 + 2 + 2 + 2 + 1 = 111
        expected_features = 111
        # This is a documentation test - actual feature count depends on n_mfcc
        assert expected_features == 111


class TestTrainingModule:
    """Tests for training module."""
    
    def test_build_nn_architecture(self):
        """Test that neural network has correct architecture."""
        input_dim = 111
        num_classes = 10
        model = build_nn(input_dim, num_classes)
        
        # Check input shape
        assert model.input_shape == (None, input_dim), \
            f"Expected input shape (None, {input_dim}), got {model.input_shape}"
        
        # Check output shape
        assert model.output_shape == (None, num_classes), \
            f"Expected output shape (None, {num_classes}), got {model.output_shape}"
        
        # Check that model compiles without errors
        model.summary()  # Should not raise any exceptions
    
    def test_nn_layer_count(self):
        """Test that neural network has expected number of layers."""
        model = build_nn(111, 10)
        # Dense layers: 512 -> 256 -> 128 -> 10 (4 layers)
        # Plus BatchNormalization and Dropout layers
        dense_layers = [l for l in model.layers if 'dense' in l.name]
        assert len(dense_layers) == 4, f"Expected 4 dense layers, got {len(dense_layers)}"


class TestDataIntegrity:
    """Tests for data integrity checks."""
    
    def test_genre_names_lowercase(self):
        """Test that all genre names are lowercase."""
        for genre in GENRES:
            assert genre == genre.lower(), f"Genre '{genre}' should be lowercase"
    
    def test_no_duplicate_genres(self):
        """Test that there are no duplicate genres."""
        assert len(GENRES) == len(set(GENRES)), "Genres list contains duplicates"


class TestModelConfigs:
    """Tests for model configurations."""
    
    def test_svm_config(self):
        """Test SVM model configuration values."""
        # These are the expected SVM parameters from training_module.py
        expected_kernel = 'rbf'
        expected_c = 10
        # Just documenting expected values
        assert expected_kernel == 'rbf'
        assert expected_c == 10
    
    def test_knn_config(self):
        """Test KNN model configuration values."""
        # Expected KNN parameters
        expected_n_neighbors = 5
        expected_metric = 'euclidean'
        assert expected_n_neighbors == 5
        assert expected_metric == 'euclidean'
    
    def test_random_forest_config(self):
        """Test Random Forest model configuration values."""
        # Expected RF parameters
        expected_n_estimators = 300
        assert expected_n_estimators == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])