"""
test.py
Unit tests for Music Genre Classification Project.

Run with: pytest test.py -v
"""

import os
import sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preparation import extract_features, GENRES
import torch
from training_module import GenreNet

class TestFeatureExtraction:
    """Tests for feature extraction module."""
    
    def test_genres_list(self):
        """Test that GENRES contains expected genres."""
        #checks genre name
        expected_genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
                          'jazz', 'metal', 'pop', 'reggae', 'rock']
        missing = [g for g in expected_genres if g not in GENRES]
        extra = [g for g in GENRES if g not in expected_genres]
        if missing or extra:
            raise AssertionError(f"Missing: {missing}, Extra: {extra}")

    
    def test_genres_count(self):
        """Test that we have 10 genres."""
        if len(GENRES) != 10:
            raise AssertionError(f"Expected 10 genres, got {len(GENRES)}")

    
    def test_extract_features_returns_none_for_invalid_file(self):
        """Test that extract_features returns None for non-existent files."""
        result = extract_features("nonexistent_file.wav")
        if result is not None:
            raise AssertionError(f"Expected None for invalid file, got {result}")


class TestTrainingModule:
    """Tests for training module."""

    def test_genre_net(self):
        input_dim, num_classes = 111, 10
        model = GenreNet(input_dim, num_classes)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(4, input_dim))
        if out.shape != (4, num_classes):
            raise AssertionError(f"Expected output shape (4, {num_classes}), got {out.shape}")

        linear_layers = []
        for m in model.net:
            if isinstance(m, torch.nn.Linear):
                linear_layers.append(m)
        if len(linear_layers) != 4:
            raise AssertionError(f"Expected 4 linear layers, got {len(linear_layers)}")


class TestDataIntegrity:
    """Tests for data integrity checks."""
    
    def test_genre_names(self):
        seen = set()
        for genre in GENRES:
            if genre != genre.lower():
                raise AssertionError(f"Genre '{genre}' should be lowercase")
            if genre in seen:
                raise AssertionError(f"Duplicate genre found: '{genre}'")
            seen.add(genre)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])