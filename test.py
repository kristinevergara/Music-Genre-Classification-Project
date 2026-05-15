import os
import sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preparation import extract_features, GENRES
import torch
from training_module import GenreNet

class TestFeatureExtraction:    
    def test_genres_list(self):
        expected_genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
                          'jazz', 'metal', 'pop', 'reggae', 'rock']
        assert GENRES == expected_genres    
    
    def test_genres_count(self):
        assert len(GENRES) == 10
 
    def test_extract_features_returns_none_for_invalid_file(self):
        assert extract_features("nonexistent_file.wav") is None


class TestTrainingModule:
    def test_genre_net(self):
        model = GenreNet(111, 10)
        with torch.no_grad():
            out = model(torch.zeros(4, 111))
        assert out.shape == (4, 10)


class TestDataIntegrity:    
    def test_genre_names(self):
        assert all(g == g.lower() for g in GENRES)
        assert len(GENRES) == len(set(GENRES))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])