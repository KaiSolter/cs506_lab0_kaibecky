## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    v1 = np.array([1, 2, 3])
    v2 = np.array([5, 6, 7])
    
    result = cosine_similarity(v1, v2); 
    
    #from solving with math
    expected_result = 38 / np.sqrt(1540)
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    
    target_vector = np.array([1, 2])
    vectors = np.array([
    [2, 3],
    [1, 1],
    [4, 5]
    ])

    result = nearest_neighbor(target_vector, vectors)
    
    expected_index = 0 
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
