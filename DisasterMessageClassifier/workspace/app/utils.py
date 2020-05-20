import numpy as np

def compute_text_length(data):
    """
    Compute the character length of texts
    """
    return np.array([len(text) for text in data]).reshape(-1, 1)