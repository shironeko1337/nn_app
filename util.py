import numpy as np

def rectified_normalize(np_arr, default_probability):
    np_arr = np.where(np_arr, np_arr > 0, 0)
    np_arr_sum = sum(np_arr)
    return np_arr / np_arr_sum if np_arr_sum > 0 else np.array([default_probability] * len(np_arr))

def normalize(np_arr, default_probability):
    np_arr_sum = sum(np_arr)
    return np_arr / np_arr_sum if np_arr_sum > 0 else np.array([default_probability] * len(np_arr))
