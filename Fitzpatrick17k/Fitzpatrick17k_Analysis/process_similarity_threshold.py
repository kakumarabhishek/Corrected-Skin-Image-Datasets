import numpy as np

from definitions import *
from utils import *
from config import *


def do_process_similarity_threshold(max_similarities, file_list_filtered):
    assert 0 <= SIMILARITY_THRESHOLD_LOWER <= 1
    assert 0 <= SIMILARITY_THRESHOLD_UPPER <= 1
    assert SIMILARITY_THRESHOLD_LOWER < SIMILARITY_THRESHOLD_UPPER

    sim_thresh_mask = (max_similarities < SIMILARITY_THRESHOLD_UPPER) & (
        max_similarities > SIMILARITY_THRESHOLD_LOWER
    )
    file_list_filtered = np.array(file_list_filtered)[sim_thresh_mask].tolist()

    print(
        f"Number of images after applying similarity threshold: {len(file_list_filtered)}"
    )

    return file_list_filtered
