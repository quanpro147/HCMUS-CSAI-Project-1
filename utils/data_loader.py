import json
import numpy as np

# Load TSP test case từ file JSON
# Dạng file:
# {
#   "coords": [[x1, y1], [x2, y2], ...],
#   (hoặc) "distance_matrix": [[..], ...]
# }
def load_tsp_case(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Load grid test case từ JSON
# Dạng file:
# {
#   "grid": [[0,1,0,...], [0,0,1,...], ...],
#   "start": [y, x],
#   "goal": [y, x]
# }
def load_grid_case(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
