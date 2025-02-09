import math

import numpy as np
import vrplib
from pprint import pprint


instance = vrplib.read_instance("Vrp-Set-A/A-n69-k9.vrp")
pprint(instance)


def get_sum_dist(route: list, coordinates: np.ndarray):
    '''
    Функция для валидации качества подсчёта стоимости пути.
    '''
    
    dist = 0.0
    for i in range(len(route) - 1):
        node_id_left = route[i] + 1
        node_id_right = route[i + 1] + 1
        res = math.sqrt((coordinates[node_id_left - 1][0] - coordinates[node_id_right - 1][0]) ** 2 + (coordinates[node_id_left - 1][1] - coordinates[node_id_right - 1][1]) ** 2)
        print(node_id_left, node_id_right, res)
        dist += res
    
    return dist
