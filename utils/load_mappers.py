import json
import os

def load_mappers():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'target_mapper.json'), 'r') as f:
        target_mapper = json.load(f)

    with open(os.path.join(dir_path, 'category_mapper.json'), 'r') as f:
        category_mapper = json.load(f)

    return target_mapper, category_mapper