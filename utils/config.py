import yaml, os


def read_yaml(fpath):
    with open(fpath, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data


def extract_suffix(fpath):
    basepath = os.path.basename(fpath)
    assert basepath.endswith('.yaml') and basepath.startswith('config')
    
    suffix = basepath.rstrip('.yaml').lstrip('config_')
    return suffix
    
    
class Config:
    def __init__(self, config_fpath):
        config_data = read_yaml(config_fpath)
        for key, value in config_data.items():
            setattr(self, key, value)