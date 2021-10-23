import yaml

def load_config(filename):
    params = {}
    
    with open(filename) as f:
        params = yaml.load(f)

    return params

