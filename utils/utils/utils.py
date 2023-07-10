import os 
import json
import socket


def mklogdir(folder, version, verbose=False):
    """make log folder to save experiment's results
    Args:
        version (str):
        verbose (bool): default is false
    """
    if verbose:
        print(f'\tHostname: {socket.gethostname()}')
        print(f'\tPath: {folder} ({version})')

    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(os.path.join(folder, 'best-param'))
        os.makedirs(os.path.join(folder, 'tensorboard'))
    else:
        raise Exception("file already exists...", folder)

def save_dict_to_json(d: dict, json_path: str):
    """Save dict of floats in json file
    Args:
        d: dict
        json_path: (string) path to json file
    """
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: v for k, v in d.items()}
            json.dump(d, f, indent=4)
    else:
        with open(json_path, 'r') as f:
            jdict = json.load(f)
        for key, val in d.items():
            jdict[key] = val
        with open(json_path, 'w') as f:
            json.dump(jdict, f, indent=4)

def save_argparser(parser, save_dir) -> dict:

    jsummary = {}
    for key, val in vars(parser).items():
        jsummary[key] = val

    save_dict_to_json(jsummary, os.path.join(save_dir, 'param-summary.json'))

    return jsummary

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(json_path)
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    def _update(self, d: dict):
        self.__dict__.update(d)
    
    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__