from os.path import abspath, dirname, join
from pathlib import Path

def  get_absolute_path(child_dir, parent_dir):        
    """
        given a child_dir along side with its parent_dir
        build the absolute path of the child_dir 
    """
    path = ['/']
    for dir in str(dirname(abspath(__file__))).split('/'):
        path.append('/'+dir)
        if str(dir) == parent_dir:
            path.append('/'+child_dir)
            break

    return Path(''.join(path))

data_dir = get_absolute_path(child_dir='data', parent_dir='cifar100')
model_dir = get_absolute_path(child_dir='model', parent_dir='cifar100')