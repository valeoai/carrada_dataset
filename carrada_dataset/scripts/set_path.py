"""Script to set the path to CARRADA in the config.ini file"""
import os
import sys
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.utils import CARRADA_HOME

if __name__ == '__main__':
    path_to_carrada = sys.argv[1]
    configurable = Configurable(os.path.join(CARRADA_HOME, 'config.ini'))
    configurable.set('data', 'warehouse', path_to_carrada)
    with open(os.path.join(CARRADA_HOME, 'config.ini'), 'w') as fp:
        configurable.config.write(fp)
