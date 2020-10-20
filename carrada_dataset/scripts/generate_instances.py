"""Script to generate instances for all sequences"""
import os
import time

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.annotation_generators.instance_generator import InstanceGenerator


def main():
    print('***** Step 1/4: Generate instances *****')
    time1 = time.time()
    config_path = os.path.join(CARRADA_HOME, 'config.ini')
    config = Configurable(config_path).config
    warehouse = config['data']['warehouse']
    carrada = os.path.join(warehouse, 'Carrada')
    with open(os.path.join(carrada, 'validated_seqs.txt')) as fp:
        sequence_names = fp.readlines()
    sequence_names = [seq.replace('\n', '') for seq in sequence_names]
    for seq_name in sequence_names:
        InstanceGenerator().process_sequence(seq_name, 0, save_boxes=False,
                                             save_masks=False, save_labels=True,
                                             save_instances_masks=False)
    print('***** Execution Time for Step 1/4:'
          ' {} secs. *****'.format(round(time.time() - time1, 2)))


if __name__ == '__main__':
    main()
