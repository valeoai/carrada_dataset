"""Script to generate centroid for all sequences"""
import os
import json
import time

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.annotation_generators.centroid_tracking import CentroidTracking


def main():
    print('***** Step 3/4: Generate Centroids *****')
    time1 = time.time()
    config_path = os.path.join(CARRADA_HOME, 'config.ini')
    config = Configurable(config_path).config
    warehouse = config['data']['warehouse']
    carrada = os.path.join(warehouse, 'Carrada')
    with open(os.path.join(carrada, 'data_seq_ref.json'), 'r') as fp:
        ref_data = json.load(fp)
    with open(os.path.join(carrada, 'validated_seqs.txt')) as fp:
        seq_names = fp.readlines()
    seq_names = [seq.replace('\n', '') for seq in seq_names]
    data_types = ['cluster']
    for seq_name in seq_names:
        print('*** Processing sequence {} ***'.format(seq_name))
        ref_ids = ref_data[seq_name]['ref_frame']
        instances = ref_data[seq_name]['instances']
        min_frame_boundaries = ref_data[seq_name]['min_frame_boundaries']
        max_frame_boundaries = ref_data[seq_name]['max_frame_boundaries']
        labels = ref_data[seq_name]['labels']
        for data_type in data_types:
            print('===> Generating {} annotations with Jensen-Shannon strategy'.format(data_type))
            CentroidTracking(seq_name, instances, ref_ids, labels, min_frame_boundaries,
                             max_frame_boundaries).create_annotations(save_annotations=True,
                                                                      save_vis_rd=False,
                                                                      save_vis_doa=False,
                                                                      data_type=data_type)
    print('***** Execution Time for Step 3/4:'
          ' {} secs. *****'.format(round(time.time() - time1, 2)))


if __name__ == '__main__':
    main()
