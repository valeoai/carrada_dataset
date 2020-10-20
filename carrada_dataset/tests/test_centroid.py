"""Script to generate centroid for all sequences"""
import os
import json
import pytest

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.annotation_generators.centroid_tracking import CentroidTracking

CONFIG_PATH = os.path.join(CARRADA_HOME, 'config.ini')
CONFIG = Configurable(CONFIG_PATH).config
WAREHOUSE = CONFIG['data']['warehouse']
CARRADA = os.path.join(WAREHOUSE, 'Carrada')
SEQUENCE = '2020-02-28-13-09-58'

@pytest.fixture
def get_true_data():
    data_path = os.path.join(CARRADA, SEQUENCE, 'centroid_tracking_cluster_jensen_shannon.json')
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    with open(os.path.join(CARRADA, 'data_seq_ref.json'), 'r') as fp:
        ref_data = json.load(fp)
    instance = ref_data[SEQUENCE]['instances'][0]
    return data[instance]

def test_centroid(get_true_data):
    with open(os.path.join(CARRADA, 'data_seq_ref.json'), 'r') as fp:
        ref_data = json.load(fp)
    # Only for one instance
    ref_ids = [ref_data[SEQUENCE]['ref_frame'][0]]
    instances = [ref_data[SEQUENCE]['instances'][0]]
    min_frame_boundaries = ref_data[SEQUENCE]['min_frame_boundaries']
    max_frame_boundaries = ref_data[SEQUENCE]['max_frame_boundaries']
    labels = ref_data[SEQUENCE]['labels']
    tracker = CentroidTracking(SEQUENCE, instances, ref_ids, labels, min_frame_boundaries,
                               max_frame_boundaries)
    computed_data = tracker.create_annotations(save_annotations=False,
                                               save_vis_rd=False,
                                               save_vis_doa=False,
                                               data_type='cluster')
    return computed_data[instances[0]] == get_true_data
