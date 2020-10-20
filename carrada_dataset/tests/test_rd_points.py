"""Test RdPointsGenerator class"""
import os
import json
import pytest
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.annotation_generators.rd_points_generator import RDPointsGenerator

CONFIG_PATH = '../config.ini'
CONFIG = Configurable(CONFIG_PATH).config
CARRADA = os.path.join(CONFIG['data']['warehouse'], 'Carrada')
SEQ_NAMES = [['2020-02-28-13-08-51'], ['2019-09-16-13-20-20'], ['2020-02-28-13-14-35']]

@pytest.fixture
def get_true_data(request):
    with open(os.path.join(CARRADA, request.param, 'rd_points.json'), 'r') as fp:
        rd_points = json.load(fp)
    return rd_points

@pytest.mark.parametrize('get_true_data', SEQ_NAMES[0], indirect=True)
def test_rd_points_generator0(get_true_data):
    with open(os.path.join(CARRADA, 'data_seq_ref.json'), 'r') as fp:
        ref_data = json.load(fp)
    instances = ref_data[SEQ_NAMES[0][0]]['instances']
    n_points = 1
    time_window = 10
    generator = RDPointsGenerator(SEQ_NAMES[0][0], n_points, instances, time_window)
    test_rd_points, _ = generator.get_rd_points(save_rd_imgs=False,
                                                save_points=False,
                                                save_points_coordinates=False,
                                                save_world_points=False)
    assert test_rd_points == get_true_data

@pytest.mark.parametrize('get_true_data', SEQ_NAMES[1], indirect=True)
def test_rd_points_generator1(get_true_data):
    with open(os.path.join(CARRADA, 'data_seq_ref.json'), 'r') as fp:
        ref_data = json.load(fp)
    instances = ref_data[SEQ_NAMES[1][0]]['instances']
    n_points = 1
    time_window = 10
    generator = RDPointsGenerator(SEQ_NAMES[1][0], n_points, instances, time_window)
    test_rd_points, _ = generator.get_rd_points(save_rd_imgs=False,
                                                save_points=False,
                                                save_points_coordinates=False,
                                                save_world_points=False)
    assert test_rd_points == get_true_data

@pytest.mark.parametrize('get_true_data', SEQ_NAMES[2], indirect=True)
def test_rd_points_generator2(get_true_data):
    with open(os.path.join(CARRADA, 'data_seq_ref.json'), 'r') as fp:
        ref_data = json.load(fp)
    instances = ref_data[SEQ_NAMES[2][0]]['instances']
    n_points = 1
    time_window = 10
    generator = RDPointsGenerator(SEQ_NAMES[2][0], n_points, instances, time_window)
    test_rd_points, _ = generator.get_rd_points(save_rd_imgs=False,
                                                save_points=False,
                                                save_points_coordinates=False,
                                                save_world_points=False)
    assert test_rd_points == get_true_data
