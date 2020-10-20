"""Script to test the DataTransformer class"""
import os
import pytest
import numpy as np

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.utils.transform_data import DataTransformer

CONFIG_PATH = os.path.join(CARRADA_HOME, 'config.ini')
CONFIG = Configurable(CONFIG_PATH).config
# CARRADA = download('Carrada', os.path.join(CONFIG['data']['warehouse'], 'Carrada'))
CARRADA = os.path.join(CONFIG['data']['warehouse'], 'Carrada')
SEQ_NAMES = [['2020-02-28-13-08-51'], ['2019-09-16-13-20-20'], ['2020-02-28-13-14-35']]
FRAME_NAME = '000100.npy'

@pytest.fixture
def get_true_ra(request):
    ra_matrix_path = os.path.join(CARRADA, request.param, 'range_angle_numpy', FRAME_NAME)
    ra_matrix = np.load(ra_matrix_path)
    return ra_matrix

@pytest.fixture
def get_true_rd(request):
    rd_matrix_path = os.path.join(CARRADA, request.param, 'range_doppler_numpy',
                                  FRAME_NAME)
    rd_matrix = np.load(rd_matrix_path)
    return rd_matrix

@pytest.mark.parametrize('get_true_ra', SEQ_NAMES[0], indirect=True)
def test_data_transfomer_ra0(get_true_ra):
    rad_matrix_path = os.path.join(CARRADA, SEQ_NAMES[0][0], 'RAD_numpy', FRAME_NAME)
    rad_matrix = np.load(rad_matrix_path)
    transformer = DataTransformer(rad_matrix)
    pred_ra = transformer.to_ra()
    assert np.array_equal(pred_ra, get_true_ra)

@pytest.mark.parametrize('get_true_rd', SEQ_NAMES[0], indirect=True)
def test_data_transfomer_rd0(get_true_rd):
    rad_matrix_path = os.path.join(CARRADA, SEQ_NAMES[0][0], 'RAD_numpy', FRAME_NAME)
    rad_matrix = np.load(rad_matrix_path)
    transformer = DataTransformer(rad_matrix)
    pred_rd = transformer.to_rd()
    assert np.allclose(pred_rd, get_true_rd)

@pytest.mark.parametrize('get_true_ra', SEQ_NAMES[1], indirect=True)
def test_data_transfomer_ra1(get_true_ra):
    rad_matrix_path = os.path.join(CARRADA, SEQ_NAMES[1][0], 'RAD_numpy', FRAME_NAME)
    rad_matrix = np.load(rad_matrix_path)
    transformer = DataTransformer(rad_matrix)
    pred_ra = transformer.to_ra()
    assert np.array_equal(pred_ra, get_true_ra)

@pytest.mark.parametrize('get_true_rd', SEQ_NAMES[1], indirect=True)
def test_data_transfomer_rd1(get_true_rd):
    rad_matrix_path = os.path.join(CARRADA, SEQ_NAMES[1][0], 'RAD_numpy', FRAME_NAME)
    rad_matrix = np.load(rad_matrix_path)
    transformer = DataTransformer(rad_matrix)
    pred_rd = transformer.to_rd()
    assert np.allclose(pred_rd, get_true_rd)

@pytest.mark.parametrize('get_true_ra', SEQ_NAMES[2], indirect=True)
def test_data_transfomer_ra2(get_true_ra):
    rad_matrix_path = os.path.join(CARRADA, SEQ_NAMES[2][0], 'RAD_numpy', FRAME_NAME)
    rad_matrix = np.load(rad_matrix_path)
    transformer = DataTransformer(rad_matrix)
    pred_ra = transformer.to_ra()
    assert np.array_equal(pred_ra, get_true_ra)

@pytest.mark.parametrize('get_true_rd', SEQ_NAMES[2], indirect=True)
def test_data_transfomer_rd2(get_true_rd):
    rad_matrix_path = os.path.join(CARRADA, SEQ_NAMES[2][0], 'RAD_numpy', FRAME_NAME)
    rad_matrix = np.load(rad_matrix_path)
    transformer = DataTransformer(rad_matrix)
    pred_rd = transformer.to_rd()
    assert np.allclose(pred_rd, get_true_rd)
