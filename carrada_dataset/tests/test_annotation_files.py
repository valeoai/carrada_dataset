"""Test function to format data """
import os
import json
import pytest

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.scripts.generate_annotation_files import get_instance_oriented, get_frame_oriented

CONFIG_PATH = os.path.join(CARRADA_HOME, 'config.ini')
CONFIG = Configurable(CONFIG_PATH).config
WAREHOUSE = CONFIG['data']['warehouse']
CARRADA = os.path.join(WAREHOUSE, 'Carrada')
SEQ_NAMES = [['2020-02-28-13-14-35'], ['2020-02-28-13-08-51'], ['2019-09-16-13-20-20']]


@pytest.fixture
def get_true_data_instance(request):
    with open(os.path.join(CARRADA, 'annotations_instance_oriented.json'), 'r') as fp:
        instance_annots = json.load(fp)
    return instance_annots[request.param]

@pytest.fixture
def get_true_data_frame(request):
    with open(os.path.join(CARRADA, 'annotations_frame_oriented.json'), 'r') as fp:
        instance_annots = json.load(fp)
    return instance_annots[request.param]

@pytest.mark.parametrize('get_true_data_instance', SEQ_NAMES[0], indirect=True)
def test_get_instance_oriented0(get_true_data_instance):
    with open(os.path.join(CARRADA, 'instance_exceptions.json'), 'r') as fp:
        instance_exceptions = json.load(fp)
    annotations_io = get_instance_oriented(SEQ_NAMES[0], instance_exceptions,
                                           CARRADA, write_results=False)
    assert annotations_io[SEQ_NAMES[0][0]] == get_true_data_instance

@pytest.mark.parametrize('get_true_data_frame', SEQ_NAMES[0], indirect=True)
def test_get_frame_oriented0(get_true_data_frame):
    with open(os.path.join(CARRADA, 'instance_exceptions.json'), 'r') as fp:
        instance_exceptions = json.load(fp)
    annotations_fo = get_frame_oriented(SEQ_NAMES[0], instance_exceptions,
                                        CARRADA, write_results=False)
    assert annotations_fo[SEQ_NAMES[0][0]] == get_true_data_frame

@pytest.mark.parametrize('get_true_data_instance', SEQ_NAMES[1], indirect=True)
def test_get_instance_oriented1(get_true_data_instance):
    with open(os.path.join(CARRADA, 'instance_exceptions.json'), 'r') as fp:
        instance_exceptions = json.load(fp)
    annotations_io = get_instance_oriented(SEQ_NAMES[1], instance_exceptions,
                                           CARRADA, write_results=False)
    assert annotations_io[SEQ_NAMES[1][0]] == get_true_data_instance

@pytest.mark.parametrize('get_true_data_frame', SEQ_NAMES[1], indirect=True)
def test_get_frame_oriented1(get_true_data_frame):
    with open(os.path.join(CARRADA, 'instance_exceptions.json'), 'r') as fp:
        instance_exceptions = json.load(fp)
    annotations_fo = get_frame_oriented(SEQ_NAMES[1], instance_exceptions,
                                        CARRADA, write_results=False)
    assert annotations_fo[SEQ_NAMES[1][0]] == get_true_data_frame

@pytest.mark.parametrize('get_true_data_instance', SEQ_NAMES[2], indirect=True)
def test_get_instance_oriented2(get_true_data_instance):
    with open(os.path.join(CARRADA, 'instance_exceptions.json'), 'r') as fp:
        instance_exceptions = json.load(fp)
    annotations_io = get_instance_oriented(SEQ_NAMES[2], instance_exceptions,
                                           CARRADA, write_results=False)
    assert annotations_io[SEQ_NAMES[2][0]] == get_true_data_instance

@pytest.mark.parametrize('get_true_data_frame', SEQ_NAMES[2], indirect=True)
def test_get_frame_oriented2(get_true_data_frame):
    with open(os.path.join(CARRADA, 'instance_exceptions.json'), 'r') as fp:
        instance_exceptions = json.load(fp)
    annotations_fo = get_frame_oriented(SEQ_NAMES[2], instance_exceptions,
                                        CARRADA, write_results=False)
    assert annotations_fo[SEQ_NAMES[2][0]] == get_true_data_frame
