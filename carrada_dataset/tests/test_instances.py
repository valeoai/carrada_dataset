"""Test for instance generation.
Note that the instance names are not the same since we do not
run the entire pipeline.
"""

import os
import json
import glob
import pytest
import torchvision

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.utils.sort import Sort
from carrada_dataset.annotation_generators.instance_generator import InstanceGenerator, ImageInstances

CONFIG_PATH = os.path.join(CARRADA_HOME, 'config.ini')
CONFIG = Configurable(CONFIG_PATH).config
CARRADA = os.path.join(CONFIG['data']['warehouse'], 'Carrada')

def get_img_points(seq_name, val_range):
    real_data = dict()
    with open(os.path.join(CARRADA, seq_name, 'points.json'), 'r') as fp:
        real_points = json.load(fp)
    sequence_path = os.path.join(CARRADA, seq_name)
    img_paths = sorted(glob.glob(os.path.join(sequence_path, 'camera_images', '*.jpg')))
    for img_path in img_paths[val_range[0]: val_range[1]]:
        img_name = img_path.split('/')[-1].split('.')[0]
        real_data[img_name] = real_points[seq_name][img_name]
    return real_data

def test_image_instances1():
    seq_name = '2020-02-28-13-09-58'
    val_range = (72, 80)
    true_data = get_img_points(seq_name, val_range)
    instance_generator = InstanceGenerator()
    paths = instance_generator.get_paths()
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    seq_mot_tracker1 = Sort()
    sequence_path = os.path.join(CARRADA, seq_name)
    img_paths = sorted(glob.glob(os.path.join(sequence_path, 'camera_images', '*.jpg')))
    n_random_points = 0
    save_instances_masks = False
    # for img_path in img_paths[73:74]:
    for img_path in img_paths[val_range[0]: val_range[1]]:
        img_name = img_path.split('/')[-1].split('.')[0]
        image1 = ImageInstances(img_path, model, paths, seq_mot_tracker1, n_random_points,
                                save_instances_masks)
        image1.update_instances()
        predicted_points = image1.get_points()[img_name]
        if predicted_points != {}:
            assert list(predicted_points['000001'][0]) == true_data[img_name]['001114'][0]

def test_image_instances2():
    seq_name = '2019-09-16-13-23-22'
    val_range = (137, 140)
    true_data = get_img_points(seq_name, val_range)
    instance_generator = InstanceGenerator()
    paths = instance_generator.get_paths()
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    seq_mot_tracker2 = Sort()
    sequence_path = os.path.join(CARRADA, seq_name)
    img_paths = sorted(glob.glob(os.path.join(sequence_path, 'camera_images', '*.jpg')))
    n_random_points = 0
    save_instances_masks = False
    for img_path in img_paths[val_range[0]: val_range[1]]:
        img_name = img_path.split('/')[-1].split('.')[0]
        image2 = ImageInstances(img_path, model, paths, seq_mot_tracker2, n_random_points,
                                save_instances_masks)
        image2.update_instances()
        predicted_points = image2.get_points()[img_name]
        if predicted_points != {}:
            assert list(predicted_points['000003'][0]) == true_data[img_name]['000709'][0]

def test_image_instances3():
    seq_name = '2019-09-16-13-20-20'
    val_range = (200, 206)
    true_data = get_img_points(seq_name, val_range)
    instance_generator = InstanceGenerator()
    paths = instance_generator.get_paths()
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    seq_mot_tracker3 = Sort()
    sequence_path = os.path.join(CARRADA, seq_name)
    img_paths = sorted(glob.glob(os.path.join(sequence_path, 'camera_images', '*.jpg')))
    n_random_points = 0
    save_instances_masks = False
    for img_path in img_paths[val_range[0]: val_range[1]]:
        img_name = img_path.split('/')[-1].split('.')[0]
        image2 = ImageInstances(img_path, model, paths, seq_mot_tracker3, n_random_points,
                                save_instances_masks)
        image2.update_instances()
        predicted_points = image2.get_points()[img_name]
        if predicted_points != {}:
            assert list(predicted_points['000007'][0]) == true_data[img_name]['000673'][0]

            
if __name__ == '__main__':
    test_image_instances1()
    test_image_instances2()
    test_image_instances3()
