"""Function to create JSON databases"""
import os
import glob
import json
import time
import numpy as np

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.utils.transform_annotations import AnnotationTransformer
from carrada_dataset.utils.generate_annotations import AnnotationGenerator

RA_SHAPE = (256, 256)
RD_SHAPE = (256, 64)

def get_instance_oriented(sequences, instance_exceptions, carrada_path, write_results=True):
    """
    Function to generate annotation file oriented by instance
    For each sequence, the keys are the observed instances.
    For each instance, keys are the frames in which it appears (with annotations)

    PARAMETERS
    ----------
    sequences: list of str
        Names of the sequences to process
    instance exceptions: dict
        Manage instance merging
    carrada_path: str
        Path to Carrada dataset

    RETURNS
    -------
    annotations: dict
        Formated annotations
    """

    with open(os.path.join(carrada_path, 'data_seq_ref.json'), 'r') as fp:
        data_seq_ref = json.load(fp)
    save_path = os.path.join(carrada_path, 'annotations_instance_oriented.json')
    annotations = dict()
    for sequence in sequences:
        print('***** Processing sequence: {} *****'.format(sequence))
        annotations[sequence] = dict()
        raw_annotations_paths = os.path.join(carrada_path, sequence,
                                             'centroid_tracking_cluster_jensen_shannon.json')
        try:
            with open(raw_annotations_paths, 'r') as fp:
                raw_annotations = json.load(fp)
        except FileNotFoundError:
            print('Annotations have not been generated for sequence: {}.'.format(sequence))
            continue
        for instance in raw_annotations.keys():
            if sequence in instance_exceptions.keys() and \
               instance in instance_exceptions[sequence].keys():
                clean_instance = instance_exceptions[sequence][instance]
            else:
                clean_instance = instance

            if clean_instance not in annotations[sequence].keys():
                annotations[sequence][clean_instance] = dict()

            
            label_index = data_seq_ref[sequence]['instances'].index(instance)
            label = data_seq_ref[sequence]['labels'][label_index]

            all_frames = list(raw_annotations[instance].keys())
            all_frames.sort()
            all_frames = all_frames[1:-1]
            for frame in all_frames:
                annotations[sequence][clean_instance][frame] = dict()
                raw_points = raw_annotations[instance][frame]
                for signal_type in ['range_doppler', 'range_angle']:
                    annotations[sequence][clean_instance][frame][signal_type] = dict()
                    if signal_type == 'range_doppler':
                        annots = AnnotationTransformer(raw_points, RD_SHAPE)
                        annot_generator = AnnotationGenerator(RD_SHAPE, annots.to_rd())
                    elif signal_type == 'range_angle':
                        annots = AnnotationTransformer(raw_points, RA_SHAPE)
                        annot_generator = AnnotationGenerator(RA_SHAPE, annots.to_ra())
                    else:
                        raise TypeError('Signal type {} not supported'.format(signal_type))
                    points = annot_generator.get_points().tolist()
                    box = annot_generator.get_box()
                    box = [[int(coord[0]), int(coord[1])] for coord in box]
                    mask = annot_generator.get_mask()
                    mask_coords = np.where(mask == True)
                    mask = [[int(x), int(y)] for x, y in zip(mask_coords[0], mask_coords[1])]
                    annotations[sequence][clean_instance][frame][signal_type]['sparse'] = points
                    annotations[sequence][clean_instance][frame][signal_type]['box'] = box
                    annotations[sequence][clean_instance][frame][signal_type]['dense'] = mask
                    annotations[sequence][clean_instance][frame][signal_type]['label'] = label
    if write_results:
        with open(save_path, 'w') as fp:
            json.dump(annotations, fp)
    return annotations

def get_frame_oriented(sequences, instance_exceptions, carrada_path, write_results=True):
    """
    Function to generate annotation file oriented by frame.
    For each sequence, each frame has a dict.
    If the frame has annotations, it will have an instance key with all the annotations.

    PARAMETERS
    ----------
    sequences: list of str
        Names of the sequences to process
    instance exceptions: dict
        Manage instance merging
    carrada_path: str
        Path to Carrada dataset

    RETURNS
    -------
    annotations: dict
        Formated annotations
    """

    with open(os.path.join(carrada_path, 'data_seq_ref.json'), 'r') as fp:
        data_seq_ref = json.load(fp)
    save_path = os.path.join(carrada_path, 'annotations_frame_oriented.json')
    annotations = dict()
    for sequence in sequences:
        print('*** Processing sequence: {} ***'.format(sequence))
        annotations[sequence] = dict()
        raw_annotations_paths = os.path.join(carrada_path, sequence,
                                             'centroid_tracking_cluster_jensen_shannon.json')
        try:
            with open(raw_annotations_paths, 'r') as fp:
                raw_annotations = json.load(fp)
        except FileNotFoundError:
            print('Annotations have not been generated for sequence: {}.'.format(sequence))
            continue

        frame_ids = glob.glob(os.path.join(carrada_path, sequence, 'range_doppler_numpy',
                                           '*.npy'))
        frame_ids.sort()
        frames_ids = [frame.split('/')[-1].split('.')[0] for frame in frame_ids]
        for frame in frames_ids:
            annotations[sequence][frame] = dict()
            for instance in raw_annotations.keys():
                label_index = data_seq_ref[sequence]['instances'].index(instance)
                label = data_seq_ref[sequence]['labels'][label_index]
                all_frames = list(raw_annotations[instance].keys())
                all_frames.sort()
                all_frames = all_frames[1:-1]
                if frame in all_frames:
                    raw_points = raw_annotations[instance][frame]
                    if sequence in instance_exceptions.keys() and \
                       instance in instance_exceptions[sequence].keys():
                        clean_instance = instance_exceptions[sequence][instance]
                    else:
                        clean_instance = instance
                    annotations[sequence][frame][clean_instance] = dict()
                    for signal_type in ['range_doppler', 'range_angle']:
                        annotations[sequence][frame][clean_instance][signal_type] = dict()
                        if signal_type == 'range_doppler':
                            annots = AnnotationTransformer(raw_points, RD_SHAPE)
                            annot_generator = AnnotationGenerator(RD_SHAPE, annots.to_rd())
                        elif signal_type == 'range_angle':
                            annots = AnnotationTransformer(raw_points, RA_SHAPE)
                            annot_generator = AnnotationGenerator(RA_SHAPE, annots.to_ra())
                        else:
                            raise TypeError('Signal type {} not supported'.format(signal_type))
                        points = annot_generator.get_points().tolist()
                        box = annot_generator.get_box()
                        box = [[int(coord[0]), int(coord[1])] for coord in box]
                        mask = annot_generator.get_mask()
                        mask_coords = np.where(mask == True)
                        mask = [[int(x), int(y)] for x, y in zip(mask_coords[0], mask_coords[1])]
                        annotations[sequence][frame][clean_instance][signal_type]['sparse'] = points
                        annotations[sequence][frame][clean_instance][signal_type]['box'] = box
                        annotations[sequence][frame][clean_instance][signal_type]['dense'] = mask
                        annotations[sequence][frame][clean_instance][signal_type]['label'] = label
    if write_results:
        with open(save_path, 'w') as fp:
            json.dump(annotations, fp)
    return annotations


def main():
    print('***** Step 4/4: Generate Annotation Files *****')
    time1 = time.time()
    config_path = os.path.join(CARRADA_HOME, 'config.ini')
    config = Configurable(config_path).config
    warehouse = config['data']['warehouse']
    carrada = os.path.join(warehouse, 'Carrada')
    with open(os.path.join(carrada, 'validated_seqs.txt')) as fp:
        sequences = fp.readlines()
    with open(os.path.join(carrada, 'instance_exceptions.json'), 'r') as fp:
        instance_exceptions = json.load(fp)
    sequences = [seq.replace('\n', '') for seq in sequences]
    annotations_io = get_instance_oriented(sequences, instance_exceptions, carrada)
    annotations_fo = get_frame_oriented(sequences, instance_exceptions, carrada)
    print('***** Execution Time for Step 4/4:'
          ' {} secs. *****'.format(round(time.time() - time1, 2)))


if __name__ == '__main__':
    main()
