"""Script to detect and track instances"""
import os
import glob
import json
import numpy as np
import PIL
from scipy import ndimage
import torchvision

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.sort import Sort
from carrada_dataset.utils.configurable import Configurable
from .utils import get_random_rgb, threshold_mask


class InstanceGenerator(Configurable):

    """
    Class to generate tracked instances in the images
    """

    def __init__(self):
        self.config_path = os.path.join(CARRADA_HOME, 'config.ini')
        super().__init__(self.config_path)
        self.paths = self._get_paths()
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.points = None # Will be set after processing

    def _get_paths(self):
        """Collect usefull paths"""
        paths = dict()
        warehouse = self.config['data']['warehouse']
        paths['carrada'] = os.path.join(warehouse, 'Carrada')
        return paths

    def _process_images(self, seq_name, n_random_points=False, save_points=True, save_boxes=False,
                        save_masks=False, save_labels=False, save_instances_masks=True):
        """Process the entire sequence to segment images and save projected centroid"""
        sequence_path = os.path.join(self.paths['carrada'], seq_name)
        seq_mot_tracker = Sort()
        points = dict()
        boxes = dict()
        masks = dict()
        labels = dict()
        points[seq_name] = dict()
        boxes[seq_name] = dict()
        masks[seq_name] = dict()
        labels[seq_name] = dict()
        img_paths = sorted(glob.glob(os.path.join(sequence_path, 'camera_images', '*.jpg')))
        print('*** Processing Sequence:  %s ***' % seq_name)
        for img_path in img_paths:
            image = ImageInstances(img_path, self.model, self.paths,
                                   seq_mot_tracker, n_random_points,
                                   save_instances_masks)
            image.update_instances()
            if isinstance(n_random_points, int):
                points[seq_name].update(image.get_points())
            if save_boxes:
                boxes[seq_name].update(image.get_boxes())
            if save_masks:
                masks[seq_name].update(image.get_masks())
            if save_labels:
                labels[seq_name].update(image.get_labels())
        self._set_points(points)
        if save_points:
            self._save_data(points, seq_name, 'points.json')
        if save_boxes:
            self._save_data(boxes, seq_name, 'boxes.json')
        if save_masks:
            self._save_data(masks, seq_name, 'masks.json')
        if save_labels:
            self._save_data(labels, seq_name, 'labels.json')

    def _save_data(self, data, seq_name, name_file):
        """Save data stored in dict"""
        directory_path = os.path.join(self.paths['carrada'], seq_name)
        os.makedirs(directory_path, exist_ok=True)
        path = os.path.join(directory_path, name_file)
        with open(path, 'w') as fp:
            json.dump(data, fp)

    def _set_points(self, points):
        """Set points attributes (projected barycenters)"""
        self.points = points

    def process_sequence(self, seq_name, n_points=False, save_points=True, save_boxes=False,
                         save_masks=False, save_labels=False, save_instances_masks=True):
        """
        Method to process a sequence. It proposes to save computed measures.

        PARAMETERS
        ----------
        seq_name: str
            Name of the sequence to process
        n_points: int
            Number of points to save per frame in the segmented object. Default: False.
        save_points: bool
            Save projected points considered as reference. Default: True.
        save_boxes: bool
        save_masks: bool
        save_labels: bool
        save_instances_masks: bool
            Save predicted masks as images (colors corresponding to tracked objects)
        """
        self._process_images(seq_name, n_points, save_points, save_boxes,
                             save_masks, save_labels, save_instances_masks)

    def get_paths(self):
        """Get usefull paths"""
        return self.paths

    def get_points(self):
        """Get dict of projected points"""
        return self.points


class ImageInstances:

    """
    Class to detect and track instances for a given image

    PARAMETERS
    ----------
    path_to_image: str
    model: torch vision model (MaskRCNN)
        Model to predict boxes and semgentation maps
    paths: dict
        Usefull paths
    mot_tracker: Sort object
        Object to track the predicted instances
    n_random_points: bool or int
        Number of reference points to keep in segmented instance
    save_instances_masks: bool
        Save predicted masks as images (colors corresponding to tracked objects)
    """

    def __init__(self, path_to_image, model, paths, mot_tracker, n_random_points=False,
                 save_instances_masks=True):
        self.path_to_image = path_to_image
        self.model = model
        self.paths = paths
        self.mot_tracker = mot_tracker
        self.n_random_points = n_random_points
        self.save_instances_masks = save_instances_masks
        self.seq_name = self.path_to_image.split('/')[-3]
        self.img_name = self.path_to_image.split('/')[-1].split('.')[0]
        self.points = dict()
        self.structured_boxes = dict()
        self.structured_masks = dict()
        self.structured_labels = dict()
        self.points[self.img_name] = dict()
        self.structured_boxes[self.img_name] = dict()
        self.structured_masks[self.img_name] = dict()
        self.structured_labels[self.img_name] = dict()
        self.image = PIL.Image.open(self.path_to_image)
        self.masks, self.boxes, self.labels, self.ids = self._get_predictions()
        self.masks, self.boxes, self.labels, self.ids = self._select_labels()

    def _get_predictions(self):
        """Use Mask RCNN to detect and segment objects in an image.
        Update SORT to track the detected objects.
        """
        image_tensor = torchvision.transforms.functional.to_tensor(self.image)
        output = self.model([image_tensor])[0]
        masks = output['masks'].detach().numpy()
        labels = output['labels'].detach().numpy()
        boxes = output['boxes'].detach().numpy()
        scores = output['scores'].detach().numpy()
        selected_ids = np.where(scores > 0.5)
        boxes = boxes[selected_ids]
        masks = masks[selected_ids]
        labels = labels[selected_ids]
        scores = scores[selected_ids]
        boxes_scores = np.concatenate((boxes, np.reshape(scores, (-1, 1))), axis=1)
        _, tracking_indexes = self.mot_tracker.update(boxes_scores)
        if tracking_indexes.shape[0] == 0:
            ids = np.empty(0)
            masks = np.empty(0)
            boxes = np.empty(0)
            labels = np.empty(0)
        else:
            ids = np.array(tracking_indexes['trk_ids'])
            masks = masks[tracking_indexes['det_ids']]
            boxes = boxes[tracking_indexes['det_ids']]
            labels = labels[tracking_indexes['det_ids']]
        return [masks, boxes, labels, ids]

    def _select_labels(self):
        """Mask RCNN is pre trained on COCO (80 classes)
        This method selects and merges labels"""
        # pedestrian = [1, 26, 27, 28, 29, 30, 31, 32, 33]
        useless_idx = list()
        vehicles = [3, 4, 5, 6, 7, 8, 9]
        for i in range(self.labels.shape[0]):
            if self.labels[i] in vehicles:
                self.labels[i] = 3
            if self.labels[i] not in [1, 2, 3]:
                useless_idx.append(i)
        if self.labels.shape[0] == 0:
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        selected_masks = np.delete(self.masks, useless_idx, axis=0)
        selected_boxes = np.delete(self.boxes, useless_idx, axis=0)
        selected_labels = np.delete(self.labels, useless_idx)
        selected_ids = np.delete(self.ids, useless_idx)
        return selected_masks, selected_boxes, selected_labels, selected_ids

    def update_instances(self, get_rgb_masks=False):
        """Get annotations (points, boxes, masks, labels) from model predictions using Instance.

        PARAMETERS
        ----------
        get_rgb_masks: bool
            If True, return the predicted RGB masks for the current image. Default: False.
        """
        rgb_masks = list()
        if self.save_instances_masks:
            instances_path = os.path.join(self.paths['carrada'], self.seq_name, 'instances')
            os.makedirs(instances_path, exist_ok=True)
        for i in range(self.labels.shape[0]):
            instance_name = str(int(self.ids[i])).zfill(6)
            instance = Instance(instance_name, self.masks[i], self.boxes[i], self.labels[i])
            if self.save_instances_masks:
                path = os.path.join(instances_path, self.img_name)
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, instance_name + '.png')
                instance.save(path)
            rgb_masks.append(instance.get_rgb_mask)
            if isinstance(self.n_random_points, int):
                self.points[self.img_name].update(instance.get_points(self.n_random_points))
            self.structured_boxes[self.img_name].update(instance.get_structured_box())
            self.structured_masks[self.img_name].update(instance.get_structured_mask())
            self.structured_labels[self.img_name].update(instance.get_structured_label())
        if get_rgb_masks:
            if rgb_masks:
                return np.stack(rgb_masks, axis=0)
            return np.empty(0)

    def get_name(self):
        """Get image (frame) name."""
        return self.img_name

    def get_predictions(self):
        """Get predictions: masks, boxes, labels and IDs"""
        return [self.masks, self.boxes, self.labels, self.ids]

    def get_points(self):
        """Get projected barycenter of the predicted masks"""
        return self.points

    def get_boxes(self):
        """Get predicted boxes in dict"""
        return self.structured_boxes

    def get_masks(self):
        """Get predicted boxes in dict"""
        return self.structured_masks

    def get_labels(self):
        """Get predicted labels in dict"""
        return self.structured_labels


class Instance():

    """
    Class to characterize an instance

    PARAMETERS
    ----------
    instance_name: str
    mask: numpy array
    box: numpy array
    label: numpy array
    """

    def __init__(self, instance_name, mask, box, label):
        self.instance_name = instance_name
        self.mask = mask
        self.box = box
        self.label = label
        self.rgb_mask = self._get_rgb_mask()

    def _get_rgb_mask(self):
        """Transform mask to RGB"""
        r = np.zeros_like(self.mask).astype(np.uint8)
        g = np.zeros_like(self.mask).astype(np.uint8)
        b = np.zeros_like(self.mask).astype(np.uint8)
        label_colors = get_random_rgb(self.instance_name)
        mask = threshold_mask(self.mask)
        idx = mask == 1
        r[idx] = label_colors[0]
        g[idx] = label_colors[1]
        b[idx] = label_colors[2]
        return np.concatenate([r, g, b])

    def _get_barycenter(self):
        """Compute barycenter of the mask"""
        values = ndimage.measurements.center_of_mass(self.mask.squeeze(0))
        values = [(int(values[0]), int(values[1]))]
        return values

    def _get_random_points(self, n_random_points):
        """Get random points in the mask if necessary"""
        np.random.seed(42)
        if not isinstance(n_random_points, int):
            raise TypeError('Number of points needs to be specified as int.')
        mask_points = np.array(np.nonzero(self.mask)).T
        random_points = mask_points[np.random.choice(mask_points.shape[0], n_random_points,
                                                     replace=False)]
        random_points = [(int(points[1]), int(points[2])) for points in random_points]
        return random_points

    def _get_bottom_coordinate(self):
        """Poject points on the ground (top down pixel of the mask).
        It requires a high quality segmentation.
        """
        mask_points = np.array(np.nonzero(self.mask)).T
        mask_points = mask_points[mask_points[:, 1].argsort()[::-1]]
        bottom_coordinate = mask_points[0][1]
        return bottom_coordinate

    def _get_projected_points(self, n_random_points):
        """Compute projection of the points (barycenter + random points) on the ground"""
        bottom_coordinate = self._get_bottom_coordinate()
        barycenter = self._get_barycenter()
        random_points = self._get_random_points(n_random_points)
        targeted_points = barycenter + random_points
        targeted_points = [(int(bottom_coordinate), int(point[1])) for point in targeted_points]
        return targeted_points

    def get_points(self, n_random_points):
        """Get projected barycenter or points (if several)"""
        projected_points = dict()
        projected_points[self.instance_name] = self._get_projected_points(n_random_points)
        return projected_points

    def get_structured_box(self):
        """Get predicted box"""
        structured_box = dict()
        structured_box[self.instance_name] = self.box.tolist()
        return structured_box

    def get_structured_mask(self):
        """Get predicted mask"""
        structured_mask = dict()
        structured_mask[self.instance_name] = self.mask.tolist()
        return structured_mask

    def get_structured_label(self):
        """Get predicted label"""
        structured_labels = dict()
        structured_labels[self.instance_name] = self.label.tolist()
        return structured_labels

    def get_rgb_mask(self):
        """Get RGB mask"""
        return self.rgb_mask

    def save(self, path_to_save):
        """Save RGB mask as image"""
        img = PIL.Image.fromarray(self.rgb_mask.transpose((1, 2, 0)))
        img.save(path_to_save)
