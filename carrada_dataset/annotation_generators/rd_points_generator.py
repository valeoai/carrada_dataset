"""Classes to generate RD points from projected points of images"""
import os
import json
import glob
import numpy as np
import pandas as pd

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.utils.transform_data import DataTransformer
from carrada_dataset.utils.visualize_signal import SignalVisualizer
from carrada_dataset.utils.camera import Camera
from .utils import compute_line_coefs, compute_orthogonal_proj

class RDPointsGenerator(Configurable):

    """
    Class to convert projected barycenter of instances in images
    into range-Doppler points

    PARAMETERS
    ----------
    seq_name: str
        Name of the processed sequence
    n_points: int
        Number of points to process
    instances: list of str
        List of instance names
    time_window: int
        Time window to compute the velocity of the instance
    """

    DOPPLER_RES = 0.41968030701528203
    RANGE_RES = 0.1953125

    def __init__(self, seq_name, n_points, instances, time_window=10):
        self.config_path = os.path.join(CARRADA_HOME, 'config.ini')
        super().__init__(self.config_path)
        self.cls = self.__class__
        self.seq_name = seq_name
        self.n_points = n_points
        self.instances = instances
        self.time_window = time_window
        self.paths = self._get_paths()
        with open(self.paths['points'], 'rb') as fp:
            self.points = json.load(fp)
        self.cam = Camera(self.paths['intrinsics'], self.paths['extrinsics'])
        self.timestamps = pd.read_csv(self.paths['timestamps'], header=None).values
        self.seq_len = self.timestamps.shape[0]
        if not isinstance(self.instances, list):
            raise TypeError('Instances should be in a list.')

    def _get_paths(self):
        """Get usefull paths"""
        year = self.seq_name.split('-')[0]
        paths = dict()
        paths['warehouse'] = self.config['data']['warehouse']
        paths['carrada'] = os.path.join(paths['warehouse'], 'Carrada')
        paths['cam_params'] = os.path.join(paths['carrada'], 'cam_params')
        paths['intrinsics'] = os.path.join(paths['cam_params'], 'intrinsics.xml')
        paths['extrinsics'] = os.path.join(paths['cam_params'], 'extrinsics_' + year + '.xml')
        paths['seq_data'] = os.path.join(paths['carrada'], self.seq_name)
        paths['points'] = os.path.join(paths['seq_data'], 'points.json')
        paths['rds'] = glob.glob(os.path.join(paths['seq_data'],
                                              'range_doppler_numpy', '*.npy'))
        paths['rds'].sort()
        paths['timestamps'] = os.path.join(paths['seq_data'], 'adc_ts_' + self.seq_name + '.txt')
        os.makedirs(paths['seq_data'], exist_ok=True)
        return paths

    def _get_rd_data(self, idx):
        """Load range-Doppler matrix, image points and timestamps"""
        rd_path = self.paths['rds'][idx]
        rd_id = rd_path.split('/')[-1].split('.')[0]
        rd_matrix = np.load(rd_path)
        rd_points = self.points[self.seq_name][rd_id]
        rd_ts = self.timestamps[idx]
        return rd_id, rd_matrix, rd_points, rd_ts

    def _compute_rd_points_coordinates(self, instance_points, doppler_shape):
        """Convert estimated distance and velocity to RD resolution"""
        rd_points_coordinates = dict()
        rd_points_coordinates['range'] = [int(instance_points['distance'][i]/ \
                                              self.cls.RANGE_RES)
                                          for i in range(self.n_points)]
        rd_points_coordinates['doppler'] = [int(doppler_shape/2 -1) +
                                            int(instance_points['velocity'][i]/ \
                                                self.cls.DOPPLER_RES)
                                            for i in range(self.n_points)]
        return rd_points_coordinates

    def _save_rd_point_images(self, rd_points_coordinates, rd_matrix, rd_id):
        """Save RD images with projected estimated points """
        annotated_rd_path = os.path.join(self.paths['seq_data'], 'rd_with_img_points')
        os.makedirs(annotated_rd_path, exist_ok=True)
        visualiser = SignalVisualizer(rd_matrix)
        instances = list(rd_points_coordinates[rd_id].keys())
        if instances:
            for idx, instance in enumerate(instances):
                rd_coord = list()
                for i in range(self.n_points):
                    range_coord = rd_points_coordinates[rd_id][instance]['range'][i]
                    doppler_coord = rd_points_coordinates[rd_id][instance]['doppler'][i]
                    rd_coord.append([range_coord, doppler_coord])
                rd_coord = np.array(rd_coord)
                visualiser.add_annotation(idx, rd_coord, 'sparse')
            visualiser.save_multiple_annotations(os.path.join(annotated_rd_path, rd_id + '.png'))

    def _save_points(self, data, name):
        """Save json file with points"""
        data_path = os.path.join(self.paths['seq_data'], name)
        with open(data_path, 'w') as fp:
            json.dump(data, fp)

    def get_rd_points(self, save_rd_imgs=False, save_points=False, save_points_coordinates=False,
                      save_world_points=False):
        """
        Method to estimate real world, RD points and RD point coordinates from image points.

        PARAMETERS
        ----------
        save_rd_imgs: bool
        save_points: bool
        save_points_coordinates: bool
        save_world_points: bool
            Save estimated real world coordinates of projected points

        RETURNS
        -------
        rd_points: dict
            Range-Doppler values for each instance in each frame
        rd_points_coordinates: dict
            Coordinates in range-Doppler for each instance in each frame
        """
        rd_points = dict()
        rd_points_coordinates = dict()
        world_points = dict()
        for i in range(self.time_window+1, self.seq_len):
            # if i % 10 == 0:
                # print('Processing step: {}/{}'.format(i, self.seq_len-self.time_window))
            try:
                _, _, rd_points_t1, rd_ts_t1 = self._get_rd_data(i-self.time_window)
                rd_id_t2, rd_matrix_t2, rd_points_t2, rd_ts_t2 = self._get_rd_data(i)
            except (KeyError, IndexError):
                break
            rd_points[rd_id_t2] = dict()
            rd_points_coordinates[rd_id_t2] = dict()
            world_points[rd_id_t2] = dict()
            delta_ts = (rd_ts_t2 - rd_ts_t1)/1000000000.
            for instance in self.instances:
                if (delta_ts == 0)[0]:
                    break
                try:
                    bottom_points_t2 = rd_points_t2[instance][:self.n_points]
                    instance_points_t2 = InstancePoints(self.cam, bottom_points_t2)
                    world_points[rd_id_t2][instance] = instance_points_t2.get_world_points()
                    bottom_points_t1 = rd_points_t1[instance][:self.n_points]
                    instance_points_t1 = InstancePoints(self.cam, bottom_points_t1)
                    instance_rd_points = InstanceRDPoints(delta_ts, instance_points_t1,
                                                          instance_points_t2)
                    rd_points[rd_id_t2][instance] = instance_rd_points.get_rd_points()
                    instance_points_coordinates = self._compute_rd_points_coordinates\
                                                  (instance_rd_points.get_rd_points(),
                                                   rd_matrix_t2.shape[1])
                    rd_points_coordinates[rd_id_t2][instance] = instance_points_coordinates
                except KeyError:
                    continue
            if save_rd_imgs:
                self._save_rd_point_images(rd_points_coordinates, rd_matrix_t2, rd_id_t2)
        if save_points:
            self._save_points(rd_points, 'rd_points.json')
        if save_points_coordinates:
            self._save_points(rd_points_coordinates, 'rd_points_coordinates.json')
        if save_world_points:
            self._save_points(world_points, 'world_points.json')
        return rd_points, rd_points_coordinates


class InstancePoints:

    """
    Class defining points in real world coordinates of an instance.

    PARAMETERS
    ----------
    cam: Camera object
        Object to convert image point in real world coordinates
    instance_points: list
        List of points in image domain
    """

    WORLD_CAMERA = (1.6654252468981425, -0.40027661590149677)

    def __init__(self, cam, instance_points):
        self.cls = __class__
        self.cam = cam
        self.instance_points = instance_points
        self.world_camera = self.cls.WORLD_CAMERA
        self.world_points = list()
        self._compute_world_points()

    def __len__(self):
        return len(self.world_points)

    def _compute_world_points(self):
        """Compute real world coordinates of the points in the image domain"""
        for point in self.instance_points:
            world_point = self.cam.imageToWorld_Z(point[1], point[0], 0)
            self.world_points.append(world_point)

    def get_world_points(self):
        """Method to get estimated real world coordinates"""
        return self.world_points


class InstanceRDPoints:

    """
    Class defining range-Doppler points of an instance

    PARAMETERS
    ----------
    delta_ts: int
        Variation of timestamp between two considered frames
    instance_points_t1: list
        List of real point coordinates at time t1
    instance_points_t2: list
        List of real point coordinates at time t2
    """

    def __init__(self, delta_ts, instance_points_t1, instance_points_t2):
        self.delta_ts = delta_ts
        self.instance_points_t1 = instance_points_t1
        self.instance_points_t2 = instance_points_t2
        self.rd_points = dict()
        self.rd_points['distance'] = list()
        self.rd_points['velocity'] = list()
        self._compute_rd_points()

    def _compute_rd_points(self):
        """Compute range and Doppler values of the points in the world domain """
        point_t1 = self.instance_points_t1.get_world_points()
        point_t2 = self.instance_points_t2.get_world_points()
        for i in range(len(self.instance_points_t1)):
            rd_point = RDPoint(point_t1[i], point_t2[i], self.delta_ts)
            self.rd_points['distance'].append(rd_point.get_distance())
            self.rd_points['velocity'].append(rd_point.get_velocity())

    def get_rd_points(self):
        """Get dictionary with range-Doppler values"""
        return self.rd_points


class RDPoint:

    """
    Class defining a single range-Doppler point

    PARAMETERS
    ----------
    point_t1: tuple
        Real point coordinates at time t1
    point_t2: tuple
        Real point coordinates at time t2
    delta_ts: int
        Variation of timestamp between two considered frames
    """

    WORLD_RADAR = (0., 0.)

    def __init__(self, point_t1, point_t2, delta_ts):
        self.cls = __class__
        self.point_t1 = point_t1
        self.point_t2 = point_t2
        self.delta_ts = delta_ts
        self.world_radar = self.cls.WORLD_RADAR
        self.distance_t1 = self._compute_l2_distance(self.point_t1)
        self.distance_t2 = self._compute_l2_distance(self.point_t2)

    def _compute_velocity(self):
        """Compute the radial velocity of the object wrt the radar"""
        radar_object_line = compute_line_coefs(self.point_t1, self.world_radar)
        projected_point = compute_orthogonal_proj(radar_object_line, self.point_t2)
        radial_velocity_vect = np.array([(projected_point[0] - self.point_t1[0])/self.delta_ts[0],
                                         (projected_point[1] - self.point_t1[1])/self.delta_ts[0]])
        radial_velocity_scalar = np.sqrt(radial_velocity_vect[0]**2 + \
                                         radial_velocity_vect[1]**2)
        if self.distance_t1 < self.distance_t2:
            return -radial_velocity_scalar
        return radial_velocity_scalar

    def _compute_l2_distance(self, world_point):
        """Compute the Euclidean distance between the object and the radar"""
        distance = np.sqrt((world_point[0] - self.world_radar[0])**2 + \
                           (world_point[1] - self.world_radar[1])**2)
        return distance

    def get_distance(self):
        """Get the distance"""
        return self.distance_t2

    def get_velocity(self):
        """Get the radial velocity"""
        return self._compute_velocity()
