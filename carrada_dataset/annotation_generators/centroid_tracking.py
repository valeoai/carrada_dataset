"""Class to track centroids of clusters in DoA"""
import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from .utils import compute_kl, convert_to_rd_points, visualise_doa, visualise_points_on_rd

class CentroidMatching:
    """
    Class to cluster a point cloud, project points
    and associate them to the closest centroids

    PARAMETERS
    ----------

    doa_points: numpy array
        DOA point cloud
    init_point: numpy array
        Reference point
    label: int
        Label of the tracked object
    """

    def __init__(self, doa_points, init_points, label):
        self.doa_points = doa_points
        self.init_points = init_points
        self.label = label
        self._create_clusters(self.doa_points, self.init_points)

    def _create_clusters(self, x_train, x_test):
        """Train an optimal Mean Shift and select the closest cluster"""
        bandwidth = self._select_bandwidth()
        mean_shift = self._get_mean_shift(bandwidth, x_train, x_test)
        self.doa_labels = mean_shift.labels_
        cluster_centers = mean_shift.cluster_centers_
        labels_pred = mean_shift.predict(x_test)
        self.centroids = [cluster_centers[k] for k in labels_pred]
        cluster_indexes = self.doa_labels == labels_pred
        self.cluster_size = np.sum(cluster_indexes)
        self.cluster = self.doa_points[cluster_indexes]

    def _get_mean_shift(self, bandwidths, x_train, x_test=None):
        """
        Compute the Mean Shift algorithm with an optimized bandwidth selection
        using the Jensen Shannon divergence.

        PARAMETERS
        ----------
        bandwidths: list of floats
            List of bandwidth values to compare
        x_train: numpy array
            DoA points to cluster
        x_test: numpy array
            Centroid to track. Default: None.

        RETURNS
        -------
        mean_shift: Sklearn Mean Shift
            Trained Mean Shift with optimal bandwidth selection
        """
        triple_js = list()
        means = list()
        covs = list()
        clusters = list()
        for single_bw in bandwidths:
            mean_shift = MeanShift(bandwidth=single_bw, n_jobs=15)
            mean_shift.fit(x_train)
            labels = mean_shift.labels_
            labels_pred = mean_shift.predict(x_test)
            targets = x_train[labels == labels_pred[0]]
            clusters.append(targets)
            if targets.shape[0] == 1:
                means.append(np.nan)
                covs.append(np.nan)
            else:
                target_mean = np.mean(targets, axis=0)
                target_cov = np.cov(targets, rowvar=False)
                means.append(target_mean)
                covs.append(target_cov)
        for i in range(1, len(bandwidths)-1):
            if not isinstance(means[i-1], (np.ndarray, np.generic)) \
               or not isinstance(covs[i-1], (np.ndarray, np.generic)) \
               or not isinstance(means[i], (np.ndarray, np.generic)) \
               or not isinstance(covs[i], (np.ndarray, np.generic)) \
               or not isinstance(means[i+1], (np.ndarray, np.generic)) \
               or not isinstance(covs[i+1], (np.ndarray, np.generic)):
                triple_js.append(np.nan)
            else:
                # Implement KS and JS divergences
                inter_mean_1 = (means[i-1] + means[i])/2.
                inter_cov_1 = (covs[i-1] + covs[i])/4.
                inter_mean_2 = (means[i] + means[i+1])/2.
                inter_cov_2 = (covs[i] + covs[i+1])/4.
                js_1 = (compute_kl(means[i-1], inter_mean_1, covs[i-1], inter_cov_1) +\
                        compute_kl(means[i], inter_mean_1, covs[i], inter_cov_1))/2.
                js_2 = (compute_kl(means[i], inter_mean_2, covs[i], inter_cov_2) +\
                        compute_kl(means[i+1], inter_mean_2, covs[i+1], inter_cov_2))/2.
                metric = js_1 + js_2
                triple_js.append(metric)

        triple_js = np.array(triple_js)
        try:
            best_bw_id = np.nanargmin(triple_js)
            best_bw = bandwidths[best_bw_id+1]
        except (IndexError, ValueError):
            best_bw = bandwidths[0]
        mean_shift = MeanShift(bandwidth=best_bw, n_jobs=15)
        mean_shift.fit(x_train)
        return mean_shift

    def _select_bandwidth(self):
        """Method to get a list of bandwidth to test depending on the category"""
        if self.label == 3:
            return list(np.arange(1., 7.))
        if self.label == 2:
            return list(np.arange(0.5, 3.5, 0.5))
        if self.label == 1:
            return list(np.arange(0.3, 2.1, 0.3))
        raise ValueError('Label {} is not supported.'.format(self.label))

    def get_doa_points(self):
        """Method to get DOA point cloud"""
        return self.doa_points

    def get_doa_labels(self):
        """Method to get Mean Shift labels for each point"""
        return self.doa_labels

    def get_centroids(self):
        """Method to get centroids of the Mean Shift clusters"""
        return np.array(self.centroids)

    def get_cluster_size(self):
        """Method to get the size of the tracked cluster"""
        return self.cluster_size

    def get_cluster(self):
        """Method to get the points in the tracked cluster"""
        return self.cluster

    def update(self, new_doa_points):
        """
        Method to update the status of tracked cluster

        PARAMETERS
        ----------
        new_doa_points: numpy array
            New DOA point cloud  to find the tracked cluster
        """
        self.doa_points = new_doa_points
        self._create_clusters(self.doa_points, self.centroids)


class CentroidTracking(Configurable):

    """
    Class to track a centroid on an entire DoA sequence

    PARAMETERS
    ----------
    seq_name: str
        Name of the sequence to process
    instances: list of str
        List of the instance names to process
    ref_ids: list of ids
        List of reference id names for each instance
    labels: list of int
        List of label associated to each instance
    min_frame_boundaries: list of str
        List of frame id where the target should be lost for each instance
    max_frame_boundaries: list of str
        List of frame id where the target should be lost for each instance
    """

    WORLD_CAMERA = (0., 0.)
    DOPPLER_RES = 0.41968030701528203
    RANGE_RES = 0.1953125

    def __init__(self, seq_name, instances, ref_ids, labels,
                 min_frame_boundaries, max_frame_boundaries):
        self.config_path = os.path.join(CARRADA_HOME, 'config.ini')
        super().__init__(self.config_path)
        self.cls = __class__
        if isinstance(instances, list):
            self.instances = instances
        else:
            raise TypeError('Instances should be in list.')
        if isinstance(ref_ids, list):
            self.ref_ids = ref_ids
        else:
            raise TypeError('Reference Ids should be in list.')
        if len(self.instances) != len(self.ref_ids):
            raise Exception('Instances and Reference Ids should match (same size)')
        if isinstance(labels, list):
            self.labels = labels
        else:
            raise TypeError('Reference Ids should be in list.')
        if isinstance(min_frame_boundaries, list):
            self.min_frame_boundaries = min_frame_boundaries
        else:
            raise TypeError('The minimum frame boundaries should be in list.')
        if isinstance(max_frame_boundaries, list):
            self.max_frame_boundaries = max_frame_boundaries
        else:
            raise TypeError('The maximum frame boundaries should be in list.')
        self.seq_name = seq_name
        self.paths = self._get_paths()
        with open(os.path.join(self.paths['seq'], 'rd_points.json'), 'rb') as fp:
            self.rd_points = json.load(fp)
        with open(os.path.join(self.paths['seq'], 'world_points.json'), 'rb') as fp:
            self.world_points = json.load(fp)
        self.doa_points = pd.read_csv(self.paths['doa_points'], sep=',')
        self.delta_distance = 0.
        self.delta_doppler = 0.

    def _get_paths(self):
        """Define paths"""
        paths = dict()
        paths['warehouse'] = self.config['data']['warehouse']
        paths['carrada'] = os.path.join(paths['warehouse'], 'Carrada')
        paths['seq'] = os.path.join(paths['carrada'], self.seq_name)
        paths['rd_numpy'] = os.path.join(paths['seq'],
                                         'range_doppler_numpy')
        paths['visualisation'] = os.path.join(paths['seq'],
                                              'centroid_tracking_jensen_shannon')
        paths['doa_points'] = os.path.join(paths['seq'], 'DOAPoints.csv')
        os.makedirs(paths['visualisation'], exist_ok=True)
        return paths

    def _select_doa_points(self, idx):
        """Select DoA points for a given index id"""
        idx_name = idx + '.png'
        doa_points = self.doa_points[self.doa_points['frameIdx'] == idx_name]
        doa_points = doa_points.iloc[:, [1, 3, 4]].to_numpy()
        return doa_points

    def _get_ref_coord(self, ref_id, instance):
        """Format DoA-Doppler point coordinates """
        world_point = self.world_points[ref_id][instance][0]
        x_coord = world_point[1]
        distance = self.rd_points[ref_id][instance]['distance'][0]
        doppler = self.rd_points[ref_id][instance]['velocity'][0]
        return np.array([[x_coord, distance, doppler]])

    def _process_sequence(self, save_vis_rd, save_vis_doa, data_type='centroid'):
        """Lead the tracking process in the future and past frames"""
        if data_type not in ('centroid', 'cluster'):
            raise ValueError('Recoreded data type should be centroid or cluster')
        annotations = dict()
        rd_paths = glob.glob(os.path.join(self.paths['rd_numpy'], '*.npy'))
        rd_paths.sort()
        for i in range(len(self.instances)):
            print('Process instance {}'.format(self.instances[i]))
            annotations[self.instances[i]] = dict()
            ids_after = list(range(int(self.ref_ids[i]), len(rd_paths)))
            ref_id_point = self._get_ref_coord(self.ref_ids[i], self.instances[i])
            annotations[self.instances[i]].update(self._track(self.instances[i],
                                                              self.ref_ids[i],
                                                              self.labels[i],
                                                              self.max_frame_boundaries[i],
                                                              ref_id_point,
                                                              ids_after,
                                                              rd_paths,
                                                              save_vis_rd,
                                                              save_vis_doa,
                                                              data_type))
            ids_before = list(reversed(range(int(self.ref_ids[i])+1)))
            annotations[self.instances[i]].update(self._track(self.instances[i],
                                                              self.ref_ids[i],
                                                              self.labels[i],
                                                              self.min_frame_boundaries[i],
                                                              ref_id_point,
                                                              ids_before,
                                                              rd_paths,
                                                              save_vis_rd,
                                                              save_vis_doa,
                                                              data_type))
        return annotations

    def _track(self, instance_id, ref_id, label, frame_boundary, ref_id_point,
               ids, rd_paths, save_vis_rd, save_vis_doa, data_type):
        """Tracking of an instance in a sequence of radar signals"""
        annotations = dict()
        ref_doa_points = self._select_doa_points(ref_id)
        centroid_matching = CentroidMatching(ref_doa_points, ref_id_point, label)
        path_vis = os.path.join(self.paths['visualisation'], instance_id)
        os.makedirs(path_vis, exist_ok=True)
        rd_matrix = np.load(rd_paths[ids[0]])
        for i in range(0, len(ids)):
            current_id = rd_paths[ids[i]].split('/')[-1].split('.')[0]
            try:
                rd_matrix = np.load(rd_paths[ids[i]])
            except FileNotFoundError:
                print('Warning: Index of RD data path has been exceeded due to lagged data')
                self._need_a_break
                break
            doa_points = self._select_doa_points(current_id)
            # Stop criteria: there are not enough points in DoA
            if doa_points.shape[0] > 0:
                if i > 0:
                    centroid_matching.update(doa_points)
            else:
                # Initialise the delta values for next step
                print('Stop criteria reached: not enough points in DoA')
                self._need_a_break
                break
            doa_centroids = centroid_matching.get_centroids()
            doa_cluster = centroid_matching.get_cluster()
            rd_points_centroids = convert_to_rd_points(doa_centroids, self.cls.WORLD_CAMERA)
            if data_type == 'cluster':
                rd_points_cluster = convert_to_rd_points(doa_cluster, self.cls.WORLD_CAMERA)

            if save_vis_rd:
                path_vis_rd_folder = os.path.join(path_vis, 'range-doppler')
                path_vis_rd_subfolder = os.path.join(path_vis_rd_folder, data_type)
                os.makedirs(path_vis_rd_folder, exist_ok=True)
                os.makedirs(path_vis_rd_subfolder, exist_ok=True)
                path_vis_rd = os.path.join(path_vis_rd_subfolder, current_id + '.png')
                if data_type == 'centroid':
                    visualise_points_on_rd(rd_matrix, path_vis_rd, rd_points_centroids,
                                           self.cls.RANGE_RES, self.cls.DOPPLER_RES)
                else:
                    visualise_points_on_rd(rd_matrix, path_vis_rd, rd_points_cluster,
                                           self.cls.RANGE_RES, self.cls.DOPPLER_RES)

            if save_vis_doa:
                path_vis_doa_folder = os.path.join(path_vis, 'doa')
                path_vis_doa_subfolder = os.path.join(path_vis_doa_folder, data_type)
                os.makedirs(path_vis_doa_folder, exist_ok=True)
                os.makedirs(path_vis_doa_subfolder, exist_ok=True)
                path_vis_doa = os.path.join(path_vis_doa_subfolder, current_id + '.png')
                doa_labels = centroid_matching.get_doa_labels()
                if data_type == 'centroid':
                    visualise_doa(doa_points, doa_labels, path_vis_doa, doa_centroids)
                else:
                    visualise_doa(doa_points, doa_labels, path_vis_doa)

            if data_type == 'centroid':
                annotations[current_id] = doa_centroids.tolist()
            else:
                annotations[current_id] = doa_cluster.tolist()
            if self.delta_distance == 0. and self.delta_doppler == 0.:
                ref_centroid = rd_points_centroids[0]
                self.delta_distance = ref_centroid[0]
                self.delta_doppler = ref_centroid[1]
            else:
                # Stop Criteria on delta distance / doppler max
                self.delta_distance = abs(ref_centroid[0] - rd_points_centroids[0][0])
                self.delta_doppler = abs(ref_centroid[1] - rd_points_centroids[0][1])
                ref_centroid = rd_points_centroids[0]
                if self.delta_distance > 5.:
                    # More than 5 meters in 0.1 sec
                    print('Stop criteria reached: Delta Distance = {}'.format(self.delta_distance))
                    self._need_a_break
                    break
                if self.delta_doppler > 5.:
                    # Variation of radial velocity > 5 m/s in 0.1s
                    print('Stop criteria reached: Delta Doppler = {}'.format(self.delta_doppler))
                    self._need_a_break
                    break
            if frame_boundary != "" and frame_boundary == current_id:
                print('Stop criteria reached: Custom frame boundary = {}'.format(frame_boundary))
                self._need_a_break
                break
        self._need_a_break
        return annotations

    def create_annotations(self, save_annotations=True, save_vis_rd=True, save_vis_doa=False,
                           data_type='centroid'):
        """
        Method to generate the tracked annotations

        PARAMETERS
        ----------
        save_annotations: bool
            If you want to save the annotations. Default: True.
        save_vis_rd: bool
            If you want to save the range-Doppler visualisation. Default: True.
        save_vis_doa: bool
            If you want to save the DoA-Doppler visualisation. Default: False.
        data_type: str
            Type of data to save. Supported types: 'cluster', 'centroid'.
        """
        np.random.seed(42)
        annotations = self._process_sequence(save_vis_rd=save_vis_rd, save_vis_doa=save_vis_doa,
                                             data_type=data_type)
        if save_annotations:
            self._save_annotations(annotations, data_type)
        return annotations

    def _save_annotations(self, annotations, data_type):
        path = os.path.join(self.paths['seq'],
                            'centroid_tracking_' + data_type + '_jensen_shannon.json')
        with open(path, 'w') as fp:
            json.dump(annotations, fp)

    @property
    def _need_a_break(self):
        """Method to instanciate attributes in case of a breaking loop"""
        self.delta_distance = 0.
        self.delta_doppler = 0.
