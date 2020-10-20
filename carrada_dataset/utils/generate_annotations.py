"""Class to generate radar annotations"""
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

class AnnotationGenerator:

    """
    Class to generate annotations from points.

    PARAMETERS
    ----------
    matrix_shape: tuple
    points: list
        List of point coordinates corresponding to annotations in the matrix
    """

    def __init__(self, matrix_shape, points):
        self.points = np.array(points)
        self.x_shape, self.y_shape = matrix_shape

    def get_points(self):
        """Method to get the sparse pôints"""
        return self.points

    def get_box(self):
        """Method to get the box"""
        x_min, y_min = np.min(self.points, axis=0)
        x_max, y_max = np.max(self.points, axis=0)
        if x_min == x_max:
            x_min -= 1
            x_max += 1
        if y_min == y_max:
            y_min -= 1
            y_max += 1
        return [[x_min, y_min], [x_max, y_max]]

    def get_box_mask(self):
        """Method to get a mask with the box shape"""
        mask = np.zeros((self.x_shape, self.y_shape))
        box = self.get_box()
        mask[box[0][1]:box[1][1],
             box[0][0]:box[1][0]] = 1.
        return mask

    def get_mask(self):
        """Method to get the segmentation mask"""
        mask = np.ones((self.x_shape, self.y_shape))
        for point in self.points:
            mask[point[0], point[1]] = 0.
        mask = distance_transform_edt(mask)
        bool_mask = mask < 3.
        return bool_mask
