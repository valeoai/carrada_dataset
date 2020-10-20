"""Class to transform annotations depending on data"""
import numpy as np


class AnnotationTransformer:

    """
    Class to transform annotations in radar representation coordinates

    PARAMETERS
    ----------
    points: numpy array
        DOA-Doppler points to convert in radar representation coordinates
    matrix_shape: tuple
        Size of the radar representation
    """

    DOPPLER_RES = 0.41968030701528203
    RANGE_RES = 0.1953125
    ANGLE_RES = 0.01227184630308513
    WORLD_RADAR = (0.0, 0.0)

    def __init__(self, points, matrix_shape):
        self.cls = self.__class__
        self.points = points
        self.x_shape, self.y_shape = matrix_shape

    def to_ra(self):
        """
        Method to convert DOA-Doppler points to range-angle representation

        PARAMETERS
        ----------
        None

        RETURNS
        -------
        transformed_annotations: list
             Coordinates in the radar representation
        """
        transformed_annotations = list()
        mid_y_shape = int(self.y_shape/2)
        for point in self.points:
            range_value = np.sqrt((point[0] - self.cls.WORLD_RADAR[0])**2 +\
                                  (point[1] - self.cls.WORLD_RADAR[1])**2)
            range_coordinate = self.x_shape - int((range_value / self.cls.RANGE_RES))
            angle_value = np.arctan(point[0]/point[1])
            angle_coordinate = mid_y_shape + int(angle_value / self.cls.ANGLE_RES)
            transformed_annotations.append([range_coordinate, angle_coordinate])
        return transformed_annotations

    def to_rd(self):
        """
        Method to convert DOA-Doppler points to range-doppler representation

        PARAMETERS
        ----------
        None

        RETURNS
        -------
        transformed_annotations: list
             Coordinates in the radar representation
        """
        transformed_annotations = list()
        for point in self.points:
            range_value = np.sqrt((point[0] - self.cls.WORLD_RADAR[0])**2 +\
                                  (point[1] - self.cls.WORLD_RADAR[1])**2)
            range_coordinate = int(range_value / self.cls.RANGE_RES)
            doppler_value = point[2]
            doppler_coordinate = int(doppler_value / self.cls.DOPPLER_RES)
            # if doppler_coordinate < 0:
                # doppler_coordinate += int(self.y_shape/2 - 1)
            # else:
            doppler_coordinate += int(self.y_shape/2)
            transformed_annotations.append([range_coordinate, doppler_coordinate])
        return transformed_annotations
