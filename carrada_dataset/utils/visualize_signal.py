"""Class to visualize a range-Doppler map"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage import transform
from scipy.ndimage.measurements import center_of_mass
plt.rcParams.update({'font.size': 24})


class SignalVisualizer():

    """
    Class to visualise radar signal

    PARAMETERS
    ----------
    matrix: numpy array
        Radar signal, supported: range-Doppler and range-angle.
    """

    SCALING_FACTOR = 4

    def __init__(self, matrix):
        self.matrix = matrix
        self.scaling_factor = self.__class__.SCALING_FACTOR
        self.x_shape, self.y_shape = self.matrix.shape
        self.image, self.color_scale_params = self._transform
        self.annotations = list()
        self.colors = [[0., 1., 0.], [0., 1., 1.], [0., 1., 1.]]
        self.masks = list()

    @property
    def _transform(self):
        min_value, max_value = np.min(self.matrix), np.max(self.matrix)
        cmap = plt.get_cmap('plasma')
        norm = colors.Normalize(vmin=min_value, vmax=max_value)
        colored_matrix = cmap(norm(self.matrix))
        colored_matrix = colored_matrix[:, :, :3]
        x_shape, y_shape, _ = colored_matrix.shape
        resized_matrix = transform.resize(colored_matrix,
                                          (self.scaling_factor*x_shape,
                                           self.scaling_factor*y_shape, 3),
                                          order=0)
        color_scale_params = [cmap, norm]
        return resized_matrix, color_scale_params

    @property
    def get_matrix(self):
        """Method to get the original signal"""
        return self.matrix

    @property
    def get_image(self):
        """Method to get the image to visualise the signal"""
        return self.image

    def add_annotation(self, index, points, annotation_type):
        """Method to add an annotation to the signal

        PARAMETERS
        ----------
        index: int
            Index of the added annotation in the annotation list
        points: numpy array
            Point coordinates corresponding to the annotation
        annotation_type: str
            Supported: 'sparse', 'box', 'dense', 'box_mask'
        """
        self.annotations.append(points)
        mask = self._get_mask(index, annotation_type)
        self.masks.append(mask)

    def save_scale(self, path, signal_type='range_doppler', color_scale=None,
                   rotation=False, save_img=True, plot_img=False):
        """
        Method to create and save a visulisation

        PARAMETERS
        ----------
        path: string
        color_scale: boolean
        Signal_type: string
            Type of signal to visualise.
            Supported: 'range_doppler', 'range_angle'
        color_scale: boolean
            Display the color scale
        rotation: boolean
            Rotate the matrix for visualization purpose
        save_img: boolean
            Save the image (at the given path)
        plot_img: boolean
            Plot the image
        """
        img = self._format_img(self.image, signal_type, color_scale, rotation)
        if save_img:
            plt.savefig(path)
        if plot_img:
            plt.show(img)
        plt.close()

    def save_annotation(self, index, path, signal_type='range_doppler', color_scale=None,
                        localize=False, rotation=False, save_img=True, plot_img=False):
        """
        Method to save the image with annotation

        PARAMETERS
        ----------
        index: int
            Index of the annotation to visualize, i.e. position in the list
        path: str
            Path to save the image
        signal_type: str
            Supported: 'range_doppler', 'range_angle'.
        color_scale: boolean
            Display the color scale
        localize: bool
            Crop the image around the annotation
        rotation: bool
            Rotate the image for visualization purpose (180 degrees)
        save_img: boolean
            Save the image (at the given path)
        plot_img: boolean
            Plot the image
        """
        transformed_image = self._get_annotated_image(index, localize)
        if localize:
            fig, ax = plt.subplots(figsize=(7, 12))
            if rotation:
                img = ax.imshow(np.rot90(transformed_image, 2), cmap=self.color_scale_params[0])
            else:
                img = ax.imshow(transformed_image, cmap=self.color_scale_params[0])
        else:
            img = self._format_img(transformed_image, signal_type, color_scale, rotation)
        if save_img:
            plt.savefig(path)
        if plot_img:
            plt.show(img)
        plt.close()

    def save_multiple_annotations(self, path, signal_type='range_doppler', color_scale=None,
                                  rotation=False, save_img=True, plot_img=False):
        """
        Method to save the image with all the added annotations

        PARAMETERS
        ----------
        path: str
            Path to save the image
        signal_type: str
            Supported: 'range_doppler', 'range_angle'.
        color_scale: boolean
            Display the color scale
        rotation: bool
            Rotate the image for visualization purpose (180 degrees)
        save_img: boolean
            Save the image (at the given path)
        plot_img: boolean
            Plot the image
        """
        transformed_image = self._get_multiple_annotated_image()
        img = self._format_img(transformed_image, signal_type, color_scale, rotation)
        if save_img:
            plt.savefig(path)
        if plot_img:
            plt.show(img)
        plt.close()

    def _format_img(self, img, signal_type, color_scale, rotation):
        image_size = img.shape
        if signal_type == 'range_doppler':
            fig, ax = plt.subplots(figsize=(7, 12))
            ax.set_xticks([0, int(image_size[1]/2)-1, image_size[1]-1])
            ax.set_yticks([0,
                           image_size[0]*1/5-1,
                           image_size[0]*2/5-1,
                           image_size[0]*3/5-1,
                           image_size[0]*4/5-1,
                           image_size[0]-1])
            ax.set_yticklabels([50, 40, 30, 20, 10, 0])
            if rotation:
                ax.set_xticklabels([13.5, 0, -13.5])
            else:
                ax.set_xticklabels([-13.5, 0, 13.5])
            ax.set_ylabel('Distance (m)')
            ax.set_xlabel('Doppler Effect')
        else:
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_xticks([0, int(image_size[1]/2)-1, image_size[1]-1])
            ax.set_yticks([0,
                           image_size[0]*1/5-1,
                           image_size[0]*2/5-1,
                           image_size[0]*3/5-1,
                           image_size[0]*4/5-1,
                           image_size[0]-1])
            ax.set_yticklabels([50, 40, 30, 20, 10, 0])
            ax.set_xticklabels([-90, 0, 90])
            ax.set_ylabel('Distance (m)')
            ax.set_xlabel('Angle (Degree)')
        if rotation:
            im = ax.imshow(np.rot90(img, 2), cmap=self.color_scale_params[0])
        else:
            im = ax.imshow(img, cmap=self.color_scale_params[0])
        if color_scale:
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Normalised Intensity')
            cbar.ax.set_yticks([0, 1024*1/4, 1024*2/4,
                                1024*3/4, 1024])
            cbar.ax.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1.])
        return fig

    def reset_annotation(self):
        """Reset the annotation list"""
        self.annotations = list()
        self.masks = list()

    def _get_mask(self, index, annotation_type):
        if annotation_type == 'sparse':
            mask = self._annotate_with_points(self.annotations[index])
        elif annotation_type == 'box':
            mask = self._annotate_with_box(self.annotations[index])
        elif annotation_type in ['box_mask', 'dense']:
            mask = self._annotate_with_mask(self.annotations[index])
        else:
            raise ValueError('Annotation type {} is not supported.'.format(annotation_type))
        return mask

    def _get_annotated_image(self, index, localize):
        transformed_image = self.image.copy()
        try:
            mask = (self.masks[index] * 255).astype(np.uint8)
        except IndexError:
            print('You need to add annotation before visualize it !')
        zeros = np.zeros(mask.shape).astype(np.uint8)
        if self.colors[index] == [0., 1., 0.]:
            mask = np.dstack([zeros, mask, zeros])
        else:
            mask = np.dstack([zeros, mask, mask])
        transformed_image = (transformed_image * 255).astype(np.uint8)
        transformed_image = cv2.addWeighted(transformed_image, 1., mask, 0.5, 0)
        if localize:
            centroid = self._get_centroid(mask)
            transformed_image = transformed_image[max(centroid[0]-80, 0):
                                                  min(centroid[0]+80,
                                                      transformed_image.shape[0]),
                                                  max(centroid[1]-80, 0):
                                                  min(centroid[1]+80,
                                                      transformed_image.shape[1])]
        return transformed_image

    def _get_multiple_annotated_image(self):
        mask = np.zeros((self.x_shape*self.scaling_factor, self.y_shape*self.scaling_factor))
        zeros = np.zeros(mask.shape).astype(np.uint8)
        if len(self.masks) == 0:
            raise Exception('You need to add annotation before visualize it!')
        transformed_image = self.image.copy()
        transformed_image = (transformed_image * 255).astype(np.uint8)
        for i in range(len(self.masks)):
            if self.colors[i] == [0., 1., 0.]:
                instance_mask = np.dstack([zeros, self.masks[i], zeros])
            else:
                instance_mask = np.dstack([zeros, self.masks[i], self.masks[i]])
            instance_mask = (instance_mask * 255).astype(np.uint8)
            transformed_image = cv2.addWeighted(transformed_image, 1., instance_mask, 0.5, 0)
        return transformed_image

    def _get_centroid(self, mask):
        centroid = center_of_mass(mask)
        centroid = [int(c) for c in centroid]
        return centroid

    def _annotate_with_points(self, annotation):
        mask = np.zeros((self.x_shape*self.scaling_factor, self.y_shape*self.scaling_factor))
        for point in annotation:
            mask[point[0]*self.scaling_factor:
                 (point[0]*self.scaling_factor+self.scaling_factor),
                 point[1]*self.scaling_factor:
                 (point[1]*self.scaling_factor+self.scaling_factor)] = 1.
        return mask

    def _annotate_with_box(self, annotation):
        mask = np.zeros((self.x_shape*self.scaling_factor, self.y_shape*self.scaling_factor))
        mask[annotation[0][0]*self.scaling_factor:annotation[1][0]*self.scaling_factor,
             annotation[0][1]*self.scaling_factor:annotation[0][1]*self.scaling_factor +
             self.scaling_factor] = 1.
        mask[annotation[0][0]*self.scaling_factor:annotation[1][0]*self.scaling_factor,
             annotation[1][1]*self.scaling_factor:annotation[1][1]*self.scaling_factor +
             self.scaling_factor] = 1.
        mask[annotation[0][0]*self.scaling_factor:annotation[0][0]*self.scaling_factor +
             self.scaling_factor,
             annotation[0][1]*self.scaling_factor:annotation[1][1]*self.scaling_factor] = 1.
        mask[annotation[1][0]*self.scaling_factor:annotation[1][0]*self.scaling_factor +
             self.scaling_factor,
             annotation[0][1]*self.scaling_factor:annotation[1][1]*self.scaling_factor] = 1.
        return mask

    def _annotate_with_mask(self, annotation):
        mask = transform.resize(annotation, (self.x_shape*self.scaling_factor,
                                             self.y_shape*self.scaling_factor), order=0)
        return mask
