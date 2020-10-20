"""Script with utils functions"""

import random
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from carrada_dataset.utils.visualize_signal import SignalVisualizer

def compute_determinant(matrix):
    """Hypothese on the third feature"""
    det = np.linalg.det(matrix)
    #if det == 0.:
         # The det = 0 could be related to the third feature
        # det = np.linalg.det(matrix[:2, :2])
    if det == 0.:
        # Singular covariance matrix, should not be taken into account
        det = np.nan
    if np.isclose(det, 0):
        det = np.abs(det)
    return det

def compute_kl(mu1, mu2, sigma1, sigma2):
    """Method to compute the Kullback-Leibler Divergence between two Gaussians

    PARAMETERS
    ----------
    mu1, mu2: numpy arrays
        Means of the Gaussians
    sigma1, sigma2: numpy arrays
        Covariances of the Gaussians

    RETURNS
    -------
    kl: float
        KL divergence between the two gaussians
    """
    k = len(mu1)
    try:
        term1 = np.trace(np.matmul(np.linalg.inv(sigma2), sigma1))
        term2 = np.matmul(np.matmul((mu2 - mu1).T,
                                    np.linalg.inv(sigma2)),
                          (mu2 - mu1))
        det_sigma1 = compute_determinant(sigma1)
        det_sigma2 = compute_determinant(sigma2)
        # term3 = np.log(np.linalg.det(sigma2)/np.linalg.det(sigma1))
        term3 = np.log(det_sigma2)
        term4 = np.log(det_sigma1)
        kl = (term1 + term2 - k + term3 - term4)/2.
        return kl
    except np.linalg.LinAlgError:
        return np.nan

def convert_to_rd_points(data, world_camera):
    """Convert real world points with Doppler to range-Doppler coordinates

    PARAMETERS
    ----------
    data: numpy array
        Points to convert
    world_camera: tuple
        Coordinates of the camera in the real world

    RETURNS
    -------
    rd_points: numpy array
        Converted points
    """
    distances = np.sqrt((data[:, 0] - world_camera[0])**2 + \
                        (data[:, 1] - world_camera[1])**2)
    dopplers = data[:, 2]
    rd_points = [(distances[i], dopplers[i]) for i in range(data.shape[0])]
    return rd_points

def visualise_doa(doa_points, doa_labels, path, centroids=None):
    """Visualise and record DoA-Doppler point clouds"""
    if isinstance(centroids, np.ndarray) and len(centroids.shape) == 1:
        centroids = centroids.reshape(1, -1)
    n_clusters = np.unique(doa_labels).shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y',
                    'blueviolet', 'brown', 'burlywood',
                    'khaki', 'indigo', 'peru', 'pink',
                    'rosybrown', 'teal', 'seagreen'])
    for k, col in zip(range(n_clusters), colors):
        indexes = doa_labels == k
        ax.scatter(doa_points[indexes, 0], doa_points[indexes, 1],
                   doa_points[indexes, 2], 'ro', c=col)
    if isinstance(centroids, np.ndarray):
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 'ro', c='black')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Doppler-Velocity (m/s)')
    ax.set_title('3D representation of the DoA points with centroid')
    plt.savefig(path)
    plt.close()

def visualise_points_on_rd(rd_matrix, path, points, range_res, doppler_res):
    """Visualise and record range-Doppler matrices with projected points

    PARAMETERS
    ----------
    rd_matrix: numpy array
        Range-Doppler signal matrix
    path: str
        Path to save the visualisation
    points: numpy array
        Points to visualise in the range-Doppler
    range_res: float
        Range resolution
    doppler_res: float
        Doppler resolution
    """
    rd_img = SignalVisualizer(rd_matrix).get_image
    for point in points:
        range_coord = (point[0] / range_res).astype(int)
        doppler_coord = (point[1] / doppler_res).astype(int)
        if point[1] < 0:
            doppler_coord += int(rd_matrix.shape[1]/2 - 1)
        else:
            doppler_coord += int(rd_matrix.shape[1]/2)
        rd_img[range_coord*4:(range_coord*4+4),
               doppler_coord*4:(doppler_coord*4+4)] = [0., 0., 0.]
    plt.imsave(path, rd_img)
    plt.close()

def get_random_rgb(seed):
    """Generate random RGB colors for a given seed"""
    random.seed(seed)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return [r, g, b]

def threshold_mask(mask, threshold=0.5):
    """Get a binary mask with threshold on probabilities"""
    mask[np.where(mask >= threshold)] = 1.
    mask[np.where(mask < threshold)] = 0.
    return mask

def compute_line_coefs(point_a, point_b):
    """
    Method to compute the coefficients of the straight line between two points
    (D): ax + by + c = 0
    Here, b is fixed to -1 in order to a have (D): y = ax + c
    """
    b_coef = -1
    if (point_b[0] - point_a[0]) == 0:
        a_coef = 0
    else:
        a_coef = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
    c_coef = point_b[1] - a_coef*point_b[0]
    return np.array([a_coef, b_coef, c_coef])

def compute_orthogonal_proj(line_coefs, point_coordinates):
    """Method to compute the orthogonal projection of a point on a straight line
    Let's define (D) the straight line which coeff are given as parameters
    Lets (D'): y = -(1/a)*x + (c'/a) the equation of the orthogonal line of (D)
    We note a, the coefficient of (D) and c' a specific coefficient for (D')
    """
    a = line_coefs[0]
    c = line_coefs[2]
    # Compute c_prime by replacing with the coordinates of the point
    c_prime = a*point_coordinates[1] + point_coordinates[0]
    x_proj = (c_prime-a*c)/((a**2)+1)
    y_proj = (a*c_prime+c)/((a**2)+1)

    return np.array([x_proj, y_proj])
