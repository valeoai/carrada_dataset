"""Class to transform data"""
import numpy as np

class DataTransformer:

    """
    Class to transform from RAD (Range-Angle-Doppler) to RA and RD

    PARAMETERS
    ----------
    rad_matrix: numpy array
        Range Angle Doppler matrix
    """

    def __init__(self, rad_matrix):
        self.rad_matrix = rad_matrix

    def to_ra(self):
        """
        Convert RAD to RA representation.
        Simple and noisy representation.
        Futur: log representation.

        PARAMETERS
        ----------
        None

        RETURNS
        -------
        ra_matrix: numpy array
            Range-Angle data with according processing
        """
        ra_matrix = np.max(self.rad_matrix, axis=2)
        return ra_matrix

    def to_rd(self):
        """
        Convert RAD to RD representation.
        The RD representation used wasn't computed on the same freq space.
        Futur: Compare with summed representation after FFT.

        PARAMETERS
        ----------
        None

        RETURNS
        -------
        rd_matrix: numpy array
            Range-Doppler data with according processing
        """
        processing = np.rot90(self.rad_matrix, 2)
        # Undo the FFT on the angle dim (Rx/Tx pairs)
        processing = np.fft.ifftshift(processing, axes=1)
        processing = np.fft.ifft(processing, axis=1)
        processing = pow(np.abs(processing), 2)
        processing = np.sum(processing, axis=1)
        rd_matrix = 10*np.log10(processing + 1)
        return rd_matrix

    def get_rad_matrix(self):
        """Get the RAD matrix"""
        return self.rad_matrix
