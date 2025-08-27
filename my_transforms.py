from monai.transforms import (
    Transform,
    MapTransform,
)
import numpy as np
from typing import Mapping, Hashable
from monai.config import KeysCollection
import os


class SliceFromArray(Transform):
    def __init__(
            self, dim: int, slice_num: int
    ) -> None:
        self.dim = dim
        self.slice_num = slice_num

    def __call__(self, data: np.ndarray
                 ) -> np.ndarray:
        data = data.take(self.slice_num, axis=self.dim)
        return data


class SliceFromArrayd(MapTransform):
    def __init__(
            self, keys: KeysCollection, dim: int, slice_num: int
    ) -> None:

        self.keys = keys
        self.sliceFromArray = SliceFromArray(dim, slice_num)
        self.dim = dim
        self.slice_num = slice_num

    def __call__(self, data: Mapping[Hashable, np.ndarray]
                 ) -> Mapping[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.sliceFromArray(d[key])
        return d


class MaximumIntensityProjection(Transform):
    """
    This will be a simplified version of MIP, it will take an axis and take the maximum through
    all the slices on that axis and return an array with a smaller dimension
    """

    def __init__(
            self, axis: int
    ) -> None:
        self.axis = axis

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """

        :param array:
        :return:
        """
        return np.max(array, axis=self.axis)


class MaximumIntensityProjectiond(MapTransform):
    def __init__(
            self, keys: KeysCollection, axis: int
    ) -> None:
        self.keys = keys
        self.axis = axis
        self.maximumIntensityProjection = MaximumIntensityProjection(axis)

    def __call__(self, arrayd: Mapping[Hashable, np.ndarray]
                 ) -> Mapping[Hashable, np.ndarray]:
        d = dict(arrayd)
        for key in self.keys:
            d[key] = self.maximumIntensityProjection(d[key])
        return d




class LoadNumpy(Transform):
    def __init__(
            self
    ) -> None:
        """
        For loading npy files
        """

    def __call__(self, filename: str
    ) -> np.ndarray:
        """
        :param filenamesd: a list of dictionary with keys as hashable and value as a path
        """
        if not os.path.isfile or filename.split('.')[-1] != 'npy':
            raise Exception(f"filename path {filename} is invalid")
        return np.load(filename)



class LoadNumpyd(MapTransform):
    def __init__(
            self, keys: KeysCollection, image_only: bool = False
    ) -> None:
        """
        For loading npy files
        :param keys:
        """
        self.loadNumpy = LoadNumpy()
        self.keys = keys

    def __call__(self, filenamed: Mapping[Hashable, str]
                 ) -> Mapping[Hashable, np.ndarray]:
        """
        :param filenamed: a list of dictionaries with keys as hashable and value as a path
        """
        d = dict(filenamed)
        for key in self.keys:
            path = filenamed[key]
            d[key] = self.loadNumpy(path)
        return d

