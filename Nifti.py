import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import ndimage


class Nifti:
    # Default initializer requires a path to nii image to load
    def __init__(self, data_path, annotation_path=""):
        self.__load_image(data_path, annotation_path)
        self.__intelligent_crop()
        # self.__crop_image((0.5, 0.5))

    """Images are stored in private __data and __annotation structure. 
    Each one is in 3D image in shape of [horizontal, frontal, median]"""

    def __load_image(self, data_path, annotation_path):
        self.__data = nib.load(data_path).get_fdata()
        self.__normalize(0, 1)
        try:
            self.__annotations = nib.load(annotation_path).get_fdata()
            if self.__annotations.shape != self.__data.shape:
                raise Exception(
                    "The annotation file shape {} and data file shape {} are not matching.".format(self.__data.shape,
                                                                                                   self.__annotations.shape))
        except FileNotFoundError:
            self.__annotations = None
            raise FileNotFoundError("No annotation!")
        # Transposing as in accordance to tf.image manner of [batch, ..., channel]
        self.__transpose_image([1, 0, 2])

    # Normalization of data to minmax values. It is a global normalization rather than slice-wise normalization.
    def __normalize(self, min_value, max_value):
        self.__data = ((self.__data - np.min(self.__data)) / (np.max(self.__data) - np.min(self.__data)) * (
                max_value - min_value) - min_value)

    def __transpose_image(self, axes):
        self.__data = np.transpose(self.__data, axes)
        if self.is_annotated():
            self.__annotations = np.transpose(self.__annotations, axes)

    def __crop_image(self, ratios):
        if self.__annotations is None:
            raise Exception("Annotations is {}".format(None))
        if ratios[0] + ratios[1] != 1:
            raise Exception("Ratios {} and {} are not summing up to 1.".format(ratios[1], ratios[2]))
        left_crop = int(ratios[0] * (self.shape()[2] - self.shape()[1]))
        right_crop = left_crop + self.shape()[1]
        self.__data = self.__data[:, :, left_crop: right_crop]
        self.__annotations = self.__annotations[:, :, left_crop: right_crop]

    def __intelligent_crop(self):
        if self.__annotations is None:
            raise Exception("Annotations is {}".format(None))
        center_of_mass = np.round(ndimage.center_of_mass(np.ceil(self.__data))).astype(int)
        # Half-length of a maximum possible rectangle side, making center of mass in the center of image.
        # Take into consideration, that center of mass is the center of gravity for whole 3D image, not just a slice.
        a = int(min([center_of_mass[1], center_of_mass[2], self.__data.shape[1] - center_of_mass[1],
                     self.__data.shape[2] - center_of_mass[2]]))
        self.__data = self.__data[:, center_of_mass[1] - a: center_of_mass[1] + a,
                      center_of_mass[2] - a:center_of_mass[2] + a]
        self.__annotations = self.__annotations[:, center_of_mass[1] - a: center_of_mass[1] + a,
                             center_of_mass[2] - a:center_of_mass[2] + a]

    def get_image(self):
        return self.__data, self.__annotations

    def get_size(self):
        return self.__data.shape

    # Returns horizontal slice of an image
    def get_image_slice(self, slice_number):
        return self.__data[slice_number, :, ]

    # Returns horizontal slice of an annotation
    def get_annotation_slice(self, slice_number):
        return self.__annotations[slice_number, :, :]

    def get_slices(self, slice_number):
        return self.get_image_slice(slice_number), self.get_annotation_slice(slice_number)

    def get_number_of_slices(self) -> int:
        return self.__data.shape[1]

    def get_number_of_annotation_slices(self) -> int:
        return self.__annotations.shape[1] if self.__annotations is not None else 0

    # Methods allows user to preview desired slice.
    def preview(self, slice_number, show_data_slice=True, show_annotation_slice=False):
        if not self.is_annotated() and show_annotation_slice:
            raise Exception("Image not provided with annotations.")

        if slice_number > self.__data.shape[0] or slice_number < 0:
            raise Exception(
                "No horizontal slice {}. Number of horizontal slices is {}".format(slice_number, self.shape()[2]))

        if show_data_slice and not show_annotation_slice:
            plt.imshow(self.__data[slice_number, :, :])
            plt.title("Raw image slice")

        elif show_data_slice and show_annotation_slice:
            plt.imshow(self.__data[slice_number, :, :] * 0.4 + self.__annotations[slice_number, :, :] * 0.6)
            plt.title("Image slice with annotated lesions")

        elif not show_data_slice and show_annotation_slice:
            plt.imshow(self.__annotations[slice_number, :, :])
            plt.title("Just lesions of a slice")
        plt.show()

    # Method checks whether image was provided with annotations.
    def is_annotated(self):
        return True if self.__annotations is not None else False


def __test():
    import os

    dataset_path = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/zenodo'

    image = Nifti(os.path.join(dataset_path,"shifts_ms_pt1","shifts_ms_pt1","msseg","dev_in","t1","4_T1_isovox.nii"), os.path.join(dataset_path,"shifts_ms_pt1","shifts_ms_pt1","msseg","dev_in","gt","4_gt_isovox.nii"))

    image.preview(slice_number=160, show_data_slice=True, show_annotation_slice=False)
    print(image.get_size())



if __name__ == "__main__":
    __test()
#%%
