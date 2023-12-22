import numpy as np
from ImageLoader import ImageLoaderHelper

class Multispectral():

    def __init__(self, output_dir):
        self.image_loader = ImageLoaderHelper(output_dir)

    def __normalize__(self, vi_array):

        normalized = (vi_array-np.min(vi_array)) / \
            (np.max(vi_array)-np.min(vi_array))
        return normalized

    def ndvi(self, nir_img, red_img):
        nir = self.image_loader.load(nir_img)
        red = self.image_loader.load(red_img)

        ndvi = np.where((nir + red) == 0.,
                        0, (nir - red) /
                        (nir + red))

        normalized_ndvi = self.__normalize__(ndvi)

        return normalized_ndvi

    def gndvi(self, green_img, nir_image):

        green = self.image_loader.load(green_img)
        nir = self.image_loader.load(nir_image)

        gndvi = np.where((nir + green) == 0., 0,
                         (nir - green) /
                         (nir + green))

        normalized_gndvi = self.__normalize__(gndvi)

        return normalized_gndvi

    def bai(self, red_img, nir_img):

        red = self.image_loader.load(red_img)
        nir = self.image_loader.load(nir_img)

        bai = 1 / ((np.power((0.1 - red), 2))
                   + np.power((0.06 - nir), 2))

        normalized_bai = self.__normalize__(bai)

        return normalized_bai

    def savi(self, red_img, nir_img, tuning_param_l=0.5):

        red = self.image_loader.load(red_img)
        nir = self.image_loader.load(nir_img)

        savi = (((nir - red) /
                (nir + red + tuning_param_l)) *
                (1 + tuning_param_l))

        normalized_savi = self.__normalize__(savi)

        return normalized_savi

    def cig(self, nir_img, green_img):

        nir = self.image_loader.load(nir_img)
        green = self.image_loader.load(green_img)

        CIg = (nir / green) - 1

        normalized_cig = self.__normalize__(CIg)
        return normalized_cig

    def cire(self, nir_img, red_e_img):

        nir = self.image_loader.load(nir_img)
        red_e = self.image_loader.load(red_e_img)

        CIre = (nir / red_e) - 1

        normalized_cire = self.__normalize__(CIre)
        return normalized_cire

    def gemi(self, nir_img, red_img):

        nir = self.image_loader.load(nir_img)
        red = self.image_loader.load(red_img)

        nir_squared = np.power(nir, 2)
        red_squared = np.power(red, 2)

        theta = ((2 * (nir_squared - red_squared) + 1.5*nir + 0.5*red) /
                 (nir + red + 0.5))

        gemi = (theta * (1 - 0.25 * theta) - ((red - 0.125)) /
                (1-red))

        normalized_gemi = self.__normalize__(gemi)

        return normalized_gemi

    def msavi2(self, nir_img, red_img):

        nir = self.image_loader.load(nir_img)
        red = self.image_loader.load(red_img)

        msavi2 = ((0.5 * (2 * (nir + 1)))
                  - (np.sqrt(np.power((2*nir + 1), 2)))
                  - 8*(nir - red))

        normalized_msavi2 = self.__normalize__(msavi2)
        return normalized_msavi2

    def mtvi2(self, red_img, green_img, nir_img):

        red = self.image_loader.load(red_img)
        green = self.image_loader.load(green_img)
        nir = self.image_loader.load(nir_img)

        mtvi2 = (1.5 * (1.2 * (nir - green) - 2.5 * (red - green)) *
                 np.sqrt(np.power((2 * nir + 1), 2) -
                         (6 * nir - 5 * np.sqrt(red)) - 0.5))

        normalized_mtvi2 = self.__normalize__(mtvi2)

        return normalized_mtvi2

    def ndre(self, nir_img, red_e_image):

        nir = self.image_loader.load(nir_img)
        red_e = self.image_loader.load(red_e_image)

        ndre = np.where((nir + red_e) == 0, 0,
                        (nir - red_e) / (nir + red_e))

        normalized_ndre = self.__normalize__(ndre)

        return normalized_ndre

    def ndwi(self, green_img, nir_image):

        green = self.image_loader.load(green_img)
        nir = self.image_loader.load(nir_image)

        ndwi = np.where((green + nir) == 0, 0,
                        (green - nir) / (green + nir))

        normalized_ndwi = self.__normalize__(ndwi)
        return normalized_ndwi

    def srre(self, nir_image, red_e_image):

        nir = self.image_loader.load(nir_image)
        red_e = self.image_loader.load(red_e_image)

        srre = nir / red_e

        normalized_srre = self.__normalize__(srre)

        return normalized_srre

    def rtvi_core(self, nir_image, green_img):

        nir = self.image_loader.load(nir_image)
        green = self.image_loader.load(green_img)

        rtvi = np.where((green + nir) == 0, 0,
                        (green - nir) / (green + nir))

        normalized_rtvi = self.__normalize__(rtvi)

        return normalized_rtvi
