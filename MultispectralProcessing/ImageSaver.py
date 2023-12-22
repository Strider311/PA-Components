from ..Enums.MultispectralEnum import MultiSpectralEnum
import numpy as np
import matplotlib.pyplot as plt
import os


class ImageSaver():
    """_summary_
        Pass the output directory where the images should be saved
    """
    def __init__(self, outputDir: str) -> None:
        self.outputDir = outputDir

    def save_image(self, img, img_file_name: str, processing_type: MultiSpectralEnum, use_color_bar=False):
        self.save_image_as_jpg(img, img_file_name, processing_type)
        self.save_image_as_np_file(img, img_file_name, processing_type)
        self.save_debug_image(img, img_file_name, processing_type)

    def save_debug_image(self, img, img_file_name: str, processing_type: MultiSpectralEnum):
        fig = plt.figure(dpi=100)
        plt.imshow(img, cmap="gray")
        plt.colorbar()
        file = os.path.join(self.outputDir, processing_type.name, "debug",
                            img_file_name)
        fig.savefig(file, dpi=100)
        plt.close(fig)

    def save_image_as_jpg(self, img, img_file_name: str, processing_type: MultiSpectralEnum):
        fig = plt.figure(frameon=False, dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ## Enable this for Red Yellow Green visualization
        # plt.imshow(img, cmap=('RdYlGn')) 

        file = os.path.join(self.outputDir, processing_type.name, "images",
                            img_file_name)
        plt.imshow(img, cmap="gray")
        fig.savefig(file, bbox_inches='tight', pad_inches=0.0, dpi=100)
        plt.close(fig)

    def save_image_as_np_file(self, img, img_file_name: str, processing_type: MultiSpectralEnum):

        file_name = img_file_name.fileName.replace(".jpg", ".txt")
        final_output_path = os.path.join(
            self.outputDir, processing_type.name, "numpy", file_name)

        np.savetxt(final_output_path, img)
