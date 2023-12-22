from PIL import Image

class ImageLoaderHelper():
    def __init__(self) -> None:
        pass

    def load(filePath: str):
        image = Image.open(filePath)
        return np.array(image).astype('float64')