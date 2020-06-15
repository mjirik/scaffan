class ImageWithMask:

    def __init__(self, image, mask, file_name, ann_id, pixel_size):
        self.image = image
        self.mask = mask
        self.file_name = file_name
        self.ann_id = ann_id
        self.pixel_size = pixel_size

