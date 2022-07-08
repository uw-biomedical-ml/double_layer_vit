from pathlib import Path

from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_dir


DLS_CONFIG_PATH = Path(__file__).parent / "config" / "dls.py"
DLS_CATS_PATH = Path(__file__).parent / "config" / "dls.yml"


class DLSSegmentation(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(
            image_size,
            crop_size,
            split,
            DLS_CONFIG_PATH,
            **kwargs,
        )
        self.names, self.colors = utils.dataset_cat_description(DLS_CATS_PATH)
        self.n_cls = 3
        self.ignore_label = 0
        self.reduce_zero_label = True

    def update_default_config(self, config):
        #root_dir = dataset_dir()
        #path = Path(root_dir) / "ade20k"
        root_dir = "data"
        path = "data/dls"
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path + "/internal_test" #"/internal_test"
        elif self.split == "trainval":
            config.data.trainval.data_root = path + "/internal_test" #"/internal_test"
        elif self.split == "val":
            config.data.val.data_root = path +"/internal_test" #"/internal_test"
        elif self.split == "test":
            config.data.test.data_root = path + "/external_test"
        config = super().update_default_config(config)
        return config

    def test_post_process(self, labels):
        return labels + 1
