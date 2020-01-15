import os
import random
import cv2

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from loader import get_data_dicts

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TEST = ("cats_val",)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

predictor = DefaultPredictor(cfg)

dataset_dicts = get_data_dicts("../data/validation")

for idx, d in enumerate(random.sample(dataset_dicts, 3)):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], 
        metadata = MetadataCatalog.get("cats_val"),
        scale = 0.8,
        instance_mode = ColorMode.IMAGE_BW
    )

    print(outputs["instances"])
    print(outputs["instances"].scores)

    # Save image predictions
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # image_name = f"inference_{idx}.jpg"
    # cv2.imwrite(image_name, v.get_image()[:, :, ::-1])
    
