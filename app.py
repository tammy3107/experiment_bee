import streamlit as st
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
from PIL import Image,ImageFilter
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

from detectron2.utils.visualizer import Visualizer
import os
from detectron2.data.datasets import register_coco_instances

os.chdir('/content/drive/MyDrive/Experiments_Bees/FRCNN')




st.write('''<style>
            body{
            text-align:center;
            background-color:#ACDDDE;
            }
            </style>''', unsafe_allow_html=True)




st.title('Honey Bee and Bumble Bee detector')

file_type = 'jpg'
uploaded_file = st.file_uploader("Choose a  file",type = file_type)




cfg = get_cfg()
cfg.MODEL.WEIGHTS = os.path.join("/content/drive/MyDrive/Experiments_Bees/FRCNN/output","model_final.pth")

cfg.DATASETS.TEST = ("my_dataset_test")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("my_dataset_test")
# print(test_metadata)

image = Image.open(uploaded_file)
image = np.array(image)
outputs =  predictor(image)
v = Visualizer(image[:, :, ::-1],
                metadata=test_metadata, 
                scale=0.8
                 )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
st.image(out.get_image()[:, :, ::-1])