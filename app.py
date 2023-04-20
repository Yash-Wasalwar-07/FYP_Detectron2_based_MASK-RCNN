import streamlit as st
import torch,torchvision
# import req
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# print(torch.__version__, torch.cuda.is_available())
# assert torch.__version__.startswith("1.8")
setup_logger()

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.title('Potato Leaf Disease Prediction')

st.write('\n')

def main() :
    add_bg_from_url()
    file_uploaded = st.file_uploader('Choose an image...', type = ['jpg','jpeg','png'])
    if file_uploaded is not None :
        image = cv2.imread(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure(figsize = (5,5))
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        predict_image(image)
        #result, confidence = predict_class(image)
        #string = f'This image likely belongs to {result} with a confidence of {confidence}%'       
        #st.success(string)
        #st.success(st.write('Prediction : {}'.format(result)))
        #st.success(st.write('Confidence : {}%'.format(confidence)))\
        
        
def predict_image(image):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATASETS.TRAIN = ("category_train",)
  cfg.DATASETS.TEST = ()
  cfg.DATALOADER.NUM_WORKERS = 2
#   cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  
  cfg.MODEL.WEIGHTS = "model_final.pth"
  
  cfg.MODEL.DEVICE = "cpu"
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.BASE_LR = 0.00025
  cfg.SOLVER.MAX_ITER = 100
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
  
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
  
  predictor = DefaultPredictor(cfg)
  outputs = predictor(image)
  
  st.write('Writing pred_classes/pred_boxes output')
  st.write(outputs["instances"].pred_classes)
  st.write(outputs["instances"].pred_boxes)
  
  st.write('Using Vizualizer to draw the predictions on Image')
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  st.image(out.get_image()[:, :, ::-1])
  
def add_bg_from_url():
   st.markdown(
         f"""
         <style>
         .stApp {{
          
             background-image: url("https://media.istockphoto.com/photos/green-leaves-pattern-background-sweet-potato-leaves-nature-dark-green-picture-id1155672947?k=20&m=1155672947&s=170667a&w=0&h=Rbx7C6PzO3sCXdnPsOhEylL4i01k7ekfENUwVXpBB5U=");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

  footer = """
  <style>
    a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
  }
  a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
  }
  .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
  }
  </style>
  <div class="footer">
  <p style = "align:center; color:white">Developed with ❤ by C_11 Group</p>
  </div>
  """

st.markdown(footer, unsafe_allow_html = True)

if __name__ == '__main__' : main()
