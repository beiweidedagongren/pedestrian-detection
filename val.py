import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/论文/MYPBFPN-TripletAttentionConv-FASFF/weights/best.pt')
    model.val(data='datasets/BDD100K/coco128.yaml',
              split='val',
              imgsz=640,
              batch=8,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='MYPBFPN-TripletAttentionConv-FASFF',
              )