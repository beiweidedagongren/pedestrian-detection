from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 加载模型
    model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    # model = YOLO('runs/detect/train5/weights/ 开始训练，使用 Adam 优化器相关的参数
    model.train(data='datasets/TS-CLAHE/coco128.yaml',
                epochs=200,  # 将训练周期设为250last.pt')
    #
                batch=8,
                device='0',
                workers=4,
                lr0=0.002,  # 初始学习率，可以根据需要调整
                optimizer='Adam',  # 使用 Adam 优化器

                resume=True,
                mosaic=True,  # 开启 mosaic 数据增强


                close_mosaic=0

                )


    