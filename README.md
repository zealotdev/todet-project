# todet-project
### Objects detection and Text extraction django-app

### Quick intro
todet is a django web application that let user detect objects/text and extract texts from their uploaded images.

- For objects detection YOLOv3(You Only Look Once) dataset was used.
- For text detection EAST(Effiecient and Accurate Scene Text) model was used.
- For text extraction Tesseract OCR was used.


### How to run in your machine

- Download ```yolov3.weights```  and place it under ```objectdet/lib/yolo-coco``` folder
- Download ```frozen_east_text_detection.pb``` and place it under ```objectdet/lib/east```
- Install tessaract-ocr in your machine
- Make sure you have pip installed and run ```pip install -r requirements.txt``` inside the project root folder to install
  python dependencies for this project
- Run django development server
