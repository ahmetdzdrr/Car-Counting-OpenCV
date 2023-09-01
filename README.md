# Car-Counting-OpenCV

![yolov8-detection-segmentation](https://github.com/ahmetdzdrr/Car-Counting-OpenCV/assets/117534684/cd560baf-fac0-4937-9397-8ca142fe0509)

First of all thank you read my article about this project. Firstly, I'll share a few tricks how you can use tihs project at your environment.
When you download all project files, you'll see requirements.txt file. This file contains some library information and the versions you need to install.
When you installed all libraries, don't forget to install ultralytics. You can install just type to terminal "pip install ultralytics"

****************************************************************************
****************************************************************************

What's the YOLOv8?

YOLOv8 is the latest iteration from the YOLO family of models. YOLO stands for You Only Look Once, and this series of models are thus named because of their ability to predict every object present in an image with one forward pass. 
The main distinction introduced by the YOLO models was the framing of the task at hand. The authors of the paper reframed the object detection task as a regression problem (predict the bounding box coordinates) instead of classification. 
YOLO models are pre-trained on huge datasets such as COCO and ImageNet. This gives them the simultaneous ability to be the Master and the Student. They provide highly accurate predictions on classes they are pre-trained on (master ability) and can also learn new classes comparatively easily (student ability). 

****************************************************************************
****************************************************************************

YOLOv8 Model Architecture

![a280bbfb](https://github.com/ahmetdzdrr/Car-Counting-OpenCV/assets/117534684/cc3c7244-9907-4e66-81ce-459dfaef43a7)

What Are The Enhacements of all YOLO alghoritms?

As there are no official results from the paper, we are going to go through the official YOLO comparison plot from the repository. 

![56bd3724](https://github.com/ahmetdzdrr/Car-Counting-OpenCV/assets/117534684/524ae6d1-5d3a-4f1d-ad07-40a9849a6b01)

As we can observe from the plot, YOLOv8 has more parameters than its predecessors, such as YOLOv5, but fewer parameters than YOLOv6. It offers about 33% more mAP for n-size models and generally a greater mAP across the board. 
From the second graph, we can observe faster inference time amongst all the other YOLO models. This is understandable and elegant. 
Within YOLOv8, we have different model sizes such as yolov8- n - nano, s - small, m - medium, l - large, and x - extra large. 

Which weight is better version in YOLOv8 alghoritm?

![db2573fc](https://github.com/ahmetdzdrr/Car-Counting-OpenCV/assets/117534684/48bed0ce-a9c7-4b6c-bdd4-486c338fff0d)



