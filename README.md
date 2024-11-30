# brains
A small repo for the projects that are done for the brains

## Chronological List of Key Papers in Deep Learning and Computer Vision

This document contains a chronological list of influential papers in the fields of deep learning, object detection, and segmentation, along with their authors and goals.

---

### 1. **LeNet-5 (1998)**  
**Authors**: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
**Goal**: Introduced one of the first convolutional neural networks (CNNs) for digit recognition on the MNIST dataset. It laid the foundation for CNNs used in computer vision tasks.

---

### 2. **AlexNet (2012)**  
**Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton  
**Goal**: Revolutionized image classification by using deep networks (8 layers), ReLU activations, dropout, and GPU acceleration to win the 2012 ImageNet competition, making deep learning mainstream.

---

### 3. **VGGNet (2014)**  
**Authors**: Karen Simonyan, Andrew Zisserman  
**Goal**: Introduced deeper CNN architectures (with small 3x3 filters) that demonstrated the importance of depth in improving performance for image classification.

---

### 4. **GoogLeNet (Inception) (2014)**  
**Authors**: Christian Szegedy, Wei Liu, Yangqing Jia, et al.  
**Goal**: Introduced the **Inception module**, which uses parallel convolution filters of different sizes and reduces computational complexity with 1x1 convolutions, making the network both deeper and more efficient.

---

### 5. **ResNet (2015)**  
**Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
**Goal**: Introduced **residual connections** (skip connections) to facilitate training of very deep networks, addressing the vanishing gradient problem and enabling the development of deeper architectures.

---

### 6. **Fast R-CNN (2015)**  
**Authors**: Ross B. Girshick  
**Goal**: Improved R-CNN by sharing convolutional features and introducing **RoI pooling** to speed up object detection and make the model more efficient.

---

### 7. **Faster R-CNN (2015)**  
**Authors**: Shaoqing Ren, Kaiming He, Ross B. Girshick, Jian Sun  
**Goal**: Introduced **Region Proposal Networks (RPN)** to eliminate the need for external region proposals like Selective Search, resulting in faster and more efficient object detection.

---

### 8. **YOLO: You Only Look Once (2016)**  
**Authors**: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi  
**Goal**: Introduced a unified approach for real-time object detection, where both classification and bounding box prediction are performed in a single forward pass.

---

### 9. **SSD: Single Shot MultiBox Detector (2016)**  
**Authors**: Wei Liu, Andrechen Anguelov, Dingfu Xu, et al.  
**Goal**: Proposed a **single-shot** object detection method, which was faster and more efficient compared to two-stage detectors like Faster R-CNN.

---

### 10. **ResNet-152 (2016)**  
**Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
**Goal**: An extension of ResNet with 152 layers, pushing the boundaries of deeper neural networks to achieve state-of-the-art performance in image classification tasks.

---

### 11. **Mask R-CNN (2017)**  
**Authors**: Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross B. Girshick  
**Goal**: Extended Faster R-CNN to perform **instance segmentation** by adding a mask prediction branch, enabling the segmentation of individual objects in an image.

---

### 12. **DenseNet (2017)**  
**Authors**: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger  
**Goal**: Introduced **dense connections** between layers, improving feature reuse, gradient flow, and parameter efficiency, resulting in better performance with fewer parameters.

---

### 13. **Xception (2017)**  
**Authors**: François Chollet  
**Goal**: Replaced traditional convolutions with **depthwise separable convolutions**, improving both performance and efficiency, and setting the stage for efficient deep learning architectures.

---

### 14. **MobileNet (2017)**  
**Authors**: Andrew G. Howard, Menglong Zhu, Bo Chen, et al.  
**Goal**: Designed an efficient CNN architecture for mobile devices using **depthwise separable convolutions**, significantly reducing computational complexity while maintaining performance.

---

### 15. **RetinaNet (2017)**  
**Authors**: Tsung-Yi Lin, Priya Goyal, Ross B. Girshick, et al.  
**Goal**: Introduced **focal loss** to address class imbalance in dense object detection tasks, making it easier to detect smaller or harder-to-find objects in dense scenes.

---

### 16. **EfficientNet (2019)**  
**Authors**: Mingxing Tan, Quoc V. Le  
**Goal**: Introduced **compound scaling** (scaling depth, width, and resolution) for CNNs, achieving better accuracy and efficiency across a range of tasks and models compared to previous architectures.

---

### 17. **Panoptic FPN (2019)**  
**Authors**: Alexander Kirillov, Ross B. Girshick, Kaiming He, et al.  
**Goal**: Proposed a **Panoptic Feature Pyramid Network (FPN)** that unified **semantic segmentation** and **instance segmentation** into a single framework, improving panoptic segmentation performance.

---

### 18. **UPSNet (2019)**  
**Authors**: Shuang Li, Yuliang Liu, Zhe Wang, et al.  
**Goal**: A unified framework for **panoptic segmentation**, integrating high-quality instance and semantic segmentation tasks in a single model.

---

### 19. **YOLACT (2019)**  
**Authors**: Daniel Bolya, Chongyi Li, Yanghao Li, et al.  
**Goal**: Introduced a **fast instance segmentation framework** that generates segmentation masks efficiently using a **prototype generation network**.

---

### 20. **RegNet (2020)**  
**Authors**: Xingyi Zhou, Jifeng Dai, Xialei Liu, et al.  
**Goal**: Proposed a systematic approach to designing efficient CNN architectures, using **regularized evolutionary methods** to explore optimal designs for scalable networks.

---

### 21. **EfficientDet (2020)**  
**Authors**: Mingxing Tan, Ruoming Pang, Quoc V. Le  
**Goal**: Introduced an efficient and scalable framework for object detection that leverages **compound scaling** (similar to EfficientNet) for both the backbone and detection heads.

---

### 22. **DetectoRS (2020)**  
**Authors**: Huizhong Chen, Zhiqiang Wei, Jun Zhang, et al.  
**Goal**: Proposed the **recursive feature pyramid** and **dynamic convolution** for improved object detection, achieving state-of-the-art results on benchmark datasets.

---

### 23. **DeepLabv3+ (2020)**  
**Authors**: Liang-Chieh Chen, George Papandreou, Ian Kokkinos, et al.  
**Goal**: Improved **panoptic segmentation** by incorporating **atrous separable convolutions** and better feature processing, enhancing segmentation accuracy on complex images.

---
