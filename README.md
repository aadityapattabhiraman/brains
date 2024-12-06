# brains
A small repo for the projects that are done for the brains

## Chronological List of Key Papers in Deep Learning and Computer Vision and NLP

---

### Currently doing R-CNN for Object detection

### 1. [LeNet-5 (1998)](https://ieeexplore.ieee.org/document/726791)  
**Authors**: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
**Goal**: Introduced one of the first convolutional neural networks (CNNs) for digit recognition on the MNIST dataset, laying the foundation for modern CNNs.

### 2. [AlexNet (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)  
**Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton  
**Goal**: Revolutionized image classification with a deep neural network (8 layers), ReLU activations, dropout, and GPU acceleration, winning the 2012 ImageNet competition.

### 3. [Word2Vec (2013)](https://arxiv.org/abs/1301.3781)  
**Authors**: Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean  
**Goal**: Introduced **Word2Vec**, a neural network model for learning word embeddings, significantly improving word representations and enabling various NLP applications like sentiment analysis and machine translation.

### 4. [Seq2Seq (2014)](https://arxiv.org/abs/1409.3215)  
**Authors**: Ilya Sutskever, Oriol Vinyals, Quoc V. Le  
**Goal**: Introduced **Sequence-to-Sequence (Seq2Seq)** models with **LSTMs** for tasks like machine translation, enabling models to translate sentences without relying on hand-crafted features.

### 5. [VGGNet (2014)](https://arxiv.org/abs/1409.1556)  
**Authors**: Karen Simonyan, Andrew Zisserman  
**Goal**: Introduced deeper CNNs with small 3x3 filters, demonstrating that depth improves performance in image classification.

### 6. [GoogLeNet (Inception) (2014)](https://arxiv.org/abs/1409.4842)  
**Authors**: Christian Szegedy, Wei Liu, Yangqing Jia, et al.  
**Goal**: Introduced the **Inception module**, using parallel convolution filters of different sizes and reducing computational complexity with 1x1 convolutions.

### 7. [ResNet (2015)](https://arxiv.org/abs/1512.03385)  
**Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
**Goal**: Introduced **residual connections** (skip connections) to enable the training of much deeper networks, solving the vanishing gradient problem.

## 8. [Fast R-CNN (2015)](https://arxiv.org/abs/1504.08083)  
**Authors**: Ross B. Girshick  
**Goal**: Improved R-CNN by sharing convolutional features and introducing **RoI pooling**, speeding up object detection and making it more efficient.

## 9. [YOLO: You Only Look Once (2016)](https://arxiv.org/abs/1506.02640)  
**Authors**: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi  
**Goal**: Proposed a unified approach for real-time object detection, performing both classification and bounding box prediction in a single forward pass.

## 10. [SSD: Single Shot MultiBox Detector (2016)](https://arxiv.org/abs/1512.02325)  
**Authors**: Wei Liu, Andrechen Anguelov, Dingfu Xu, et al.  
**Goal**: Proposed a **single-shot** object detection method that is faster and more efficient than two-stage detectors like Faster R-CNN.

## 11. [SqueezeNet (2016)](https://arxiv.org/abs/1602.07360)  
**Authors**: Forrest N. Iandola, Song Han, Matthew W. Moskewicz, et al.  
**Goal**: Introduced **SqueezeNet**, an ultra-lightweight CNN architecture for image classification, achieving comparable performance to AlexNet while using fewer parameters, ideal for mobile devices.

## 12. [Mask R-CNN (2017)](https://arxiv.org/abs/1703.06870)  
**Authors**: Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross B. Girshick  
**Goal**: Extended Faster R-CNN to perform **instance segmentation** by adding a mask prediction branch, enabling the segmentation of individual objects.

### 13. [Attention is All You Need (2017)](https://arxiv.org/abs/1706.03762)  
**Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.  
**Goal**: Introduced the **Transformer** architecture, revolutionizing NLP by replacing recurrence with self-attention mechanisms, leading to state-of-the-art performance in tasks like machine translation and text generation.

## 14. [DenseNet (2017)](https://arxiv.org/abs/1608.06993)  
**Authors**: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger  
**Goal**: Introduced **dense connections** between layers, improving feature reuse, gradient flow, and parameter efficiency.

## 15. [Xception (2017)](https://arxiv.org/abs/1610.02357)  
**Authors**: François Chollet  
**Goal**: Replaced traditional convolutions with **depthwise separable convolutions**, improving both performance and efficiency, setting the stage for efficient deep learning architectures.

## 16. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://arxiv.org/abs/1810.04805)  
**Authors**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova  
**Goal**: Introduced **BERT** (Bidirectional Encoder Representations from Transformers), a pre-trained language model that achieved state-of-the-art results on a wide range of NLP tasks by leveraging deep bidirectional context.

## 17. [GPT (Generative Pre-trained Transformer) (2018)](https://openai.com/research/language-unsupervised)  
**Authors**: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever  
**Goal**: Introduced **GPT**, a transformer-based generative pre-trained model for language understanding and generation, demonstrating the power of large-scale unsupervised pretraining for NLP tasks.

## 18. [YOLOv3 (2018)](https://arxiv.org/abs/1804.02767)  
**Authors**: Joseph Redmon, Ali Farhadi  
**Goal**: Improved the original YOLO architecture for real-time object detection, enhancing performance by introducing multi-scale predictions and batch normalization.

## 19. [ELMo: Deep Contextualized Word Representations (2018)](https://arxiv.org/abs/1802.05365)  
**Authors**: Matthew Peters, Mark Neumann, Mohit Iyyer, et al.  
**Goal**: Introduced **ELMo**, a deep contextualized word representation model that captures both syntax and semantics by using bi-directional LSTMs over character-level and word-level embeddings, significantly improving the performance of downstream tasks.

## 20. [GPT-2 (2019)](https://openai.com/research/better-language-models)  
**Authors**: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever  
**Goal**: Introduced **GPT-2**, a large transformer-based model for text generation, capable of producing coherent and contextually relevant text over long passages, setting a new benchmark for generative language models.

## 21. [XLNet: Generalized Autoregressive Pretraining for Language Understanding (2019)](https://arxiv.org/abs/1906.08237)  
**Authors**: Zhilin Yang, Mark Dredze, et al.  
**Goal**: Introduced **XLNet**, which combined the benefits of autoregressive and autoencoder models, outperforming BERT on several NLP benchmarks.

## 22. [EfficientNet (2019)](https://arxiv.org/abs/1905.11946)  
**Authors**: Mingxing Tan, Quoc V. Le  
**Goal**: Introduced **compound scaling** for CNNs, scaling depth, width, and resolution simultaneously for better accuracy and efficiency.

## 23. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (2019)](https://arxiv.org/abs/1909.11942)  
**Authors**: Zhenzhong Lan, Mingda Chen, Sebastian Goodman, et al.  
**Goal**: Introduced **ALBERT**, a smaller, more efficient version of BERT, which used parameter sharing and factorized embedding parameterization, achieving state-of-the-art performance on various NLP benchmarks.

## 24. [BART: Denosing Sequence-to-Sequence Pretraining for Natural Language Generation, Translation, and Comprehension (2019)](https://arxiv.org/abs/1910.13461)  
**Authors**: Mike Lewis, Yinhan Liu, Naman Goyal, et al.  
**Goal**: Introduced **BART**, a denoising autoencoder for pretraining sequence-to-sequence models, excelling in both generation and comprehension tasks.

## 25. [Panoptic FPN (2019)](https://arxiv.org/abs/1901.02446)  
**Authors**: Alexander Kirillov, Ross B. Girshick, Kaiming He, et al.  
**Goal**: Proposed a **Panoptic Feature Pyramid Network (FPN)** that unified **semantic segmentation** and **instance segmentation** into a single framework, improving panoptic segmentation performance.

## 26. [UPSNet (2019)](https://arxiv.org/abs/1901.03784)  
**Authors**: Shuang Li, Yuliang Liu, Zhe Wang, et al.  
**Goal**: A unified framework for **panoptic segmentation**, integrating high-quality instance and semantic segmentation tasks into a single model.

## 27. [YOLACT (2019)](https://arxiv.org/abs/1904.02689)  
**Authors**: Daniel Bolya, Chongyi Li, Yanghao Li, et al.  
**Goal**: Introduced a **fast instance segmentation framework** that generates segmentation masks efficiently using a **prototype generation network**.

## 28. [DistilBERT: A Smaller, Faster, Cheaper Version of BERT (2019)](https://arxiv.org/abs/1910.01108)  
**Authors**: Victor Sanh, Lysandre Debut, Julien Chaumond, et al.  
**Goal**: Proposed **DistilBERT**, a smaller and faster version of BERT that maintains 97% of BERT's language understanding with only 60% of its parameters, making it more efficient for deployment in resource-constrained environments.

## 29. [ERNIE: Enhanced Representation through Knowledge Integration (2019)](https://arxiv.org/abs/1904.09223)  
**Authors**: Yu Sun, Shuohuan Wang, Yiming Cui, et al.  
**Goal**: Introduced **ERNIE**, a knowledge-enhanced pretraining model that incorporates world knowledge into pretraining through structured knowledge graphs, improving performance on tasks that require external knowledge.

## 30. [RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019)](https://arxiv.org/abs/1907.11692)  
**Authors**: Yinhan Liu, Myle Ott, Naman Goyal, et al.  
**Goal**: Introduced **RoBERTa**, a variant of BERT that improves pretraining by removing the Next Sentence Prediction task, training with more data, and using larger batch sizes, achieving state-of-the-art performance on several benchmarks.

## 31. [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (2020)](https://arxiv.org/abs/1910.10683)  
**Authors**: Colin Raffel, Noam Shazeer, Adam Roberts, et al.  
**Goal**: Proposed **T5** (Text-to-Text Transfer Transformer), a unified framework where every NLP problem is cast as a text generation task, achieving state-of-the-art results on various benchmarks.

## 32. [EfficientNet-Lite (2020)](https://arxiv.org/abs/2009.07409)  
**Authors**: Mingxing Tan, Ruoming Pang, Quoc V. Le  
**Goal**: An extension of **EfficientNet**, optimized for edge devices, providing a more efficient version of the original EfficientNet for mobile and embedded systems.

## 33. [Reformer: The Efficient Transformer (2020)](https://arxiv.org/abs/2001.04451)  
**Authors**: Nikita Shlezinger, et al.  
**Goal**: Proposed **Reformer**, a transformer model with efficient memory usage and reduced computational complexity, using locality-sensitive hashing (LSH) to speed up attention mechanisms.

## 34. [DetectoRS (2020)](https://arxiv.org/abs/2006.02334)  
**Authors**: Huizhong Chen, Zhiqiang Wei, Jun Zhang, et al.  
**Goal**: Proposed the **recursive feature pyramid** and **dynamic convolution** for improved object detection, achieving state-of-the-art results on benchmark datasets.

## 35. [EfficientDet (2020)](https://arxiv.org/abs/1911.09070)  
**Authors**: Mingxing Tan, Ruoming Pang, Quoc V. Le  
**Goal**: Introduced an efficient and scalable framework for object detection that leverages **compound scaling** for both the backbone and detection heads.

## 36. [DeepLabv3+ (2020)](https://arxiv.org/abs/1802.02611)  
**Authors**: Liang-Chieh Chen, George Papandreou, Ian Kokkinos, et al.  
**Goal**: Improved **panoptic segmentation** by incorporating **atrous separable convolutions**, enhancing segmentation accuracy on complex images.

## 37. [Longformer: The Long-Document Transformer (2020)](https://arxiv.org/abs/2004.05150)  
**Authors**: Iz Beltagy, Matthew E. Peters, Arman Cohan  
**Goal**: Introduced **Longformer**, a transformer-based model optimized for long documents, employing efficient sliding window attention to handle inputs far beyond the usual transformer context window.

## 38. [RegNet (2020)](https://arxiv.org/abs/2003.13678)  
**Authors**: Xingyi Zhou, Jifeng Dai, Xialei Liu, et al.  
**Goal**: Proposed a systematic approach to designing efficient CNN architectures, using **regularized evolutionary methods** to explore optimal designs for scalable networks.

## 39. [DETR: End-to-End Object Detection with Transformers (2020)](https://arxiv.org/abs/2005.12872)  
**Authors**: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, et al.  
**Goal**: Introduced **DETR**, a transformer-based architecture for object detection, eliminating the need for hand-crafted region proposals or post-processing, achieving end-to-end object detection.

## 40. [Turing-NLG: A 17-Billion-Parameter Language Model by Microsoft (2020)](https://arxiv.org/abs/2002.11903)  
**Authors**: Microsoft Research  
**Goal**: Presented **Turing-NLG**, a 17-billion-parameter language model that set new records in natural language generation tasks, such as story generation, summarization, and question-answering, pushing the limits of scale in NLP models.

## 41. [GPT-3: Language Models are Few-Shot Learners (2020)](https://arxiv.org/abs/2005.14165)  
**Authors**: Tom B. Brown, Benjamin Mann, Nick Ryder, et al.  
**Goal**: Proposed **GPT-3**, a large-scale autoregressive language model with 175 billion parameters. It demonstrated strong few-shot learning capabilities, achieving impressive results on a variety of NLP tasks with minimal fine-tuning.

## 42. [YOLOv4 (2020)](https://arxiv.org/abs/2004.10934)  
**Authors**: Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao  
**Goal**: Enhanced the original YOLO architecture, adding a range of improvements including better performance on small objects and more effective use of multi-scale predictions.

## 43. [Swin Transformer (2021)](https://arxiv.org/abs/2103.14030)  
**Authors**: Ze Liu, et al.  
**Goal**: Introduced **Swin Transformer**, a hierarchical vision transformer that uses a shifted window approach for efficient computation, achieving state-of-the-art results in image classification and dense prediction tasks.

## 44. [Swin-UNETR (2021)](https://arxiv.org/abs/2105.05537)  
**Authors**: Yihao Liu, et al.  
**Goal**: A **Swin Transformer-based architecture** for medical image segmentation, showing how transformers can be adapted for fine-grained segmentation tasks.
















