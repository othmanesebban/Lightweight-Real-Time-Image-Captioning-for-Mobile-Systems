# Lightweight-Real-Time-Image-Captioning-for-Mobile-Systems App for Visually Impaired
A lightweight real-time image captioning framework for mobile systems with efficient deep learning deployment and assistive vision applications.

# Lightweight Real-Time Image Captioning for Mobile Systems

This repository provides the implementation resources associated with the paper:

**Lightweight Real-Time Image Captioning for Mobile Systems: An Optimized Multimodal Framework and Benchmarking Study**

The project includes image captioning models based on CNN-RNN encoder-decoder architectures and deployment resources for the SeeAround Android assistive application.

## Main Features

- Image captioning using InceptionV3-GRU, ResNet50-GRU, and InceptionV4-LSTM architectures
- Caption-length optimization from 22 to 18 tokens
- MSCOCO 2017 training and evaluation pipeline
- Frozen ProtoBuf model export for Android deployment
- OCR, translation, and text-to-speech integration in the SeeAround mobile application
- Real-time mobile testing on Samsung Galaxy A11

## Main Results

The optimized InceptionV4-LSTM model achieved:

- BLEU-4: 0.4189
- ROUGE-L: 0.688
- SPICE: 0.0599
- Inference time: 178 ms/image
- Throughput: 5.6 FPS
- Memory usage: 410 MB
- Composite operational score: 92.6%

## Dataset

The models were trained and evaluated using MSCOCO 2017.

Download the dataset from:

https://cocodataset.org
