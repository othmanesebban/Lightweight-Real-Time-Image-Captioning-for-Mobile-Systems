# SeeAround Android Client

This repository contains the Android client application used in the paper:

**Lightweight Real-Time Image Captioning for Mobile Systems: An Optimized Multimodal Framework and Benchmarking Study**

The application integrates image captioning, OCR, translation, and text-to-speech feedback in a mobile assistive pipeline for visually impaired users.

## Main Features

- Android-based assistive vision application
- On-device image caption generation using a frozen TensorFlow ProtoBuf model
- OCR support using mobile OCR backends
- Translation support for multilingual accessibility
- Text-to-speech audio feedback
- Real-time deployment tested on Samsung Galaxy A11
- Support for French, Spanish, Italian, Arabic, and Hindi

## Mobile Deployment Configuration

| Component | Specification |
|---|---|
| Deep learning framework | TensorFlow Android |
| Model format | Frozen ProtoBuf (.pb) |
| Inference runtime | On-device Android inference |
| Input resolution | 299 × 299 × 3 RGB |
| Numerical precision | FP32 |
| Batch size | 1 image |
| Caption length | 18 tokens |
| OCR engines | Google Mobile Vision API, Firebase Vision Text Detector, Google Firebase ML Kit, TessTwo-Android |
| Translation engines | Google Cloud Translation API, Google ML Kit, Firebase ML Kit, Google Translate API |
| Speech synthesis | Android TextToSpeech |
| Supported languages | French, Spanish, Italian, Arabic, Hindi |
| Inference mode | Offline captioning with framework-dependent OCR and translation support |

## Runtime Performance

The deployed application was evaluated on a Samsung Galaxy A11 smartphone.

| Metric | Value |
|---|---|
| Inference latency | 178 ms/image |
| Throughput | 5.6 FPS |
| Memory usage | 410 MB |
| Model parameters | 43M |

## Project Structure

```text
app/
 ├── src/
 │   ├── main/
 │   │   ├── assets/        # Frozen ProtoBuf model and vocabulary/idmap files
 │   │   ├── java/          # Android Java source code in MainActivity.java
 │   │   └── res/           # Layouts, drawables, and UI resources
 ├── build.gradle
 └── AndroidManifest.xml
