# REALME

<p align="center">
  <img src="https://img.shields.io/badge/Deepfake-Detection-blue" alt="Deepfake Detection"/>
</p>

## Project Demo

🎬 **Demo Video:** [demo.mp4](./demo.mp4)

> The `demo.mp4` file demonstrates the SeekReality platform in action, showing the process of uploading media and detecting deepfakes in real time.

---

## Project Structure

```text
Deep Fake Detection Model/
├── demo.mp4                # Demo video of the project in action
├── Project_Report.docx     # Project documentation
├── README.md               # This file
├── Image model training/   # Image model scripts, notebooks, weights
│   ├── Dataset/            # Datasets for training/testing
│   │   ├── Test/
│   │   │   ├── Fake/
│   │   │   └── Real/
│   │   ├── Train/
│   │   └── Validation/
│   ├── models/
│   ├── weights/
│   └── ...
├── SeekReality/
│   ├── backend/            # Flask backend
│   │   ├── main.py
│   │   └── requirements.txt
│   └── frontend/           # React frontend
│       ├── src/
│       └── ...
├── Voice model training/   # Voice model scripts, notebooks
└── ...
```

---

*Document Type: DOCX*

## Table of Contents

  - [**WORK SUMMARY**](#work-summary)
  - [**1. Motivation Behind the Project**](#1-motivation-behind-the-project)
  - [**2. Type of Project**](#2-type-of-project)
    - [**3. Critical Analysis of Research Papers and Technologies Learned**](#3-critical-analysis-of-research-papers-and-technologies-learned)
  - [**Technologies Learned:**](#technologies-learned)
  - [**4. Overall Design of the Project**](#4-overall-design-of-the-project)
  - [**5. Features Built and Programming Languages Used**](#5-features-built-and-programming-languages-used)
  - [**6. Proposed Methodology**](#6-proposed-methodology)
  - [**7. Algorithm/Description of the Work**](#7-algorithmdescription-of-the-work)
  - [**Image-Level Detection (Meso4)**](#image-level-detection-meso4)
  - [** Video-Level Detection (****MesoLSTM****)**](#-video-level-detection-mesolstm)

## **WORK SUMMARY**

SeekReality is a multi-modal deepfake detection platform for images, videos, and audio. It combines state-of-the-art AI models (MesoNet, MesoLSTM) with a modern web interface (React + Flask) and secure authentication (Auth0). The platform is designed for real-time, user-friendly verification of media to combat misinformation.


## **1. Motivation Behind the Project**


SeekReality was motivated by the growing threat of deepfakes, which erode digital trust through sophisticated media manipulation. Key drivers include:

**Misinformation Surge**: AI-generated deepfakes (fake videos, images, audio) threaten journalism, reputations, and security (e.g., voice scams).
**Inadequate Tools**: Existing detection tools are often single-modal, slow, or inaccessible to non-experts.
**Restoring Trust**: SeekReality aims to provide a user-friendly, real-time platform to verify media, empowering users to combat misinformation.

## **2. Type of Project**


Development cum Research Project:

**Hybrid Nature**: Combines software development and AI research within Computer Science Engineering.
**Domain Focus**: Centers on machine learning, computer vision, audio processing, and web development.
**Core Objective**: Develops a multi-modal deepfake detection platform for images, videos, and audio to combat misinformation.
**Academic Context**: Fulfills partial requirements for a Bachelor of Technology degree, emphasizing theoretical and practical contributions.
**Technical Scope**: Integrates AI models (MesoNet, MesoLSTM, ) with a React-Flask web application and Auth0 authentication.
**Practical Application**: Designed for real-world use in journalism, content moderation, and security, enhancing digital trust.
**Dual Purpose**: Balances academic research with a deployable solution addressing technical and societal challenges.

### **3. Critical Analysis of Research Papers and Technologies Learned**

**Research Papers Reviewed**:

**Deepfake Video Detection Based on ****MesoNet**** with Preprocessing Module**: Highlighted MesoNet’s effectiveness with preprocessing for video deepfake detection, but overlooked real-time constraints and multi-modal applicability, limiting its scope for SeekReality’s broader goals.
**A Review on the Long Short-Term Memory Model**: Provided a comprehensive overview of LSTM’s theoretical foundations and applications, emphasizing its role in sequence modeling (e.g., time series, and video processing).
**The Effect of Deep Learning Methods on Deepfake Audio Detection for Digital Investigation**: Explored CNN-based audio detection with spectrogram features, validating SeekReality’s approach, but lacked real-time focus and generalizability, constraining its relevance for scalable deployment.
## **Technologies Learned:**

**MesoNet****/****MesoLSTM**** for Deepfake Detection**: Praised for lightweight, accurate image and video analysis, but computationally intensive for videos and vulnerable to advanced deepfakes, requiring optimization for SeekReality’s real-time needs.
**React and Flask for Web Development**: Lauded for enabling a responsive, modular platform, but Flask’s scalability limitations under high traffic highlight the need for cloud-based enhancements in SeekReality’s architecture.
## **4. Overall Design of the Project**


**Architecture**:

**Frontend**: React.js UI for media uploads, result display, and account management.
**Backend**: Flask server for API handling, preprocessing, and model inference.
**AI Core**: Meso4 (images), MesoLSTM (videos),
**Authentication**: Auth0 with Google Sign-In for secure access.


**Modularity**:


Separate modules for frontend, backend, and AI models enable independent updates.
Standardized preprocessing (face detection, spectrograms) ensures consistency.
**Diagrams**:

Use Case: Shows user interactions (login, upload, view results).
Sequence: Details media upload to result display workflow.
Class: Incorrectly included (healthcare-related); needs SeekReality-specific version.
**Scalability/Security**: Flask supports moderate loads; Auth0 and encryption ensure security. Cloud deployment proposed for scaling.

## **5. Features Built and Programming Languages Used**


**Features**:

**Multi-Modal Detection**: Image (Meso4), video (MesoLSTM), with confidence scores.
**Secure Authentication**: Auth0 with Google Sign-In.
**Explainable Results**: Visual/audio cues and confidence metrics.
**Web Interface**: React portal for uploads, results, and history.
**Real-time Feedback**: Immediate results with progress indicators.
**Reporting**: Downloadable detection reports.
**Languages/Technologies**:


**Python**: Flask, TensorFlow, Pandas, NumPy, OpenCV.
**JavaScript**: React.js for UI.
**HTML/CSS**: Web structure, Tailwind CSS for styling.



## **6. Proposed Methodology**


**Data Collection**: Kaggle datasets

**Preprocessing**:

Images/Videos: Face detection, resize to 256x256, normalize to range [0,1].
Audio: Mel-spectrograms/MFCCs.
**Model Training**:

Meso4: CNN for images.
MesoLSTM: CNN+LSTM for videos.
**Integration**:

React frontend for user interaction.
Flask backend for API and inference.
Auth0 for security.
**Testing**: Unit, integration, system, and user acceptance tests.


**Future Work**: Cloud scaling, social media integration, model tuning.


## **7. Algorithm/Description of the Work**


## **Image-Level Detection (Meso4)**


**Input**: Single image.
**Preprocess**: Resize to 256×256, normalize to [0, 1].
**Prediction**: Pass through Meso4 model.
**Output**: If score > 0.6 → **Real**, else → **Fake**.





## ** Video-Level Detection (****MesoLSTM****)**


**Input**: Video file.
**Frame Extraction**: Select ~30 key frames.
**Preprocess**: Resize and normalize frames.
**Feature Extraction**: Use Meso4 for each frame.
**Temporal ****Modeling**: Feed features into LSTM.
**Output**: If score > 0.6 → **Real**, else → **Fake**.

---

## Getting Started

### Backend (Flask)

```bash
cd SeekReality/backend
pip install -r requirements.txt
python main.py
```

### Frontend (React)

```bash
cd SeekReality/frontend
npm install
npm run dev
```

---

## License

This project is for academic and research purposes.

