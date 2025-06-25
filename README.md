# iDetect-Source (Internal Development Repository)

# iDetect: An End-to-End Deep Learning System for Video Action Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)
![Status](https://img.shields.io/badge/Status-Completed-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-Private%20Code-red?style=for-the-badge)

</div>

<br>

**Note: This is the private, internal repository for team collaboration. The public-facing showcase can be found [here](https://github.com/jahnavikonatala21/iDetect).**

---

## 🚀 Overview

This is the central development repository for the **iDetect** Action Recognition project. All code, experiments, and collaboration happen here.

## ⚙️ Setup Guide

This guide is for authorized team members to set up the project on their local machines.

### **Step 1: Clone the Repository**

**Prerequisite:** Ensure the repository owner has invited you as a collaborator and you have accepted the invitation via email.

These commands should be run in your terminal (like Command Prompt, PowerShell, Git Bash, or the macOS Terminal).

1.  **Clone the repository:** This will download the project to your computer. You will need to authenticate using your GitHub username and a **Personal Access Token (PAT)** as your password.
    ```bash
    git clone https://github.com/jahnavikonatala21/iDetect-Source.git
    ```

2.  **Navigate into the project directory:** After cloning, you must move into the newly created folder.
    ```bash
    cd iDetect-Source
    ```

> **Authentication Note:** If you don't have a Personal Access Token (PAT), you must create one from your GitHub account settings. It is required for command-line access to private repositories. Please follow the official guide: [**Creating a PAT**](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token). Make sure to give the token the `repo` scope.

### **Step 2: Set Up the Python Environment**

These commands set up an isolated environment for the project's dependencies.

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    *   On Windows Command Prompt:
        ```cmd
        venv\Scripts\activate
        ```
    *   On macOS or Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install all required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### **Step 3: Add Required Assets (Manually)**

Our large files are not stored in Git. You must add them from our team's shared drive.

1.  **Download the Trained Model:**
    -   Go to our shared Google Drive folder: **{https://drive.google.com/file/d/1_MQDQ6qrmRZFDq1bfO5qFNczzL5EQk8p/view?usp=sharing)**
    -   Download the latest `lstm_model_best_val_acc.pt` file.
    -   Place it inside the `models/` directory in your local project folder.

2.  **Download the Dataset (Only if training from scratch):**
    -   Download the `UCF-101.zip` from our shared drive, unzip it, and place the contents into a `data/UCF-101/` directory.

### **Step 4: Run the Application Locally**

With your environment activated and the model file in place, you can start the web app.

```bash
python backend/app.py

```
## ✨ Key Features

-   **High-Fidelity Action Classification:** A custom-trained **Bidirectional LSTM** network provides robust classification across 101 action categories from the UCF-101 dataset.
-   **Efficient Transfer Learning:** Utilizes a pre-trained **ResNet-50** for powerful visual feature extraction, drastically reducing training time and resource requirements.
-   **Interactive & Responsive UI:** A modern and user-friendly frontend built with HTML, CSS, and JavaScript allows for easy video uploads and displays results dynamically.
-   **Insightful Predictions:** The application provides the **Top 3** most likely actions with their confidence scores, offering a deeper insight into the model's decision-making process.
-   **End-to-End Pipeline:** Demonstrates a complete, modular workflow from raw data ingestion to a deployed, user-facing application.

---

## 🛠️ Technology Stack & Dataset

| Category          | Technology / Resource                                          | Purpose                                                      |
| :---------------- | :------------------------------------------------------------- | :----------------------------------------------------------- |
| **Backend**       | [Python](https://www.python.org/)                              | Core programming language for the entire pipeline.           |
| **Deep Learning** | [PyTorch](https://pytorch.org/)                                | Building, training, and deploying the neural network models. |
| **Web Framework** | [Flask](https://flask.palletsprojects.com/)                    | Serving the backend API and the frontend application.        |
| **CV & Numerics** | [OpenCV](https://opencv.org/)                                  | Video processing and frame extraction.                       |
|                   | [NumPy](https://numpy.org/)                                    | Efficient numerical computation and data manipulation.       |
| **Frontend**      | HTML5, CSS3, JavaScript                                        | Creating the interactive user interface.                     |
| **Dataset**       | [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)             | The benchmark dataset used for training and validation.      |

---

## 🏛️ System Architecture

The iDetect system is built on a powerful two-stage architecture, visually represented below. This design separates spatial understanding from temporal analysis for maximum efficiency and performance.
![image](https://github.com/user-attachments/assets/fa32d6bf-3219-42ee-a68c-b6c7313a27db)

## 🔬 How It Works: The Complete Project Workflow

The iDetect system is the result of a comprehensive, four-phase machine learning pipeline. Here’s a detailed breakdown of the entire process from data to a live prediction.

### Phase 1: Data Preparation and Labeling

The foundation of the project is built on well-structured data.

1.  **Dataset:** The system is trained on the **UCF-101 dataset**, a widely used benchmark containing over 13,000 video clips across 101 action categories.
2.  **Class Mapping:** A script first creates a `classInd.txt` map, assigning a unique numerical ID to each action name (e.g., `1 Archery`). This is essential for the model to understand the labels during training.

### Phase 2: Offline Feature Extraction ("Smart Preprocessing")

This is the most computationally intensive phase, performed once before training to convert every video into a lightweight, information-rich format.

1.  **Frame Extraction:** For each video, **16 frames** are extracted at even intervals and resized to 224x224 pixels, capturing a snapshot of the action's progression.
2.  **Deep Feature Extraction:** These frames are passed through a pre-trained **ResNet-50** Convolutional Neural Network. By using the network without its final classification layer, it becomes a powerful feature extractor.
3.  **Vector Generation:** For each frame, the ResNet-50 outputs a **2048-dimensional feature vector**—a dense numerical representation of that frame's visual information.
4.  **Feature Caching:** The resulting sequence of 16 feature vectors for each video is saved to disk as a compact `.npy` file. This offline caching dramatically speeds up the training phase, as the model can load these small files instead of repeatedly processing raw videos.

### Phase 3: Model Training and Validation

This is where the core learning happens.

1.  **Data Loading:** The system loads all pre-computed `.npy` feature files and splits them into a training set (~80%) and a validation set (~20%).
2.  **Model Architecture:** We defined our custom **Bidirectional LSTM (Long Short-Term Memory)** model in PyTorch. This architecture is specifically designed to find patterns in sequences. Its "bidirectional" nature allows it to process the sequence both forwards and backwards, granting it a deeper understanding of temporal context.
3.  **Training Loop:** The model is trained for 50 epochs using **Cross-Entropy Loss** as the objective function and the **Adam Optimizer** to adjust the model's weights and improve its accuracy.
4.  **Validation & Model Selection:** After each training epoch, the model's performance is tested on the unseen validation set. If the model achieves a new highest accuracy, its state is saved. This ensures we deploy only the best-performing version.

### Phase 4: Live Inference in the Web Application

This describes what happens when a user uploads a video.

1.  **File Upload:** A user uploads a video through the web interface to the **Flask** backend.
2.  **On-the-Fly Pipeline:** The server instantly performs the **exact same feature extraction pipeline** from Phase 2 on the user's video: it extracts 16 frames, passes them through the ResNet-50, and generates a sequence of feature vectors.
3.  **Prediction:** This new sequence is fed into our loaded, pre-trained **Bidirectional LSTM model**.
4.  **Displaying Results:** The model outputs probabilities for all 101 classes. The system identifies the **top 3 predictions**, formats them with their confidence scores, and sends them back to the user's browser. The frontend JavaScript then dynamically renders the results on the page.

## 🤝 Our Team

This project was brought to life by a dedicated and collaborative team. For professional inquiries, please feel free to reach out.

-   **Konatala Jahnavi Sri Harshita**
    -   *Role:* Team Lead, Lead Architect & System Integrator
    -   *Profiles:* [GitHub](https://github.com/jahnavikonatala21) | [LinkedIn](https://www.linkedin.com/in/jahnavi-konatala-6a533a255/)

-   **Indugapalli Satya Asritha**
    -   *Role:* Core Contributor & Model Development
    -   *Profiles:* [GitHub](https://github.com/asritha1102) | [LinkedIn](http://linkedin.com/in/asritha33)

-   **A.Satya Vaishnavi**
    -   *Role:* Core Contributor & Data Pipeline Management
    -   *Profiles:* [GitHub](https://github.com/satyavaishnavi03) | [LinkedIn](https://www.linkedin.com/in/YOUR_LINKEDIN_PROFILE)

-   **Md.Ameer Unnisa**
    -   *Role:* Core Contributor & Frontend Development
    -   *Profiles:* [GitHub](https://github.com/2003-sonu) | [LinkedIn](https://www.linkedin.com/in/mohammad-amerunnisa-2a90ab259)
