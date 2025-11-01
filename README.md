# EDITH (Enhanced Detection Intelligent Tech Headwear)

EDITH is an open-source, offline-first assistive device framework. It uses object detection and a wake-word-powered voice interface to enhance accessibility and safety by providing real-time audio feedback about the user's surroundings.

This project is designed to be lightweight, running entirely on-device without a continuous internet connection, making it ideal for portable hardware like a Raspberry Pi.

## 🚀 Core Features

* **Always-On Wake Word:** Uses the Porcupine engine to listen for the "Hey Sid" wake word.
* **Real-Time Object Detection:** Employs a local YOLOv8n model to instantly identify objects in the camera's view.
* **Offline First:** All AI processing (both wake word and object detection) happens 100% on the device.
* **Audio Feedback:** Uses a text-to-speech engine to provide a natural, conversational summary of what it sees (e.g., "I see a person and a remote.").

## 🛠️ Tech Stack

* **Python 3**
* **Porcupine (`pvporcupine`)**: For lightweight, on-device wake word detection.
* **YOLOv8 (`ultralytics`)**: For high-performance, local object detection.
* **OpenCV (`opencv-python`)**: For camera access and image processing.
* **PyAudio**: For capturing microphone input.
* **pyttsx3**: For offline text-to-speech synthesis.

---

## ⚙️ Setup and Installation

Follow these steps to set up the project on your development machine (e.g., a macOS laptop).

### 1. Clone the Repository

```bash
git clone [Your Repository URL]
cd [Your Project Folder]
