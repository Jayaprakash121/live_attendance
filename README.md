# Smart Attendance System

## Overview

The **Smart Attendance System** is a computer vision-based application that automates the process of marking attendance using facial recognition. The system allows users to add their faces, mark attendance in real-time, and check attendance records.

This system utilizes OpenCV for face detection, scikit-learn for machine learning (K-Nearest Neighbors Classifier), and Streamlit for building a web interface.

## Features

- **Add Face**: Users can add their face data to the system, which will be saved for future attendance marking.
- **Mark Attendance**: The system detects faces in real-time and marks attendance if the detected face matches one of the stored faces.
- **Check Attendance**: Users can view the attendance records for the current day.

## Requirements

- Python 3.x
- Libraries:
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `pickle`
  - `streamlit`
  - `scikit-learn`
  - `gtts`
  - `datetime`
  - `csv`

To install the required libraries, run the following command:

pip install opencv-python numpy pandas scikit-learn streamlit gtts

## Setup and Usage

### 1. Clone the Repository

Clone the repository to your local machine:

git clone <repository_url>  
cd <repository_name>

### 2. Run the Application

Start the application using Streamlit by running the following command:

streamlit run app.py

### 3. Adding Faces

- Click on **Add Face** checkbox to start adding a new face.
- Enter your name and click the checkbox once the name is entered.
- The camera will be activated, and the system will capture 100 images of your face for training.

### 4. Marking Attendance

- Click on **Mark Attendance** checkbox to mark your attendance.
- The system will detect faces in real-time and mark attendance when a face is recognized.

### 5. Checking Attendance

- Click on **Check Attendance** to view the attendance record for the current day.
- The attendance data will be displayed in a table.

## Screenshots

### Adding Face:
In this step, the user is adding their face by allowing the system to capture their facial data. The captured image is then processed and stored for future attendance marking.

### Marking Attendance:
After adding the face, the system is marking the user's attendance by detecting the face in real-time.

## Attendance Data Storage

- Face data and corresponding names are stored in `data/data.pkl` and `data/names.pkl`.
- Attendance records are stored in CSV files located in the `attendance/` directory. Each day’s attendance is saved in a separate CSV file with the format `attendance_<date>.csv`.

## File Structure

- `app.py`: The main Python script that runs the application.
- `haarcascade_frontalface_default.xml`: Pre-trained model for face detection.
- `attendance/`: Directory where attendance CSV files are stored.
- `data/`: Directory where face data and names are stored.

