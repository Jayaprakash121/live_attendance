# Smart Attendance System

## Overview

The **Smart Attendance System** is a computer vision-based application that automates the process of marking attendance using facial recognition. The system allows users to add their faces, mark attendance in real-time, and check the attendance records.

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

```bash
pip install opencv-python numpy pandas scikit-learn streamlit gtts
