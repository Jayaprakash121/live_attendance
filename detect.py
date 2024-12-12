import cv2
import numpy as np
import pandas as pd
import pickle
import os
import time
import csv
from datetime import datetime
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from io import BytesIO
from gtts import gTTS


st.title("Smart Attendance System")
st.write("If your face is not added to the data then add your face first, and then mark your attendance ")
col1, col2, col3 = st.columns(3)
add_face = col1.checkbox('Add Face')
mark_attendance = col2.checkbox('Mark Attendance')
check_attendance = col3.checkbox('Check Attendance')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
m = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

#Adding face to the data
if(add_face):
    name = st.text_input("Enter your name")
    entered = st.checkbox('Name Entered')
    i = 0

while (add_face and entered and i != 100):
    data = []

    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    d = m.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in d:
        crop = frame[y:y + h, x:x + w, :]
        resize = cv2.resize(crop, (50, 50))
        if len(data) <= 100 and i % 10 == 0:
            data.append(resize)
        i += 1
        cv2.putText(frame, str(i), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255,), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 3)
        FRAME_WINDOW.image(frame)

    data = np.array(data)
    data = data.reshape(100, -1)

    if 'names.pkl' not in os.listdir("data/"):
        names = [name] * 100
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * 100
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    if 'data.pkl' not in os.listdir("data/"):
        with open('data/data.pkl', 'wb') as f:
            pickle.dump(data, f)
    else:
        with open('data/data.pkl', 'rb') as f:
            d = pickle.load(f)
        d = np.append(d, data)
        with open('data/data.pkl', 'wb') as f:
            pickle.dump(d, f)

j = 0
#Marking the Attendance
while mark_attendance:
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open('data/data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    LABELS = np.asanyarray(LABELS)
    FACES = np.asanyarray(FACES)

    #if(len(LABELS) != len(FACES)):
    #    np.delete(FACES, len(FACES)-1)

    while (len(LABELS) > len(FACES)):
        LABELS = np.delete(LABELS, len(LABELS) - 1)

    while (len(LABELS) < len(FACES)):
        FACES = np.delete(FACES, len(FACES) - 1)

    FACES = FACES.reshape(-1, 1)
    LABELS = LABELS.reshape(-1, 1)

    #if len(FACES) > len(LABELS):
    #    pca = PCA(n_components=len(LABELS))
    #    FACES = pca.fit_transform(FACES)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    col = ['names', 'time']
    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    d = m.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in d:
        crop = frame[y:y + h, x:x + w, :]
        resize = cv2.resize(crop, (50, 50)).flatten().reshape(1, -1)
        new_resize = np.array(resize[0]).reshape(-1, 1)
        output = knn.predict(new_resize)


        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 3)
        FRAME_WINDOW.image(frame)

        t = time.time()
        date = datetime.fromtimestamp(t).strftime('%Y-%m-%d ')
        times = datetime.fromtimestamp(t).strftime('%H:%M:%S ')
        s = os.path.isfile("attendance/attendance_" + date + ".csv")

        attendance = [str(output[0]), str(times)]

        if (mark_attendance and j==0):
            j = j+1
            #c = st.text_input('Enter o to mark the attendance')
            # c == ord('o'):
             #   speak("attendence taken")
            sound_file = BytesIO()
            tts = gTTS('Attendance Taken', lang='en')
            tts.write_to_fp(sound_file)

            st.audio(sound_file, format='audio/ogg', autoplay=True)

            if s:
                with open("attendance/attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
                csvfile.close()
            else:
                with open("attendance/attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(col)
                    writer.writerow(attendance)
                csvfile.close()

#Checking Attendance
if check_attendance:
    t = time.time()
    date = datetime.fromtimestamp(t).strftime('%Y-%m-%d ')
    times = datetime.fromtimestamp(t).strftime('%H:%M:%S ')
    df = pd.read_csv("attendance/attendance_" + date + ".csv")

    st.dataframe(df.style.highlight_max(axis=0))
