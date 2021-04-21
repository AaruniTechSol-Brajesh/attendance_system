import os
import cv2
import requests
import numpy as np
import pandas as pd
import face_recognition
from datetime import date, datetime
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
app = Flask(__name__, template_folder = 'Templates')
@app.route('/', methods = ['GET'])
def Home():    
    return render_template('home.html')
standard_to = StandardScaler()
@app.route('/attendance', methods = ['POST'])
def attendance():
    if request.method == 'POST':
        path = 'Train Images'
        myList = os.listdir(path)
        images, names = [], []
        for n in myList:
            image = cv2.imread(f'{path}/{n}')
            images.append(image)
            names.append(os.path.splitext(n)[0])
        def encodings(images):
            encoded_list = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encoded_list.append(encode)
            return encoded_list
        def log_attendance(name):
            attendance = pd.read_csv('Attendance.csv')
            log = pd.read_csv('Log.csv')
            today = date.today()
            today = today.strftime('%d-%m-%Y')
            time = datetime.now()
            time = time.strftime('%H:%M:%S')
            data_1 = {'Date' : [today], 'Time' : [time], 'Name' : [name]}
            data_1 = pd.DataFrame(data_1)
            log = pd.concat([log, data_1], axis = 0)
            log.to_csv('Log.csv', index = None)
            df_1 = log[(log['Date'] == today) & (log['Name'] == name)]
            in_time = df_1['Time'].min()
            out_time = df_1['Time'].max()
            error = []
            if in_time == out_time:
                error = 'MIS-PUNCH'
            else:
                error = ''
            data_2 = {'Date' : [today], 'Name' : [name], 'In Time' : [in_time], 'Out Time' : [out_time], 'Comment' : [error]}
            data_2 = pd.DataFrame(data_2)        
            df_2 = attendance[attendance['Date'] == today]        
            if df_2['Name'].isin([name]).any():
                ind = attendance.loc[attendance['Name'] == name].index
                attendance.at[ind, 'Out Time'] = out_time
                attendance.at[ind, 'Comment'] = error
                attendance.to_csv('Attendance.csv', index = None)
            else:
                attendance = pd.concat([attendance, data_2], axis = 0)
                attendance.to_csv('Attendance.csv', index = None)
        encoded_images = encodings(images)
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encoded_images, encodeFace)
                faceDis = face_recognition.face_distance(encoded_images, encodeFace)
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = names[matchIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    log_attendance(name)
            cv2.imshow('Webcam', img)
            key = cv2.waitKey(0)
            if key % 256 == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        return render_template('home.html')
    else:
        return render_template('home.html')
if __name__=="__main__":
    #app.run(host = '0.0.0.0', port = 8080)
    app.run(debug = True)