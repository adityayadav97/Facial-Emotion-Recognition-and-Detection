import tkinter as tk
from tkinter import ttk
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image, ImageTk

# Load the Haar cascade classifier for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained emotion detection model
classifier = load_model("model.h5")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")

        # Video frame
        self.video_frame = tk.Label(root)
        self.video_frame.pack()

        # Start/Stop button
        self.start_stop_button = ttk.Button(root, text="Start", command=self.toggle_video)
        self.start_stop_button.pack()

        # Emotion label
        self.emotion_label_var = tk.StringVar()
        self.emotion_label_var.set("Emotion: None")
        self.emotion_label = ttk.Label(root, textvariable=self.emotion_label_var)
        self.emotion_label.pack()

        # Quit button
        self.quit_button = ttk.Button(root, text="Quit", command=self.quit)
        self.quit_button.pack()

        # Webcam capture
        self.capture = None
        self.is_capturing = False
        self.update_video()

    def update_video(self):
        if self.is_capturing:
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.detect_emotion(frame)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                self.video_frame.configure(image=frame)
                self.video_frame.image = frame
        self.root.after(10, self.update_video)

    def detect_emotion(self, frame):
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                self.emotion_label_var.set(f"Emotion: {label}")
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                self.emotion_label_var.set("Emotion: None")
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def toggle_video(self):
        if self.is_capturing:
            self.is_capturing = False
            self.start_stop_button.configure(text="Start")
            self.release_capture()
        else:
            self.is_capturing = True
            self.start_stop_button.configure(text="Stop")
            self.capture = cv2.VideoCapture(0)

    def release_capture(self):
        if self.capture:
            self.capture.release()

    def quit(self):
        self.release_capture()
        self.root.quit()

def main():
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
