#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.listdir())  # Lists all files in the current directory


# In[2]:


import zipfile

zip_path = "StateFarmDataset.zip"  # Use the new name
extract_path = "StateFarmDataset"  # Destination folder

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction Successful!")


# In[3]:


get_ipython().system('pip install opencv-python')


# In[4]:


import os
import shutil


original_path = "StateFarmDataset/State Farm Dataset/imgs/train"
filtered_path = "StateFarmDataset/filtered_drinking"


class_mapping = {
    "Safe": ["c0"],
    "Drinking": ["c6"]
}


for new_class, original_classes in class_mapping.items():
    dest_dir = os.path.join(filtered_path, new_class)
    os.makedirs(dest_dir, exist_ok=True)

    for orig_class in original_classes:
        src_dir = os.path.join(original_path, orig_class)
        if not os.path.exists(src_dir):
            print(f" Source folder missing: {src_dir}")
            continue

        files = os.listdir(src_dir)
        for file in files:
            src_file = os.path.join(src_dir, file)
            dest_file = os.path.join(dest_dir, file)

            try:
                shutil.copy2(src_file, dest_file)
            except Exception as e:
                print(f" Error copying {file}: {e}")

print(" Drinking detection dataset created at:", filtered_path)


# In[5]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os


filtered_path = "StateFarmDataset/filtered_drinking"


img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
   rescale=1./255,
   validation_split=0.2,
   rotation_range=10,
   zoom_range=0.1,
   horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
   filtered_path,
   target_size=img_size,
   batch_size=batch_size,
   class_mode='categorical',
   subset='training'
)
print(train_generator.class_indices)
val_generator = train_datagen.flow_from_directory(
   filtered_path,
   target_size=img_size,
   batch_size=batch_size,
   class_mode='categorical',
   subset='validation'
)


model = Sequential([
   Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
   MaxPooling2D(2, 2),
   
   Conv2D(64, (3, 3), activation='relu'),
   MaxPooling2D(2, 2),
   
   Conv2D(128, (3, 3), activation='relu'),
   MaxPooling2D(2, 2),
   
   Flatten(),
   Dense(128, activation='relu'),
   Dropout(0.5),
   Dense(2, activation='softmax')  # 2 classes: Safe, Drinking
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
   train_generator,
   epochs=5,
   validation_data=val_generator
)


model.save("drinking_detection_model.h5")

print("✅ Model trained and saved as drinking_detection_model.h5")


# In[1]:


import cv2
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('drinking_detection_model.h5')


class_labels = ['Drinking', 'Safe']


img_size = (224, 224)

def test_drinking_with_webcam():
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🚫 Error: Couldn't read the frame.")
            break

        
        resized_frame = cv2.resize(frame, img_size)
        normalized_frame = resized_frame / 255.0
        input_tensor = np.expand_dims(normalized_frame, axis=0)

       
        prediction = model.predict(input_tensor)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        label = class_labels[predicted_class]

       
        if label == "Drinking":
            display_text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

       
        cv2.imshow("Drinking Detection (Webcam)", frame)

        # Exit on 'q' 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Exiting.")
            break

    
    cap.release()
    cv2.destroyAllWindows()


test_drinking_with_webcam()


# In[ ]:




