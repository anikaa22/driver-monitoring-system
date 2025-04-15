#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.listdir())  


# In[2]:


import zipfile

zip_path = "StateFarmDataset.zip"  
extract_path = "StateFarmDataset"  

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction Successful!")


# In[3]:


get_ipython().system('pip install opencv-python')


# In[4]:


import os
import shutil


original_path = "StateFarmDataset/State Farm Dataset/imgs/train"
filtered_path = "StateFarmDataset/filtered_phone_usage"


class_mapping = {
    "Safe": ["c0"],
    "PhoneUsage": ["c1", "c2", "c3", "c4"]
}


for new_class, original_classes in class_mapping.items():
    dest_dir = os.path.join(filtered_path, new_class)
    os.makedirs(dest_dir, exist_ok=True)

    for orig_class in original_classes:
        src_dir = os.path.join(original_path, orig_class)
        if not os.path.exists(src_dir):
            print(f"⚠️ Source folder missing: {src_dir}")
            continue

        files = os.listdir(src_dir)
        for file in files:
            src_file = os.path.join(src_dir, file)
            dest_file = os.path.join(dest_dir, file)

            try:
                shutil.copy2(src_file, dest_file)
            except Exception as e:
                print(f"❌ Error copying {file}: {e}")

print("✅ Phone usage dataset created at:", filtered_path)


# In[5]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os


train_path = 'StateFarmDataset/filtered_phone_usage'


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80-20 split
)


train_data = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
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
    Dense(2, activation='softmax')  
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(train_data, validation_data=val_data, epochs=10)


model.save('phone_usage_model_noweights.h5')


# In[1]:


import cv2
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('phone_usage_model_noweights.h5')


class_labels = ['PhoneUsage', 'Safe']


img_size = (224, 224)


def test_phone_usage_with_video(video_source=0):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(" Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" End of video or can't read frame.")
            break

        
        resized_frame = cv2.resize(frame, img_size)
        normalized_frame = resized_frame / 255.0
        input_tensor = np.expand_dims(normalized_frame, axis=0)

        
        prediction = model.predict(input_tensor)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        label = class_labels[predicted_class]

        
        if label == "PhoneUsage":
            display_text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

       
        cv2.imshow("Phone Usage Detection", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Exiting.")
            break

    
    cap.release()
    cv2.destroyAllWindows()

# Run the function to test on webcam (video_source=0)
test_phone_usage_with_video(video_source=0)



# In[ ]:




