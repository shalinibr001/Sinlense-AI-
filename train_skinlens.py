import os

import shutil

import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

import numpy as np



# Paths (change as needed)

CSV = "HAM10000_metadata (1).csv"   # <-- UPDATED CSV FILENAME

IMG_FOLDER = "HAM10000_images/"

WORK_DIR = "data"

TRAIN_DIR = os.path.join(WORK_DIR, "train")

VAL_DIR = os.path.join(WORK_DIR, "val")

MODEL_DIR = "model"

MODEL_PATH = os.path.join(MODEL_DIR, "skinlens_model.h5")



os.makedirs(MODEL_DIR, exist_ok=True)

os.makedirs(TRAIN_DIR, exist_ok=True)



# Mapping ham10000 dx labels to 4 classes

map4 = {

    'nv': "benign",

    'bkl': "benign",

    'df': "suspicious",

    'vasc': "suspicious",

    'akiec': "precancerous",

    'bcc': "cancerous",

    'mel': "cancerous"

}



# Step 1: read metadata and copy images into train/<class>

df = pd.read_csv(CSV)



# Create class folders

for cls in set(map4.values()):

    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)

    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)



print("Copying images to class folders (this may take a while)...")



for _, row in df.iterrows():

    fname = row['image_id'] + ".jpg"

    src = os.path.join(IMG_FOLDER, fname)

    if not os.path.exists(src):

        continue

    target_cls = map4.get(row['dx'], 'benign')

    dst = os.path.join(TRAIN_DIR, target_cls, fname)

    if not os.path.exists(dst):

        shutil.copy(src, dst)



# Step 2: Split into validation set

print("Creating validation set...")

for cls in os.listdir(TRAIN_DIR):

    cls_folder = os.path.join(TRAIN_DIR, cls)

    files = [f for f in os.listdir(cls_folder) if f.lower().endswith('.jpg')]

    if len(files) < 2:

        continue

    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    for f in val_files:

        src = os.path.join(cls_folder, f)

        dst = os.path.join(VAL_DIR, cls, f)

        shutil.move(src, dst)



# Step 3: Data Generators

IMG_SIZE = (224, 224)

BATCH = 16



train_gen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,

    zoom_range=0.15, horizontal_flip=True

)

val_gen = ImageDataGenerator(rescale=1./255)



train_flow = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')

val_flow = val_gen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')



# Step 4: Build Model

base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')

x = base.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

out = Dense(train_flow.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)



for layer in base.layers:

    layer.trainable = False



model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])



# Step 5: Train

EPOCHS1 = 6

EPOCHS2 = 8



model.fit(train_flow, epochs=EPOCHS1, validation_data=val_flow)



# Fine-tune last layers

for layer in base.layers[-40:]:

    layer.trainable = True



model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_flow, epochs=EPOCHS2, validation_data=val_flow)



# Save model

os.makedirs(MODEL_DIR, exist_ok=True)

model.save(MODEL_PATH)



# Save label mapping

labels = [None] * train_flow.num_classes

for k, v in train_flow.class_indices.items():

    labels[v] = k



np.save(os.path.join(MODEL_DIR, "label_classes.npy"), labels)

print("Saved label_classes.npy:", labels)

print("Training completed successfully!")