import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation and preprocessing
datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "dataset/train", target_size=(64, 64), batch_size=32, class_mode='binary', subset="training"
)
val_data = datagen.flow_from_directory(
    "dataset/train", target_size=(64, 64), batch_size=32, class_mode='binary', subset="validation"
)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10, validation_data=val_data)

# Save the trained model
model.save("face_eye_liveness_model.h5")
