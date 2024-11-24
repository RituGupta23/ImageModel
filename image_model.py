import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Data Preprocessing with Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load Training and Validation Data
train_data = datagen.flow_from_directory(
    r'C:\Users\acer\Downloads\datasets\datasets\Data',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    r'C:\Users\acer\Downloads\datasets\datasets\Data',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# Load Test Data
test_data = test_datagen.flow_from_directory(
    r'C:\Users\acer\Downloads\datasets\datasets\Data',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# Load Pre-trained MobileNetV2 Model
base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Custom Classification Head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

# Evaluate on Validation Data
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Evaluate on Test Data
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model as an H5 file
model.save('image_model.h5')

print("Model saved as 'image_model..h5'")

model_output = model.predict(input_data)
output_shape = model_output.shape
