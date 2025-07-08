# Additional comments for clarity and learning:
# This script demonstrates transfer learning, which means using a pre-trained model (ResNet50) as a starting point for your own task.
# Data augmentation is used to artificially increase the diversity of the training dataset, helping the model generalize better.
# The model is trained in two phases: first with most layers frozen (to preserve learned features), then with more layers unfrozen (fine-tuning).
# The use of callbacks like EarlyStopping and ReduceLROnPlateau helps prevent overfitting and makes training more efficient.
# The final model is saved to disk for later use in predictions or deployment.
# This script trains a Convolutional Neural Network (CNN) for multi-label image classification using Keras and TensorFlow.
# It uses transfer learning with ResNet50, data augmentation, and includes best practices for training and evaluation.
#optimised version uses cnn-rnn with spatial lstm for multilabel image classification
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# PARAMETERS AND DATASET SETUP
# -----------------------------
# Image dimensions expected by ResNet50
IMG_HEIGHT, IMG_WIDTH = 224, 224
# Number of output classes (plastic, metal, biomedical, shoes)
NUM_CLASSES = 4
# Number of images per batch during training
BATCH_SIZE = 32  # Larger batch size can help with training stability
# Number of training epochs (full passes through the data)
EPOCHS = 30
# Path to your dataset directory (update as needed)
DATASET_DIR = 'c:/proj1/dataset/'

# -----------------------------
# CUSTOM MULTI-LABEL GENERATOR
# -----------------------------
# This function wraps a Keras generator to allow for multi-label (not just one-hot) outputs.
def convert_to_multilabel(generator):
    while True:
        images, labels = next(generator)
        # Convert one-hot to multi-label: soften labels for partial class membership
        labels = np.where(labels > 0.5, 1, labels * 0.5)  # Allows for partial multi-label
        yield images, labels

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
# ImageDataGenerator applies random transformations to images to help the model generalize.
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    validation_split=0.2,  # Reserve 20% of data for validation
    rotation_range=20,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Randomly flip images
    fill_mode='nearest'  # Fill in new pixels after transformations
)

# Create generators for training and validation data
train_generator_base = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # One-hot encoded labels
    subset='training',
    classes=['plastic', 'metal', 'biomedical', 'shoes'],
    shuffle=True
)

validation_generator_base = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    classes=['plastic', 'metal', 'biomedical', 'shoes'],
    shuffle=True  # Shuffle validation data as well
)

# Wrap the generators to allow for multi-label outputs
train_generator = convert_to_multilabel(train_generator_base)
validation_generator = convert_to_multilabel(validation_generator_base)

# -----------------------------
# MODEL BUILDING: TRANSFER LEARNING
# -----------------------------
# Use ResNet50 pre-trained on ImageNet as the base model (excluding its top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
# Add global average pooling to reduce feature maps to a vector
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a dense layer with L2 regularization to prevent overfitting
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
# Dropout layer for further regularization
x = Dropout(0.3)(x)
# Output layer: one sigmoid per class for multi-label classification
predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# -----------------------------
# FINE-TUNING STRATEGY
# -----------------------------
# Freeze most of the base model layers to retain pre-trained features
for layer in base_model.layers[:-50]:  # Unfreeze only the last 50 layers
    layer.trainable = False

# Compile the model with Adam optimizer and binary crossentropy loss (for multi-label)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True)]
)

# -----------------------------
# CALLBACKS FOR TRAINING
# -----------------------------
# ReduceLROnPlateau: lower learning rate if validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
# EarlyStopping: stop training if validation loss doesn't improve
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# -----------------------------
# INITIAL TRAINING (FROZEN BASE)
# -----------------------------
# Train the model with most of the base frozen
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator_base.samples // BATCH_SIZE),
    epochs=15,  # Initial training for 15 epochs
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator_base.samples // BATCH_SIZE),
    callbacks=[reduce_lr, early_stopping]
)

# -----------------------------
# FINE-TUNING (UNFREEZE MORE LAYERS)
# -----------------------------
# Unfreeze more layers for fine-tuning
for layer in base_model.layers[-70:]:  # Unfreeze last 70 layers
    layer.trainable = True

# Re-compile with a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True)]
)

# Continue training (fine-tuning)
history_fine = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator_base.samples // BATCH_SIZE),
    epochs=EPOCHS,  # Continue up to EPOCHS
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator_base.samples // BATCH_SIZE),
    initial_epoch=history.epoch[-1] + 1,  # Resume from where previous training stopped
    callbacks=[reduce_lr, early_stopping]
)

# -----------------------------
# SAVE THE TRAINED MODEL
# -----------------------------
model.save('c:/proj1/waste_cnn_model.h5')
print("Model saved as 'c:/proj1/waste_cnn_model.h5'")

# -----------------------------
# PLOT TRAINING HISTORY
# -----------------------------
# Plot accuracy curves for both training and validation
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot AUC (Area Under Curve) for both training and validation
plt.plot(history.history['auc'] + history_fine.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'] + history_fine.history['val_auc'], label='Validation AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()

# -----------------------------
# FINAL EVALUATION
# -----------------------------
# Evaluate the model on the validation set (using the base generator for true labels)
val_loss, val_accuracy, val_auc = model.evaluate(validation_generator_base)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Validation AUC: {val_auc}")