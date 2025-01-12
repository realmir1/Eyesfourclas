import os #============================================================================
import numpy as np #============================================================================
import tensorflow as tf #============================================================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator #============================================================================
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
#============================================================================
#============================================================================
base_dir = '/kaggle/input/eye-diseases-classification/dataset/'
classes = ['katarakt', 'diyabetik', 'göz tansiyonu (glokom)', 'normal']
image_size = (224, 224)#============================================================================
batch_size = 32#============================================================================
data_gen = ImageDataGenerator(rescale=1.0 / 255,validation_split=0.1)
train_gen = data_gen.flow_from_directory(base_dir, target_size=image_size,batch_size=batch_size,class_mode='categorical',subset='training')
val_gen = data_gen.flow_from_directory(base_dir,target_size=image_size,batch_size=batch_size,class_mode='categorical',subset='validation')
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#============================================================================
x = base_model.output#============================================================================
x = GlobalAveragePooling2D()(x)#============================================================================
x = Dense(128, activation='relu')(x)#============================================================================
output = Dense(len(classes), activation='softmax')(x)#============================================================================
#============================================================================
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_gen,validation_data=val_gen,epochs=10)
#==================================================================
plt.figure()#============================================================================
plt.plot(history.history['accuracy'], label='Training Accuracy')#============================================================================
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')#============================================================================
plt.title('Model Accuracy')#============================================================================
plt.xlabel('Epochs')#============================================================================
plt.ylabel('Accuracy')#============================================================================
plt.legend()#============================================================================
plt.show()#============================================================================
plt.figure()#============================================================================
plt.plot(history.history['loss'], label='Training Loss')#============================================================================
plt.plot(history.history['val_loss'], label='Validation Loss')#============================================================================
plt.title('Model Loss')#============================================================================
plt.xlabel('Epochs')#============================================================================
plt.ylabel('Loss')#============================================================================
plt.legend()#============================================================================
plt.show()#============================================================================
#==================================================================================================================
#============================================================================
val_gen.reset()#============================================================================
predictions = model.predict(val_gen)#============================================================================
predicted_classes = np.argmax(predictions, axis=1)#============================================================================
true_classes = val_gen.classes #============================================================================
print(classification_report(true_classes, predicted_classes, target_names=classes))#============================================================================
sample_images, sample_labels = next(val_gen)#============================================================================
sample_predictions = model.predict(sample_images)#============================================================================
for i in range(10): #============================================================================
    true_label = classes[np.argmax(sample_labels[i])]#============================================================================
    predicted_label = classes[np.argmax(sample_predictions[i])]#============================================================================
    print(f"True Label: {true_label}, Predicted: {predicted_label}")#============================================================================

