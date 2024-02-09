#importing the necessary libraires 
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.applications import VGG16
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from sklearn.metrics import cohen_kappa_score, classification_report

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Function to create a transfer learning model with VGG16
def create_transfer_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation="softmax"))

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def create_cnn_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                     activation='relu', input_shape=(48, 48, 3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation="softmax"))

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def train_model(model, training_set, validation_set, epochs=30, batch_size=32):

    print(f"\nTraining of {str(model)}")

    history = model.fit(
        training_set,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_set,
        shuffle=True)

    return history

def plot_training_results(history, model):
    fig, ax = plt.subplots(1, 2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    fig.set_size_inches(12, 4)

    # Plot Training and Validation Accuracy
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title(f'{model} Training Accuracy vs Validation Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot Training and Validation Loss
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title(f'{model} Training Loss vs Validation Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')

    plt.show()

train_datagen = ImageDataGenerator(
    rescale = 1./255,   ## rescale or normalize the images pixels, by dividing them 255
    shear_range = 0.2,  ## angle for slant of image in degrees
    zoom_range = 0.2,   ## for zoom in or out
    horizontal_flip = True 
)

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    r'Shuffle\Train',   ## give path of training set
    target_size=(48,48),      ## target_size of image in which you want
    batch_size=32,
    color_mode = "rgb",
    class_mode = 'categorical'
)

validation_set = validation_datagen.flow_from_directory(
    r'Shuffle\Validation',
    target_size = (48,48),
    batch_size = 32,
    color_mode = "rgb",
    class_mode = 'categorical'
)

test_set = test_datagen.flow_from_directory(
    r'Shuffle\Test',
    target_size = (48,48),
    batch_size = 32,
    color_mode = "rgb",
    class_mode = 'categorical',
    shuffle=False
)

print("\n")

# Create the model
cnn_model = create_cnn_model()
vgg16_model = create_transfer_model()

# Train the model
epochs = 30
cnn_training_history = train_model(cnn_model, training_set, validation_set, epochs=epochs)
vgg16_training_history = train_model(vgg16_model, training_set, validation_set,epochs=epochs)

plot_training_results(history=cnn_training_history, model='CNN')
plot_training_results(history=vgg16_training_history, model='vgg16')

print("\nSummary of training")

# Make predictions on the test data
cnn_y_pred_probs = cnn_model.predict(test_set)
vgg16_y_pred_probs = vgg16_model.predict(test_set)

cnn_pred_labels = cnn_y_pred_probs.argmax(axis=1)
vgg16_pred_labels = vgg16_y_pred_probs.argmax(axis=1)

# Get true labels from the generator
y_true = test_set.classes

cnn_kappa = cohen_kappa_score(y_true, cnn_pred_labels)
print(f"\nCNN Cohen's Kappa: {cnn_kappa:.2f}")

vgg16_kappa = cohen_kappa_score(y_true, vgg16_pred_labels)
print(f"vgg16 Cohen's Kappa: {vgg16_kappa:.2f}")

# Generate classification report
cnn_report = classification_report(y_true, cnn_pred_labels)
print("\nCNN Classification Report:")
print(cnn_report)

vgg16_report = classification_report(y_true, vgg16_pred_labels)
print("vgg16 Classification Report:")
print(vgg16_report)