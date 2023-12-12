import os
import cv2
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
logging.getLogger("tensorflow").setLevel(logging.ERROR) # Suppress TensorFlow logs

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from math import ceil
import numpy as np
import string
import random

MODEL_FILE = R'C:\Projects\Hochschulsport-Bot\src\captcha\neural_net\captcha_nn.keras'
CHARACTERS = string.ascii_uppercase + string.digits

def has_glyph(font, glyph) -> bool:
    for table in font['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False

def get_windows_fonts():
    fonts_dir = 'C:\\Windows\\Fonts'
    font_paths = [os.path.join(fonts_dir, font) for font in os.listdir(fonts_dir) if font.endswith('.ttf')]
    return font_paths

def get_supported_fonts(characters):
    
    fonts = [
        'arial.ttf',
        'arialbd.ttf',
        'arialbi.ttf',
        'ariali.ttf',
        'cour.ttf',
        'courbd.ttf',
        'courbi.ttf',
        'couri.ttf',
        'times.ttf',
        'timesbd.ttf',
        'timesbi.ttf',
        'timesi.ttf',
        'verdana.ttf',
        'verdanab.ttf',
        'verdanai.ttf',
        'verdanaz.ttf',
        'calibri.ttf',
        'calibrib.ttf',
        'calibrii.ttf',
        'calibriz.ttf',
        'cambria.ttc',
        'cambriab.ttf',
        'cambriai.ttf',
        'cambriaz.ttf',
        'comic.ttf',
        'comicbd.ttf',
    ]
    
    yield from [f'C:\\Windows\\Fonts\\{font}' for font in fonts]
    
    return
    
    
    for font_path in get_windows_fonts():
        font = TTFont(font_path)
        for char in characters:
            if not has_glyph(font, char):
                print(f"Font {font_path} does not have char: {char}")
                break
        else:
            yield font_path

def create_synthetic_data(size=5000, image_size=(35, 35), rotation_range=45):# -> tuple[ndarray[Any, dtype[floating[Any]]], NDArray]:
    fonts = [ImageFont.truetype(font, 30) for font in get_supported_fonts(CHARACTERS)]
    print(f"Using {len(fonts)} fonts")

    X = []
    Y = []
    for _ in range(size):
        img = Image.new('L', image_size, 'black')
        draw = ImageDraw.Draw(img)
        char = random.choice(CHARACTERS)
        font = random.choice(fonts)
        
        # Approximate centering
        text_x = (image_size[0] - random.uniform(28, 32)) // 2  # Approximate x position
        text_y = (image_size[1] - random.uniform(28, 32)) // 2  # Approximate y position
        draw.text((text_x, text_y), char, 'white', font=font)
        
        img = img.rotate(random.uniform(-rotation_range, rotation_range))
        X.append(np.array(img))
        Y.append(CHARACTERS.index(char))

    X = np.array(X) / 255.0  # Normalize
    X = X.reshape(-1, 35, 35, 1)  # Reshape for CNN
    return X, np.array(Y)

def pad_image_to_size(image, target_size=(35, 35)):
    pad_width = (target_size[0] - image.shape[0]) // 2 + 1
    pad_height = (target_size[1] - image.shape[1]) // 2 + 1
    
    padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    if padded_image.shape[0] > 35 or padded_image.shape[1] > 35:
        padded_image = cv2.resize(padded_image, (35, 35), interpolation=cv2.INTER_NEAREST)
        
    return padded_image

def load_and_prepare_mnist(dataset, target_size=(35, 35)):
    (X_mnist, Y_mnist) = dataset
    X_mnist = X_mnist / 255.0  # Normalize

    # Resize and pad MNIST images
    padded_images = [
        pad_image_to_size(img, target_size)
        for img in X_mnist
    ]

    X_mnist_padded = np.array(padded_images).reshape(-1, 35, 35, 1)  # Reshape for CNN
    
    label_indices = [
        CHARACTERS.index(str(label))
        for label in Y_mnist
    ] 
    return X_mnist_padded, np.array(label_indices)

def combine_datasets(X_synthetic, Y_synthetic, X_mnist, Y_mnist):
    # Combine the datasets
    X_combined = np.concatenate((X_synthetic, X_mnist), axis=0)
    Y_combined = np.concatenate((Y_synthetic, Y_mnist), axis=0)
    return X_combined, Y_combined


def get_training_data(synthetic_size=100000):
    
    X_synthetic, Y_synthetic = create_synthetic_data(size=synthetic_size)
    train, test = mnist.load_data()
    X_mnist, Y_mnist = load_and_prepare_mnist(train)
    X_combined, Y_combined = combine_datasets(X_synthetic, Y_synthetic, X_mnist, Y_mnist)
        
    X_thresh = []
    for img in X_combined:
        _, thresh = cv2.threshold(img, 0.3, 1, cv2.THRESH_BINARY)
        X_thresh.append(thresh)
        
    X_thresh = np.array(X_thresh)
        
    # run a 5x5 kernel over the image and only keep the pixel if the sum of the kernel is > 0.4
    # for _ in range(5):
    #     kernel = np.ones((2, 2), np.float32) / 4
    #     img_filtered = cv2.filter2D(X_thresh, -1, kernel)
    #     X_thresh[img_filtered < (1 * 0.26)] = 0
        
    # remove all data points that are completely black
    X_out, Y_out = [], []
    for x, y in zip(X_thresh, Y_combined):
        if np.sum(x) > 0:
            X_out.append(x)
            Y_out.append(y)    
    
    return np.array(X_out), np.array(Y_out)


def indices_to_labels(indices):
    return np.array([CHARACTERS[index] for index in indices])

def predict(model, images):
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)

    return indices_to_labels(predicted_labels), confidence_scores


def test_model(model):
    X_test, Y_test = get_training_data(synthetic_size=60000)        
    predicted_labels, confidence_scores = predict(model, X_test)
    
    Y_test = indices_to_labels(Y_test)
    correct_predictions = np.sum(predicted_labels == Y_test)
    
    print(f"Correct predictions: {correct_predictions} / {len(Y_test)}")
    print(f"Accuracy: {correct_predictions / len(Y_test) * 100:.2f}%")
    
    if False:
        indices = np.random.choice(len(X_test), size=200, replace=False)
        X_test = X_test[indices]
        Y_test = Y_test[indices]
        predicted_labels = predicted_labels[indices]
        confidence_scores = confidence_scores[indices]
    else:
        # choose 10 indices where Y_test in Y2AF8
        X_final = []
        Y_final = []
        predicted_labels_final = []
        confidence_scores_final = []
        for label in 'Y2AF8':
            indices = np.where(Y_test == label)[0]
            indices = np.random.choice(indices, size=10, replace=False)
            X_final.append(X_test[indices])
            Y_final.append(Y_test[indices])
            predicted_labels_final.append(predicted_labels[indices])
            confidence_scores_final.append(confidence_scores[indices])
            
        X_test = np.concatenate(X_final)
        Y_test = np.concatenate(Y_final)
        predicted_labels = np.concatenate(predicted_labels_final)
        confidence_scores = np.concatenate(confidence_scores_final)
            
    plot_predictions(X_test, Y_test, predicted_labels, confidence_scores)
        
    
def plot_predictions(X_test, Y_test, predictions, confidence_scores) -> None:
    import matplotlib.pyplot as plt

    # Number of rows and columns
    n_rows = ceil(X_test.shape[0] / 10)
    n_cols = 10 if X_test.shape[0] >= 10 else X_test.shape[0]

    # Create a figure with subplots for each character
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(X_test):
            ax.imshow(X_test[idx].reshape(35, 35), cmap='gray')
            ax.set_title(f'T: {Y_test[idx]}\nP: {predictions[idx]}\nC: {confidence_scores[idx]:.2f}', fontsize=8)
        ax.axis('off')
            
    # Shrink the figure so that the plot fits
    fig.tight_layout()    
    #plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.savefig('predictions.png')


def get_model():
    if os.path.exists(MODEL_FILE):
        # Load the model from the file
        model = load_model(MODEL_FILE)
        model.summary()
    else:
        model = Sequential([
            Conv2D(4, (3, 3), activation='relu', input_shape=(35, 35, 1)),
            MaxPooling2D(2, 2),
            Conv2D(4, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            #Conv2D(4, (3, 3), activation='relu'),
            #MaxPooling2D(2, 2),
            #Conv2D(32, (3, 3), activation='relu'),
            #MaxPooling2D(2, 2),
            Flatten(),
            Dense(8, activation='relu'),
            Dense(len(CHARACTERS), activation='softmax')
        ])

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        print("Generating data...")
        
        X, Y = get_training_data(synthetic_size=200000)
        
        Y = to_categorical(Y, num_classes=len(CHARACTERS))
        
        print("Done!")
        print(f"Training on {len(X)} images")

        model.fit(X, Y, epochs=15, validation_split=0.1, batch_size=64)

        # Save the model after training
        model.save(MODEL_FILE)

    return model


if __name__ == '__main__':
    model = get_model()
    test_model(model)