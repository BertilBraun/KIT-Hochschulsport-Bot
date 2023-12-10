import os
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import string
import random

MODEL_FILE = R'C:\Projects\Hochschulsport-Bot\src\captcha\neural_net\captcha_nn.h5'


def get_windows_fonts():
    fonts_dir = 'C:\\Windows\\Fonts'
    font_paths = [os.path.join(fonts_dir, font) for font in os.listdir(fonts_dir) if font.endswith('.ttf')]
    return font_paths

def create_synthetic_data(size=5000, image_size=(35, 35), rotation_range=45):
    characters = string.ascii_uppercase + string.digits
    fonts = [ImageFont.truetype(font, 30) for font in get_windows_fonts()]

    X = []
    Y = []
    for _ in range(size):
        img = Image.new('L', image_size, 'black')
        draw = ImageDraw.Draw(img)
        char = random.choice(characters)
        font = random.choice(fonts)
        
        # Approximate centering
        text_x = (image_size[0] - 28) // 2  # Approximate x position
        text_y = (image_size[1] - 28) // 2  # Approximate y position
        draw.text((text_x, text_y), char, 'white', font=font)
        
        img = img.rotate(random.uniform(-rotation_range, rotation_range))
        X.append(np.array(img))
        Y.append(characters.index(char))

    X = np.array(X) / 255.0  # Normalize
    X = X.reshape(-1, 35, 35, 1)  # Reshape for CNN
    return X, np.array(Y)


def __index_to_label(index):
    characters = string.ascii_uppercase + string.digits
    return characters[index]


def predict(model, images):
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)

    predicted_labels = [__index_to_label(index) for index in predicted_labels]
    return predicted_labels, confidence_scores


def test_model(model):
    X_test, Y_test = create_synthetic_data(size=200)
    Y_test = [__index_to_label(index) for index in Y_test]
    predicted_labels, confidence_scores = predict(model, X_test)
    
    correct_predictions = 0
    for predicted_label, actual_label in zip(predicted_labels, Y_test):
        if predicted_label == actual_label:
            correct_predictions += 1
    
    print(f"Correct predictions: {correct_predictions} / {len(Y_test)}")
    print(f"Accuracy: {correct_predictions / len(Y_test) * 100:.2f}%")
    
    plot_predictions(X_test, Y_test, predicted_labels, confidence_scores)
    
    
def plot_predictions(X_test, Y_test, predictions, confidence_scores):
    import matplotlib.pyplot as plt

    # Number of rows and columns
    n_rows = 20
    n_cols = 10

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 40))  # Adjust the size as needed

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]
            ax.imshow(X_test[idx].reshape(35, 35), cmap='gray')
            ax.set_title(f'T: {Y_test[idx]}\nP: {predictions[idx]}\nC: {confidence_scores[idx]:.2f}', fontsize=8)
            ax.axis('off')

    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.savefig('predictions.png')


def get_model():
    if os.path.exists(MODEL_FILE):
        # Load the model from the file
        model = load_model(MODEL_FILE)
    else:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(35, 35, 1)),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(36, activation='softmax')  # 26 letters + 10 digits
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("Generating data...")
        X, Y = create_synthetic_data(size=1000000)
        print("Done!")

        model.fit(X, Y, epochs=5, validation_split=0.1)

        # Save the model after training
        model.save(MODEL_FILE)

    return model


if __name__ == '__main__':
    model = get_model()
    test_model(model)