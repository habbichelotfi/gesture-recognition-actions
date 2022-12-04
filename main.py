import numpy as np
import os
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import cv2

GESTURE_NAME = ["A", "C", "D", "E", "F", "G", "h", "K", "L", "N", "P", "AQ", "S", "U", "V", "W", "Z"]


def predict_image(image,model):
    image = np.array(image,dtype='float32')
    pred_arr = model.predict(image)

    print(pred_arr)
    result = GESTURE_NAME[int(np.argmax(pred_arr))]
 
    score = float("%0.2f" % (max(pred_arr[0]) * 100))
    print(f'Result: {result}, Score: {score}')
    return result, score
# Loops through imagepaths to load images and labels into arrays
def train():
    imagepaths = "datasets/Below_CAM/"
    X = []  # Image data
    y = []  # Labels
    i = 0
    for p in os.listdir(imagepaths):
        pa=imagepaths+p
        print(pa)
        for path in os.listdir(pa):
            print(pa+path)
            img = cv2.imread(pa+"/"+path)  # Reads image and returns np.array
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts into the corret colorspace (GRAY)
            img = cv2.resize(img, (300, 300))  # Reduce image size so training can be faster
            X.append(img)
            i=i+1
            # Processing label in image path
            category =( pa+"/"+path).split("/")[3]
            category=category.split("_")[1].split(".")[0]
            #label = int(category.split("_")[0][1])  # We need to convert 10_down to 00_down, or else it crashes
            y.append(category)
    # Turn X and y into np.array to speed up train_test_split

    X = np.array(X, dtype="uint8")
    print(X.shape)
    #X = X.reshape(len(imagepaths), 120, 320, 1)  # Needed to reshape so CNN knows it's different images
    X=np.expand_dims(X, axis=3)
    print(X.shape)
    y = np.array(y)
    print("Images loaded: ", len(X))
    print("Labels loaded: ", len(y))

    ts = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
    # Construction of model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(300, 300, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Configures the model for training
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Trains the model for a given number of epochs (iterations on a dataset) and validates it.
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2, validation_data=(X_test, y_test))

    model.save("my_model")
    return model
#train()

'''
c=cv2.VideoCapture(1)
while True:

    _,cap=c.read()
    cv2.imshow("s",cap)
    cap=cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
    cap = cv2.resize(cap, (300, 300),cv2.INTER_AREA)
    cap = np.array(cap, dtype="uint8")
    cap=cap.reshape((1,300,300,1))
    #X = X.reshape(len(imagepaths), 120, 320, 1)  # Needed to reshape so CNN knows it's different images
    model=keras.models.load_model('my_model')
    print(predict_image(cap,model))
    keypress = cv2.waitKey(1) & 0xFF

    if keypress == ord("q"):
        break

# free up memory
c.release()
cv2.destroyAllWindows()
'''
ima=cv2.imread("datasets/Below_CAM/D/P1_001.jpg")
cv2.imshow("s",ima)
cap=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
cap = cv2.resize(cap, (300, 300),cv2.INTER_AREA)
cap = np.array(cap, dtype="uint8")
cap=cap.reshape((1,300,300,1))
    #X = X.reshape(len(imagepaths), 120, 320, 1)  # Needed to reshape so CNN knows it's different images
model=keras.models.load_model('my_model')
print(predict_image(cap,model))
cv2.waitKey(0)
# free up memory
