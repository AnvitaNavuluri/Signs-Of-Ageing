import cv2
import numpy as np
from keras.models import model_from_json

json_file = open("age.json", "r")
loaded_model_json = json_file.read()
json_file.close
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("ageweights.h5")
print("loaded the model successfully")

# CHANGE PATH TO FILE HERE.................
file_name = "C:/Users/jonah/Downloads/PuffyEyes_Sample1.jpg"


def getClassName(classNo):
    if classNo == 0:
        return 'Dark Spots'
    elif classNo == 1:
        return 'Puffy Eyes'
    elif classNo == 2:
        return 'Wrinkled Skin'


def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255
    return image


image = cv2.imread(file_name)
imagearr = np.asarray(image)  # image is converted to array
imagearr = cv2.resize(imagearr, (100, 100))  # resize the array change dimension accordingly
imagearr = preprocessing(imagearr)  # applying preprocessing technique
imagearr = imagearr.reshape(1, 100, 100, 1)  # converting 2D image to 4D image cause training data was 4D
predictions = loaded_model.predict(imagearr)  # collection of probablities
classIndex = loaded_model.predict_classes(imagearr)  # gives the index having highest Probablity value
cv2.putText(image, "CLass : ", (2, 15), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 255), 2)
cv2.putText(image, "Prediction : ", (2, 55), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 255), 2)
probabilityValue = np.amax(predictions)

if probabilityValue > 0.50:
    cv2.putText(image, getClassName(classIndex), (60, 15), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 255), 2)
    cv2.putText(image, str(round(probabilityValue * 100, 3)) + " %", (95, 55), cv2.FONT_HERSHEY_PLAIN, 0.9,
                (255, 0, 255), 2)

image = cv2.resize(image, (300, 300))
cv2.imshow("IMAGE", image)
cv2.waitKey(0)
