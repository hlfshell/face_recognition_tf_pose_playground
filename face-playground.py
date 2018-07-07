import face_recognition
import cv2
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import pickle

facedirs = [filename for filename in os.listdir("./faces") if not isfile(join("./faces", filename))]

faces = dict.fromkeys(facedirs, [])

for name in facedirs:
    faces[name] = []

    for filepath in [filename for filename in os.listdir(join("./faces", name)) if isfile(join('./faces', name, filename))]:
        faces[name].append(join("./faces", name, filepath))


def save_image(filepath, img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, bgr)

def draw_image(img, title=None):
    if title is None:
        title = "Output"

    plt.figure()
    plt.suptitle(title)
    plt.imshow(img)
    plt.show()

# Now that we have all the images organized per person, let's go through and create the encodings.
def get_face_encodings(filepath):
    # First, load the file
    img = cv2.imread(filepath)
    # Convert to rgb for face_recognition/dlib library
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Do I have to resize?
    height, width, channels = img.shape
    if height > 600 or width > 600:
        # Resize!
        if height > width:
            ratio = 600 / height
            img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
        else:
            ratio = 600 / width
            img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)

    #Localized faces
    bounding_boxes = face_recognition.face_locations(img, model="cnn")

    if len(bounding_boxes) <= 0:
        print("No faces found for ", filepath)
        return

    # top, right, bottom and left 
    top, right, bottom, left = bounding_boxes[0]
    # cv2.rectangle(img, (bounding_boxes[0][1], bounding_boxes[0][0]), (bounding_boxes[0][3], bounding_boxes[0][2]), (255, 0, 0), 2, 0)

    #Grab just the face
    # face = img[top:bottom, left:right]


    encodings = face_recognition.face_encodings(img, bounding_boxes)

    return encodings


def create_encodings():
    print("Creating encodings...")
    encodings = {}

    for name in faces:
        encodings[name] = []
        
        for filepath in faces[name]:
            try:
                print("Processing file...", filepath)
                encoding = get_face_encodings(filepath)
                encodings[name].append(encoding[0])
            except:
                print("No face found in this img", filepath)

    f = open("./encodings.p", "wb")
    f.write(pickle.dumps(encodings))
    f.close()

    return encodings
    
def load_encodings():
    return pickle.loads(open("./encodings.p", "rb").read())

# create_encodings()

encodings = load_encodings()

camera = cv2.VideoCapture(0)

ret, frame = camera.read()
print(ret)
print(frame)
print(camera.isOpened())

def compare_encoding(encoding):
    counts = {}
    
    for name in encodings:
        results = face_recognition.compare_faces(encodings[name], encoding)
        counts[name] = results.count(True)
        
    biggest_match = max(counts, key=counts.get)
    if(counts[biggest_match] <= 3):
        biggest_match = "Unknown"

    return biggest_match

def identify_in_image(img):
    bounding_boxes = face_recognition.face_locations(img, model="cnn")

    if len(bounding_boxes) <= 0:
        return

    # top, right, bottom and left 
    for bbox in bounding_boxes:
        # top, right, bottom, left = bbox
        cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 2, 0)

    encodings = face_recognition.face_encodings(img, bounding_boxes)

    for index, encoding in enumerate(encodings):
        name = compare_encoding(encoding)

        cv2.putText(img, name, (bounding_boxes[index][1], bounding_boxes[index][2]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    draw_image(img)


def handle_frame():
    _, frame = camera.read()





# key = cv2.waitKey(1) & 0xFF
 
# # if the `q` key was pressed, break from the loop
# if key == ord("q"):
#     break

# handle_frame()
img = cv2.imread("./test2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
identify_in_image(img)