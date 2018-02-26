import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import glob
import os

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

jitters = 10
tolerance = 0.39
detection_model = "cnn"
unknown_image_file = "./unknown/Unknown9.jpg"

known_face_encodings = []
known_face_names = []
known_face_landmarks = []

# get encodings and names of all non-family images
for i in [glob.glob('./known/*.%s' % ext) for ext in ["jpg","gif","png","tga"]]: 
    for item in i:
        image = face_recognition.load_image_file(item)
        face_encoding = face_recognition.face_encodings(image, num_jitters=jitters)[0]
        known_face_encodings.append(face_encoding)
        basename = os.path.basename(item)
        filename, file_extension = os.path.splitext(basename)
        known_face_names.append(filename)
        known_face_landmarks.append(face_recognition.face_landmarks(image))

shen_face_encodings = []
shen_face_names = []
shen_face_landmarks = []

# get encodings and names of all family images
for i in [glob.glob('./known/Shens/*.%s' % ext) for ext in ["jpg","gif","png","tga"]]:
    for item in i:
        image = face_recognition.load_image_file(item)
        face_encoding = face_recognition.face_encodings(image, num_jitters=jitters)[0]
        known_face_encodings.append(face_encoding)
        basename = os.path.basename(item)
        filename, file_extension = os.path.splitext(basename)
        known_face_names.append(filename)
        face_landmark = face_recognition.face_landmarks(image)
        known_face_landmarks.append(face_landmark)
        shen_face_encodings.append(face_encoding)
        shen_face_names.append(filename)
        shen_face_landmarks.append(face_landmark)
print("Shens names: ", shen_face_names)

print("known names: ", known_face_names)
print("known face landmarks: ", known_face_landmarks)

# Euclidean Distance between the Kins
shen_dist = []
shen_face_landmarks_dist = []
for i in range(len(shen_face_names)):
    shen_dist.append(face_recognition.face_distance(shen_face_encodings, shen_face_encodings[i]))
    shen_face_landmarks_dist.append(np.linalg.norm(shen_face_landmarks - shen_face_landmarks[i], axis=1))
print("Shen distance ", shen_dist)
print("Shen face landmarks dist ", shen_face_landmarks_dist)
                                    

# Euclidean Distance between face landmarks


# Load an image with an unknown face
unknown_image = face_recognition.load_image_file(unknown_image_file)

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image, model=detection_model)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    distance = face_recognition.face_distance(known_face_encodings, face_encoding)
    matches = list(distance <= tolerance)
    print("names, distance, matches: ", known_face_names, distance, matches)
    name = "Unknown"
    if matches.count(True) > 1:
        all = True
        # if all results are part of the family, then further minimize the rms
        for i in [i for i,val in enumerate(matches) if val==True]:
            if known_face_names[i] not in shen_face_names:
                all = False
                break

        if all == False:
            name = known_face_names[np.argmin(distance)]
        else:
            dist = face_recognition.face_distance(shen_face_encodings, face_encoding)
            print("dist ", dist)
            rms = np.ones(len(shen_face_names))
            for i in range(len(shen_face_names)):
                # return the one with lowest delta in Euclidean distance
                rms[i] = np.sqrt(((shen_dist[i] - dist)**2).mean())
            print("rms: ", rms)
            name = shen_face_names[np.argmin(rms)]
            print("Name ", name)

    elif matches.count(True) == 1:
        name = known_face_names[matches.index(True)]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face with distance
    name_distance = name
    if name in ["Unknown", "Eman", "Eden"]:
        name_distance += "\nEm" + str(distance[known_face_names.index("Eman")]) + "\nEd" + str(distance[known_face_names.index("Eden")])
    
    text_width, text_height = draw.textsize(name_distance)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 7, bottom - text_height - 6), name_distance, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()

# You can also save a copy of the new image to disk if you want by uncommenting this line
pil_image.save("image_with_boxes.jpg")
