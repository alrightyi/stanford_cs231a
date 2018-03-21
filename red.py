import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from sklearn.svm import LinearSVC
import glob
import os
import matplotlib.pyplot as plt


'''
    Draw box on all face_locations of an image
'''
def draw_face_box(image, face_locations):
    # Convert the image to a PIL-format image
    pil_image = Image.fromarray(image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)
    # Loop through each face found in the unknown image
    for (top, right, bottom, left) in face_locations:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    return draw, pil_image

'''
Get encodings and names of all images
'''
def get_encoding(path,jitters=10,det_model="hog"):
    face_encodings = []
    for i in [glob.glob(path+'*.%s' % ext) for ext in ["jpg","gif","png","tga","jpeg"]]:
        for item in i:
            image = face_recognition.load_image_file(item)
            face_locations = face_recognition.face_locations(image, model=det_model)
            face_encoding = face_recognition.face_encodings(image, face_locations, num_jitters=jitters)
            
            if (face_encoding == [] or len(face_encoding) > 1):
                print("image, face_encoding len: ", item, len(face_encoding))
                
                draw, pil_image = draw_face_box(image, face_locations)
                # Remove the drawing library from memory as per the Pillow docs
                del draw
                # Display the resulting image
                pil_image.show()

            if face_encoding != []:
                face_encodings.append(face_encoding[0])

    face_encodings = np.array(face_encodings)
    return face_encodings


# Compute or load the known data
def load_data(image_path):
    if not (os.path.exists(image_path+'encodings.npy')):
        i = 0
        for k_image_path in glob.glob(image_path+'*'):
            print("image: ", k_image_path)
            if k_image_path == []:
                continue
            k_encodings = get_encoding(k_image_path+'/',jitters,det_model=detection_model)
            if k_encodings != []:
                head, tail = os.path.split(k_image_path)
                if i==0:
                    X = k_encodings
                    Y = np.repeat(i, len(k_encodings))
                    names = [tail]
                else:
                    X = np.vstack((X,k_encodings))
                    Y = np.hstack((Y,np.repeat(i, len(k_encodings))))
                    names = np.hstack((names,tail))
                i+=1
    
        print("X.shape, Y.shape: ", X.shape, Y.shape)
        print("names: ", names)
        
        np.save(image_path+'encodings.npy', X)
        np.save(image_path+'classes.npy', Y)
        np.save(image_path+'names.npy', names)
    else:
        X = np.load(image_path+'encodings.npy')
        Y = np.load(image_path+'classes.npy')
        names = np.load(image_path+'names.npy')

    return X, Y, names

def relative_euclidean_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to another face encoding and get a relative euclidean distance for each comparison face. The distance tells you how similar the faces are.
    
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings/np.sqrt(np.mean(face_encodings**2)) - face_to_compare/np.sqrt(np.mean(face_to_compare**2)), axis=1)
    #return np.linalg.norm(face_encodings - face_to_compare, axis=1)

if __name__ == '__main__':
    jitters = 10
    tolerance = 0.5
    threshold = 0
    detection_model = "cnn"
    image_path = "./data/"
    neg_image_path = "./data/Negative/"
    known_image_path = "./data/known_resized/"
    test_image_path = "./data/test_resized/"
    test_short_image_path = "./data/twin_test_resized/Side_By_Side/"
    side_by_side = True
    
    # load the sample and test data
    X, Y, names = load_data(known_image_path)
    X_t, Y_t, names_t = load_data(test_image_path)
    
    # Names of twins and family members (incl. Unknown)
    #family_names = ["Eman", "Eden", "Enoch", "Albert", "Sandy", "Unknown"]
    all_family_names = ["Eden", "Eman", "Enoch", "Albert", "Sandy", "Hailey", "Ivy", "Ivan"]
    twin_names = ["Eman", "Eden"]
    
    family_sizes = np.arange(2,9)
    twin_accuracy_family = np.zeros_like(family_sizes, dtype=float)
    red_twin_accuracy_family = np.zeros_like(family_sizes, dtype=float)
    
    for x in family_sizes:
        family_names = all_family_names[:x]
    
        # # of classes in the first stage
        stage1_classes = np.arange(len(family_names)+1,len(names)+1)
        print("stage classes: ", stage1_classes)
        
        # One-stage Linear SVC accuracy
        accuracy = np.zeros_like(stage1_classes, dtype=float)
        twin_accuracy = np.zeros_like(stage1_classes, dtype=float)
        non_twin_accuracy = np.zeros_like(stage1_classes, dtype=float)
        
        # Two-stage Linear SVC with RED accuracy
        red_accuracy = np.zeros_like(stage1_classes, dtype=float)
        red_twin_accuracy = np.zeros_like(stage1_classes, dtype=float)
        red_non_twin_accuracy = np.zeros_like(stage1_classes, dtype=float)
        
        X_n = np.empty((0,X.shape[1]))
        Y_n = np.empty((0))
        names_n = []
        # load data of the family
        for member_name, i in zip(family_names, range(len(family_names))):
            member_X = X[[index for index in range(X.shape[0]) if Y[index] == np.where( names==member_name )]]
            X_n = np.vstack((X_n,member_X))
            Y_n = np.hstack((Y_n, np.repeat(i,member_X.shape[0])))
            names_n = np.hstack((names_n, member_name))
        #print("X_n, Y_n, names_n: ", X_n.shape, Y_n.shape, names_n)

        # calculate Relative Euclidean Distance of the samples
        family_red_X = np.empty((0,X_n.shape[0]))
        for encoding in X_n:
            red = relative_euclidean_distance(X_n, encoding)
            family_red_X = np.vstack((family_red_X,red))
        family_Y = np.copy(Y_n)
        #print("family_red_X, family_Y: ", family_red_X.shape, family_Y)
        
        # Train Family RED SVC
        family_clf = LinearSVC(tol=1e-6, max_iter=20000, loss='hinge', class_weight='balanced')
        family_clf.fit(family_red_X, family_Y)
        score = family_clf.score(family_red_X, family_Y)
        print("Family RED SVC score: ", score)

        # load negatives
        neg_X = X[[index for index in range(X.shape[0]) if Y[index] == np.where( names=="Unknown" )]]
        X_n = np.vstack((X_n,neg_X))
        Y_n = np.hstack((Y_n, np.repeat(len(family_names),neg_X.shape[0])))
        names_n = np.hstack((names_n, "Unknown"))
        
        # load the rest of the classes
        class_names = [x for x in names if x not in names_n]
        for class_name, n in zip(class_names, range(len(class_names))):
            class_X = X[[index for index in range(X.shape[0]) if Y[index] == np.where( names==class_name )]]
            X_n = np.vstack((X_n,class_X))
            Y_n = np.hstack((Y_n, np.repeat(len(family_names)+n+1,class_X.shape[0])))
            names_n = np.hstack((names_n, class_name))
        #print("X_n, Y_n, names_n: ", X_n.shape, Y_n, names_n)

        # loop through to calculate accuracies with different # of classes
        for classes, m in zip(stage1_classes, range(len(stage1_classes))):

            # Train the first stage SVM
            clf = LinearSVC(tol=1e-6, max_iter=20000, loss='hinge', class_weight='balanced')
            Y_m = np.copy(Y_n)
            if classes < len(names_n):
                index = np.min(np.where(Y_n == classes))
                Y_m[index:] = np.repeat(len(family_names), len(Y_n[index:]))
            clf.fit(X_n, Y_m)
            score = clf.score(X_n, Y_m)
            print("Classes, m, SVC score: ", classes, m, score)
            #print("X_n, Y_m, names_n: ", X_n.shape, Y_m, names_n[:classes])

            # Calculate accuracy with regular Linear SVC
            twin_total = 0
            correct = 0
            twin_correct = 0
            red_correct = 0
            red_twin_correct = 0
            for face_encoding, i in zip(X_t,range(X_t.shape[0])):
                scores = clf.decision_function(face_encoding.flatten().reshape(1, -1))
                prediction = clf.predict(face_encoding.flatten().reshape(1, -1))
                name_predict = names_n[int(prediction[0])]
                name_actual = names_t[int(Y_t[i])]
                
                # all test data with classes not in SVC is considered Unknown
                if name_actual in class_names[m:]:
                    name_actual = "Unknown"

                # this is test data of a twin
                if name_actual in twin_names:
                    twin_total+=1
                
                # Correct at the 1st stage
                if name_actual == name_predict:
                    correct+=1
                    red_correct+=1
                    # Correctly identified twin in the 1st stage
                    if name_actual in twin_names:
                        twin_correct+=1
                        red_twin_correct+=1

                # secondary assessment based on family RED
                if name_predict in family_names:
                    # calculate the RED of the test encoding
                    family_end_index = np.min(np.where(Y_n == len(family_names)))
                    red = relative_euclidean_distance(X_n[:family_end_index], face_encoding)
                    red_scores = family_clf.decision_function(red.flatten().reshape(1,-1))
                    red_prediction = family_clf.predict(red.flatten().reshape(1, -1))
                    red_name_predict = names_n[int(red_prediction[0])]

                    if (name_actual == red_name_predict):
                        # Corrected a wrong prediction from the 1st stage
                        if red_name_predict != name_predict:
                            #print("corrected!")
                            red_correct+=1
                            if name_actual in twin_names:
                                red_twin_correct+=1
                                print("twin corrected: 1-stage, 2-stage, actual: ", name_predict, red_name_predict, name_actual)
                    else:
                        # Made a wrong choice on the 2nd stage that was correct
                        if (name_actual == name_predict):
                            #print("wrong!")
                            red_correct-=1
                            if name_actual in twin_names:
                                red_twin_correct-=1

            accuracy[m] = correct / X_t.shape[0] * 100
            twin_accuracy[m] = twin_correct / twin_total * 100
            non_twin_accuracy[m] = (correct - twin_correct) / (X_t.shape[0] - twin_total) * 100
            #print("acc, twin_acc, non_twin_acc: ", accuracy[m], twin_accuracy[m], non_twin_accuracy[m])

            red_accuracy[m] = red_correct / X_t.shape[0] * 100
            red_twin_accuracy[m] = red_twin_correct / twin_total * 100
            red_non_twin_accuracy[m] = (red_correct - red_twin_correct) / (X_t.shape[0] - twin_total) * 100
            #print("red acc, twin_acc, non_twin_acc: ", red_accuracy[m], red_twin_accuracy[m], red_non_twin_accuracy[m])

        plt.plot(stage1_classes, accuracy, 'r--', label='Overall Accuracy (1-stage)')
        plt.plot(stage1_classes, twin_accuracy, 'r*-', label='Twin Accuracy (1-stage)')
        plt.plot(stage1_classes, non_twin_accuracy, 'rs-', label='Non-Twin Accuracy (1-stage)')
        plt.plot(stage1_classes, red_accuracy, 'b--', label='Overall Accuracy (2-stage RED)')
        plt.plot(stage1_classes, red_twin_accuracy, 'b*-', label='Twin Accuracy (2-stage RED)')
        plt.plot(stage1_classes, red_non_twin_accuracy, 'bs-', label='Non-Twin Accuracy (2-stage RED)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel('# of classes in Stage 1 SVC')
        plt.ylabel('Accuracy (%)')
        plt.title('1-Stage Linear SVC vs. 2-Stage Family of %s RED SVC' % len(family_names))
        plt.grid(True)
        plt.savefig("test%s.png" % len(family_names))
        plt.show()

        twin_accuracy_family[x-2] = twin_accuracy[-1]
        red_twin_accuracy_family[x-2] = red_twin_accuracy[-1]
     
    # plot aginst family sizes
    plt.plot(family_sizes, twin_accuracy_family, 'r--', label='Twin Accuracy (1-stage)')
    plt.plot(family_sizes, red_twin_accuracy_family, 'b--', label='Twin Accuracy (2-stage RED)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('# of Family Members')
    plt.ylabel('Twin Accuracy (%)')
    plt.title('1-Stage Linear SVC vs. 2-Stage Family RED SVC')
    plt.grid(True)
    plt.savefig("test2.png")
    plt.show()


    if side_by_side:
        # test all images in test file
        image_files = [os.path.join(test_short_image_path, f) for f in os.listdir(test_short_image_path) if (f.endswith('.jpg') or f.endswith('.png'))]

        for u_image_file in image_files:
            print("Image file: ", u_image_file)
            # Load an image with an unknown face
            u_image = face_recognition.load_image_file(u_image_file)
        
            # Find all the faces and face encodings in the unknown image
            face_locations = face_recognition.face_locations(u_image, model=detection_model)
            face_encodings = face_recognition.face_encodings(u_image, face_locations)
        
            # Convert the image to a PIL-format image
            pil_image = Image.fromarray(u_image)
            # Create a Pillow ImageDraw Draw instance to draw with
            draw = ImageDraw.Draw(pil_image)
        
            # Loop through each face found in the unknown image
            for (top, right, bottom, left), face_encoding, i in zip(face_locations, face_encodings, range(len(face_encodings))):
                
                # See if the face is a match for the known face(s)
                scores = clf.decision_function(face_encoding.flatten().reshape(1, -1))
                prediction = clf.predict(face_encoding.flatten().reshape(1, -1))

                name_predict = red_name_predict = names_n[int(prediction[0])]
                
                # secondary assessment based on family RED
                if name_predict in family_names:
                    # calculate the RED of the test encoding
                    family_end_index = np.min(np.where(Y_n == len(family_names)))
                    red = relative_euclidean_distance(X_n[:family_end_index], face_encoding)
                    red_scores = family_clf.decision_function(red.flatten().reshape(1,-1))
                    red_prediction = family_clf.predict(red.flatten().reshape(1, -1))
                    red_name_predict = names_n[int(red_prediction[0])]

                head, tail = os.path.split(u_image_file)
                if tail.startswith("Eman"):
                    name_actual = ["Eman", "Eden"]
                else:
                    name_actual = ["Eden", "Eman"]

                print("name_predict, red_name_predict, name_actual, location: ", name_predict, red_name_predict, name_actual[i], top, right, bottom, left)

                # Draw a box around the face using the Pillow module
                draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

                # Draw a label with a name below the face with distance of the twins in the family
                name_distance = "1-Stage: " + name_predict + "\n"
                name_distance += "2-Stage: " + red_name_predict + "\n"
                name_distance += "Actual: " + name_actual[i]

                text_width, text_height = draw.textsize(name_distance)
                draw.rectangle(((left, bottom), (right, bottom - text_height + 50)), fill=(0, 0, 255), outline=(0, 0, 255))
                draw.text((left + 7, bottom - text_height + 30), name_distance, fill=(255, 255, 255, 255))

            # Remove the drawing library from memory as per the Pillow docs
            del draw

            # Display the resulting image
            pil_image.show()

