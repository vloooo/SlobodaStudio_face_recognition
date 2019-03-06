import base64
import face_recognition
import cv2
import copy
import numpy as np
from random import choice
from string import ascii_uppercase
import os
import shutil


def rcgn(frame, known_face_encodings, names):

    frame_for_box = copy.deepcopy(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    face_names = []

    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(known_face_encodings,
                                                 encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matched_idxs:
                name = names[i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        face_names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, face_names):
            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            cv2.rectangle(frame_for_box, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
        # cv2.imshow('fff', frame)
        # cv2.waitKey(0)
    return frame_for_box, frame


def location(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)

    for (top, right, bottom, left) in boxes:
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)

    return frame


def prepare_img(req):
    recognized = []

    name = ''.join(choice(ascii_uppercase) for _ in range(12))
    os.mkdir(name)

    for ind, i in enumerate(req):

        image_url = np.asarray(bytearray(i.read()), dtype="uint8")
        im_orig = cv2.imdecode(image_url, cv2.IMREAD_COLOR)
        cv2.imwrite(name + '/' + name + str(ind) + '.png', im_orig)

        im_orig = cv2.resize(im_orig, (0, 0), fx=0.75, fy=0.75)
        im_orig = location(im_orig)

        retval, buffer = cv2.imencode('.jpg', im_orig)
        jpg_as_text = base64.b64encode(buffer)

        recognized.append(jpg_as_text)

    s = ''
    for i in recognized:
        s += """<img alt="recognized_photo" src="data:image/jpeg;base64,""" + str(i)[2:-1] + '">'

    return s, name


def prepare_data(req, names):
    known_encodings = []
    known_names = []
    for i in req:

        image_url = np.asarray(bytearray(i.read()), dtype="uint8")
        im_orig = cv2.imdecode(image_url, cv2.IMREAD_COLOR)
        im_orig = cv2.resize(im_orig, (0, 0), fx=0.75, fy=0.75)

        rgb = cv2.cvtColor(im_orig, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            known_encodings.append(encoding)
            known_names.append(names[0])

    return known_encodings, known_names


def prepare_img_rcgn(req, file_name, names):
    recognized = []
    images_to_rcgn = os.listdir(file_name)

    enc, names = prepare_data(req, names)
    for i in range(len(images_to_rcgn)):

        im_orig = cv2.imread(file_name + '/' + file_name + str(i) + '.png')

        im_orig = cv2.resize(im_orig, (0, 0), fx=0.75, fy=0.75)
        im_orig_box, im_orig = rcgn(im_orig, enc,  names)

        retval, buffer = cv2.imencode('.jpg', im_orig)
        jpg_as_text = base64.b64encode(buffer)

        recognized.append(jpg_as_text)

    shutil.rmtree(file_name)
    rcg = ''
    for i in recognized:
        rcg += """<img alt="recognized_photo" src="data:image/jpeg;base64,""" + str(i)[2:-1] + '">'

    return rcg
