# Cut and paste from:
#   http://dlib.net/face_recognition.py.html
#   https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py
#   https://medium.com/towards-data-science/facial-recognition-using-deep-learning-a74e9059a150

import os
import dlib
import numpy as np
from skimage import io
import cv2

data_dir = os.path.expanduser('~/data')

# Globals
dlib_frontal_face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(data_dir + '/dlib/shape_predictor_5_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1(data_dir + '/dlib/dlib_face_recognition_resnet_model_v1.dat')
face_classifier_opencv = cv2.CascadeClassifier(data_dir + '/opencv/haarcascade_frontalface_default.xml')


# def to_dlib_rect(w, h):
#     return dlib.rectangle(left=0, top=0, right=w, bottom=h)
#
#
# def to_rect(dr):
#     #  (x, y, w, h)
#     return dr.left(), dr.top(), dr.right()-dr.left(), dr.bottom()-dr.top()
#
#
# def face_detector_opencv(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return face_classifier_opencv.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(100, 100),
#         flags=cv2.CASCADE_SCALE_IMAGE)
#
#
# def face_detector_dlib(image):
#     bounds = dlib_frontal_face_detector(image, 0) # second parameter is upsample; 1 or 2 will detect smaller faces. 0 performs similar to opencv with current parameters
#     return list(map(lambda b: to_rect(b), bounds))


def get_face_matches(known_faces, face):
    return np.linalg.norm(known_faces - face, axis=1)


def dlib_landmarks_to_array(dlib_landmarks):
    return [(p.x, p.y) for p in dlib_landmarks.parts()]


def load_face_metrics(face_image_id, path_to_image):
    image = io.imread(path_to_image)

    faces_bounds = dlib_frontal_face_detector(image, 0) # second parameter is upsample; 1 or 2 will detect smaller faces. 0 performs similar to opencv with current parameters

    if len(faces_bounds) != 1:
        print("Expected one and only one face per image: " + path_to_image + " - it has " + str(len(faces_bounds)))
        return None

    face_bounds = faces_bounds[0]
    face_landmarks = shape_predictor(image, face_bounds)
    face_embedding = np.array(
        face_recognition_model.compute_face_descriptor(image, face_landmarks, 1)
    )

    metrics = {}
    metrics["image_id"] = face_image_id
    metrics["path"] = path_to_image
    metrics["bounds"] = face_bounds
    metrics["landmarks"] = dlib_landmarks_to_array(face_landmarks)
    metrics["embedding"] = face_embedding

    return metrics


def load_person(person_id, path):
    print("Loading faces for ", person_id)
    person_stuff = {}
    person_stuff["person_id"] = person_id
    person_file_names = [f for f in os.listdir(path) if os.path.isfile(path + "/" + f)]
    faces_metrics = []
    for face_file_name in person_file_names:
        face_metrics = load_face_metrics(face_file_name, path + "/" + face_file_name)
        if (face_metrics is not None):
            faces_metrics.append(face_metrics)

    person_stuff["faces"] = faces_metrics
    return person_stuff


def load_people(path):
    person_dirs = [d for d in os.listdir(path) if os.path.isdir(path + "/" + d)]
    all_persons = {}
    for person_dir in person_dirs:
        person_id = person_dir
        person_stuff = load_person(person_id, path + "/" + person_dir)
        all_persons[person_id] = person_stuff
    return all_persons


def calculate_quality(own_matches, other_matches):
    own = np.average(own_matches)
    other = np.average(other_matches)
    return (other-own, own, other)

def dict_minus(dict, key):
    d = dict.copy()
    d.pop(key)
    return d


def find_other_embeddings(embeddings_per_person, person_id):
    others_embeddings = dict_minus(embeddings_per_person, person_id).values()
    return [item for sublist in others_embeddings for item in sublist]  # just efficient flatten()


def calculate_qualities(people):
    print("Calculating embedding qualities")
    embeddings_per_person = dict([(k, [f["embedding"] for f in v["faces"]]) for k, v in people.items()])

    for person_id, person in people.items():
        print("  " + person_id + ":")
        own_embeddings = [f["embedding"] for f in person["faces"]]
        others_embeddings = find_other_embeddings(embeddings_per_person, person_id)

        for face in person["faces"]:
            own_embeddings_except_this_face = own_embeddings # TODO minus self
            some_others_embeddings = others_embeddings # For now use all, it runs fast :)

            own_matches = get_face_matches(own_embeddings_except_this_face, face["embedding"])
            other_matches = get_face_matches(some_others_embeddings, face["embedding"])
            quality = calculate_quality(own_matches, other_matches)

            face["quality"] = (face["image_id"], quality)

        print("   - qualities: ", [face["quality"] for face in person["faces"]])


# def measure_goodness(face, faces, algorithm):
#     return None
#
#
# def measure_goodness(people, algorithm):
#     all_embeddings = people.
#     for person in people:
#         best_face = algorithm(person.faces)
#         own_embeddings =
#         own_distance = faces
#
#     return None


# # Algorithms
#
# def random(faces):
#     return faces[0]
#
# def simple_landmark(faces):
#     return faces[0]
#
# def learned(faces):
#     return faces[0]
#

##########



# Start
people = load_people(data_dir + '/ms-celeb/MsCelebV1-Faces-Aligned.Samples/samples/')
#print(all_faces)
calculate_qualities(people)
# measure_goodness(people, random)
# measure_goodness(people, simple_landmark)
# measure_goodness(people, learned)



# people:
#{
#    'm.03g19n': {
#        'person_id': 'm.03g19n',
#        'best_landmark':, ''
#        'faces': [
#            {
#                'image_id': '82-FaceId-0.jpg',
#                'quality': TBD: 0..1
#                'path':  '/Users/trygve/data/ms-celeb/MsCelebV1-Faces-Aligned.Samples/samples//m.03g19n/82-FaceId-0.jpg',
#                'bounds': rectangle(21,37,93,109),
#                'landmarks': [(80, 52), (66, 54), (31, 52), (44, 54), (56, 84)],
#                'embedding': array([<128-vector embedding>])
#            },
#            {
#                'image_id': '91-FaceId-0.jpg',
#                'path':  '/Users/trygve/data/ms-celeb/MsCelebV1-Faces-Aligned.Samples/samples//m.03g19n/91-FaceId-0.jpg',
#                'bounds': rectangle(21,37,93,109),
#                'landmarks': [(80, 52), (66, 54), (31, 52), (44, 54), (56, 84)],
#                'embedding': array([<128-vector embedding>])
#            },
#            ...
#        ]
#    },
#    'm.044wfaf': {...}
#}


