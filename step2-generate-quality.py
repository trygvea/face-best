# Cut and paste from:
#   http://dlib.net/face_recognition.py.html
#   https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py
#   https://medium.com/towards-data-science/facial-recognition-using-deep-learning-a74e9059a150

import os
import dlib
import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt

from util import dict_minus_immutable, load_dict, save_dict

intermediate_file = '.intermediate/faces.npy'

def get_face_matches(known_faces, face):
    return np.linalg.norm(known_faces - face, axis=1)


def calculate_quality(own_matches, other_matches):
    """
    IDEAS:
        - ? drop the 10% worst images of ourselves
            > or probably not - we want to score high on these too
        - ! drop the 80% worst images of other
            > we are primarily interested in other images that look a bit like us
        - subtract standard deviation
    """
    details = {}
    avg_own = np.average(own_matches)
    avg_oth = np.average(other_matches)
    std_own = np.std(own_matches)
    std_oth = np.std(other_matches)

    details["avg_own"] = avg_own
    details["avg_oth"] = avg_oth
    details["std_own"] = std_own
    details["std_oth"] = std_oth

    quality = (avg_oth - avg_own, avg_own, avg_oth)
    return (quality, details)


def find_other_embeddings(embeddings_per_person, person_id):
    others_embeddings = dict_minus_immutable(embeddings_per_person, person_id).values()
    return [item for sublist in others_embeddings for item in sublist]  # just efficient flatten()


def calculate_qualities(people):
    print("Calculating embedding qualities")
    embeddings_per_person = dict([(k, [f["embedding"] for f in v["faces"]]) for k, v in people.items()])

    for person_id, person in people.items():
        print("  " + person_id + ":")
        others_embeddings = find_other_embeddings(embeddings_per_person, person_id)

        for face in person["faces"]:
            own_embeddings_except_this_face = [f["embedding"] for f in person["faces"] if f["image_id"] != face["image_id"]]
            some_others_embeddings = others_embeddings # For now use all, it runs fast :)

            own_matches = get_face_matches(own_embeddings_except_this_face, face["embedding"])
            other_matches = get_face_matches(some_others_embeddings, face["embedding"])
            quality, quality_details = calculate_quality(own_matches, other_matches)

            face["quality"] = quality
            face["quality_details"] = quality_details

        print("   - qualities: ", [face["quality"] for face in person["faces"]])


def print_best_image(people):
    for person_id, person in people.items():
        faces = sorted(person["faces"], key=lambda f: -f['quality'][0])
        print("Best image for recognition for "+person_id+" is "+faces[0]["image_id"]+"; having quality:", faces[0]["quality"])



def plot_images(faces):
    return None

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



people = load_dict(intermediate_file)
calculate_qualities(people)

print_best_image(people)

# TODO
# TODO
# TODO
# TODO
# (x,y) = find_landmark_and_quality(people)
# Do linear regression on these
# Check out https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# TODO try 68-points landmarks
# TODO make some visualisations





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
#                'quality_details': {
#                   avg_own, avg_oth, std_own, std_oth, ...
#                },
#                'path':  '/Users/trygve/data/ms-celeb/MsCelebV1-Faces-Aligned.Samples/samples//m.03g19n/82-FaceId-0.jpg',
#                'bounds': rectangle(21,37,93,109),
#                'landmarks': [(80, 52), (66, 54), (31, 52), (44, 54), (56, 84)],
#                'embedding': array([<128-vector embedding>])
#            },
#            {
#                'image_id': '91-FaceId-0.jpg',
#                'quality': TBD: 0..1
#                'quality_details': {
#                   avg_own, avg_oth, std_own, std_oth, ...
#                },
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


