import numpy as np
from skimage import io
import matplotlib.pyplot as plt

from util import save_dict, timing
from config import *


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


def dlib_landmarks_to_array(dlib_landmarks):
    return [(p.x, p.y) for p in dlib_landmarks.parts()]


@timing
def read_file(path_to_image):
    return io.imread(path_to_image)

@timing
def detect_face(image):
    return dlib_frontal_face_detector(image, 0) # second parameter is upsample; 1 or 2 will detect smaller faces. 0 performs similar to opencv with current parameters

@timing
def get_landmarks_5(image, face_bounds):
    return shape_predictor_5(image, face_bounds)

@timing
def get_landmarks_68(image, face_bounds):
    return shape_predictor_68(image, face_bounds)

@timing
def get_embedding(image, landmarks):
    return np.array(
        face_recognition_model.compute_face_descriptor(image, landmarks, 1)
    )

def load_face_metrics(face_image_id, path_to_image):
    image = read_file(path_to_image)

    faces_bounds = detect_face(image)
    if len(faces_bounds) != 1:
        print("Expected one and only one face per image: " + path_to_image + " - it has " + str(len(faces_bounds)))
        return None

    face_bounds = faces_bounds[0]
    face_landmarks_5 = get_landmarks_5(image, face_bounds)
    face_landmarks_68 = get_landmarks_68(image, face_bounds)
    face_embedding = get_embedding(image, face_landmarks_5)

    metrics = {}
    metrics["image_id"] = face_image_id
    metrics["path"] = path_to_image
    metrics["bounds"] = face_bounds
    metrics["landmarks-5"] = dlib_landmarks_to_array(face_landmarks_5)
    metrics["landmarks-68"] = dlib_landmarks_to_array(face_landmarks_68)
    metrics["embedding"] = face_embedding

    return metrics


def load_person(person_id, path):
    print("Loading faces for ", person_id)
    person_stuff = {}
    person_stuff["person_id"] = person_id
    person_file_names = [f for f in os.listdir(path) if os.path.isfile(path + "/" + f)]
    faces_metrics = []
    error_faces = []
    for face_file_name in person_file_names:
        path_to_image = path + "/" + face_file_name
        face_metrics = load_face_metrics(face_file_name, path_to_image)
        if (face_metrics is not None):
            faces_metrics.append(face_metrics)
        else:
            error_faces.append({"image_id": face_file_name, "path": path_to_image})

    person_stuff["faces"] = faces_metrics
    return person_stuff, error_faces


def load_people(path):
    person_dirs = [d for d in os.listdir(path) if os.path.isdir(path + "/" + d)]
    people = {}
    errors = []
    for person_dir in person_dirs:
        person_id = person_dir
        person_stuff, error_faces = load_person(person_id, path + "/" + person_dir)
        people[person_id] = person_stuff
        errors.append(error_faces)
    return people, errors


def plot_error_faces(people_errors):
    errors = [e for e in people_errors if len(e) > 0]
    errors_per_person = [len(e) for e in errors]
    num_people_with_errors = len(errors)
    max_errors = np.max(errors_per_person)
    plt.figure("Unrecognized faces", figsize=(15, 10))
    for row, error in enumerate(errors):
        for col, error_face in enumerate(error):
            image = io.imread(error_face["path"])
            ax = plt.subplot(num_people_with_errors, max_errors, row*max_errors + col + 1)
            ax.set_axis_off()
            plt.imshow(image)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)


@timing
def run():
    people, errors = load_people(ms_celeb_samples_dir)
    num_faces = [len(p["faces"]) for p in people.values()]
    num_errors = sum([len(e) for e in errors])

    print("Loaded ", len(people.keys()), " people with an average of ", np.average(num_faces), " recognized face images each, and a total of ", num_errors, " faces with no (or many) recognized faces")

    print("Saving people and embeddings to file "+intermediate_file)
    save_dict(intermediate_file, people)

    print("Plotting unrecognized faces")
    plot_error_faces(errors)


run()
plt.show()

# people:
#{
#    'm.03g19n': {
#        'person_id': 'm.03g19n',
#        'faces': [
#            {
#                'image_id': '82-FaceId-0.jpg',
#                'path':  '/Users/trygve/data/ms-celeb/MsCelebV1-Faces-Aligned.Samples/samples//m.03g19n/82-FaceId-0.jpg',
#                'bounds': rectangle(21,37,93,109),
#                'landmarks-5': [(80, 52), (66, 54), (31, 52), (44, 54), (56, 84)],
#                'landmarks-68': [(80, 52), (66, 54), (31, 52), (44, 54), (56, 84), ...],
#                'embedding': array([<128-vector embedding>])
#            },
#            {
#                'image_id': '91-FaceId-0.jpg',
#                'path':  '/Users/trygve/data/ms-celeb/MsCelebV1-Faces-Aligned.Samples/samples//m.03g19n/91-FaceId-0.jpg',
#                'bounds': rectangle(21,37,93,109),
#                'landmarks-5': [(80, 52), (66, 54), (31, 52), (44, 54), (56, 84)],
#                'landmarks-68': [(80, 52), (66, 54), (31, 52), (44, 54), (56, 84), ...],
#                'embedding': array([<128-vector embedding>])
#            },
#            ...
#        ]
#    },
#    'm.044wfaf': {...}
#}


