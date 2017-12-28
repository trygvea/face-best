import numpy as np
from skimage import io
import matplotlib.pyplot as plt

from util import dict_minus_immutable, load_dict, timing
from config import *


@timing
def get_face_matches(known_faces, face):
    return np.linalg.norm(known_faces - face, axis=1)


@timing
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

    others_like_me_n = int(len(other_matches)*0.20) # the 20% that looks most like me
    details["avg_others_like_me"] = np.average(sorted(other_matches)[:others_like_me_n])

    # Number of own images that fails comparison
    details["n_false_negative"] = len(own_matches[own_matches > 0.6])

    # Number of other images that tests positive
    details["n_false_positive"] = len(other_matches[other_matches < 0.6])

    quality = avg_oth - avg_own
    return (quality, details)


def find_other_embeddings(embeddings_per_person, person_id):
    others_embeddings = dict_minus_immutable(embeddings_per_person, person_id).values()
    return [item for sublist in others_embeddings for item in sublist]  # just efficient flatten()


def calculate_qualities(people):
    print("Calculating embedding qualities")
    embeddings_per_person = dict([(k, [f["embedding"] for f in v["faces"]]) for k, v in people.items()])

    for person_id, person in people.items():
        others_embeddings = find_other_embeddings(embeddings_per_person, person_id)
        print("  " + person_id + ": faces: ", len(person["faces"]), ", other faces: ", len(others_embeddings))

        for face in person["faces"]:
            own_embeddings_except_this_face = [f["embedding"] for f in person["faces"] if f["image_id"] != face["image_id"]]
            some_others_embeddings = others_embeddings # For now use all, it runs fast :)

            own_matches = get_face_matches(own_embeddings_except_this_face, face["embedding"])
            other_matches = get_face_matches(some_others_embeddings, face["embedding"])
            quality, quality_details = calculate_quality(own_matches, other_matches)

            face["quality"] = quality
            face["quality_details"] = quality_details

        # print("   - qualities: ", [face["quality"] for face in person["faces"]])


def print_best_image(people):
    for person_id, person in people.items():
        faces = sorted(person["faces"], key=lambda f: -f['quality'])
        print("Best image for recognition for "+person_id+" is "+faces[0]["image_id"]+"; having quality:", faces[0]["quality"])



def plot_persons_faces(person):
    faces = sorted(person["faces"], key=lambda f: -f['quality'])    # best quality first
    faces = faces[0:18] + faces[-18:]   # Just some, not all
    num_cols = int(np.sqrt(len(faces))-(1e-7)) + 1
    print("Plotting face for ", person["person_id"])
    plt.figure(figsize=(num_cols, num_cols))
    for i, face in enumerate(faces):
        image = io.imread(face["path"])
        ax = plt.subplot(num_cols, num_cols, i + 1)

        #ax.text(0, 40, "{0:.2f}".format(face["quality"]), fontsize=10, color='red')
        ax.text(0, 40, "{0:.2f}".format(face["quality_details"]["avg_own"]), fontsize=10, color='red')
        ax.text(100, 40, "{0:.2f}".format(face["quality_details"]["avg_oth"]), fontsize=10, color='green')
        ax.text(0, 80, "{0:.2f}".format(face["quality_details"]["n_false_negative"]), fontsize=10, color='blue')
        ax.text(100, 80, "{0:.2f}".format(face["quality_details"]["n_false_positive"]), fontsize=10, color='yellow')
        ax.text(0, 120, "{0:.2f}".format(face["quality_details"]["avg_others_like_me"]), fontsize=10, color='white')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(image)
        plt.gray()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.draw()  # plt.show()
    return None

def plot_images(people):
    for person in people.values():
        plot_persons_faces(person)
    # people_arr = list(people.values())
    # plot_persons_faces(people_arr[0])
    # plot_persons_faces(people_arr[7])
    # plot_persons_faces(people_arr[8])

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

plot_images(people)
plt.show()

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
#                'landmarks-5': [(80, 52), (66, 54), (31, 52), (44, 54), (56, 84)],
#                'landmarks-68': [(80, 52), (66, 54), (31, 52), (44, 54), (56, 84), ...],
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
#                'landmarks-5': [(80, 52), (66, 54), (31, 52), (44, 54), (56, 84)],
#                'landmarks-68': [(80, 52), (66, 54), (31, 52), (44, 54), (56, 84), ...],
#                'embedding': array([<128-vector embedding>])
#            },
#            ...
#        ]
#    },
#    'm.044wfaf': {...}
#}


