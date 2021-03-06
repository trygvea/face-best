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


def filter_dataset_errors(people):
    """
    The MS-Celeb dataset have lots of people placed in wrong
    """
    calculate_qualities(people)     # Or use a tailored calculation for filtering???
    print("Filtering dataset for people not belonging")
    for person_id, person in people.items():
        faces = sorted(person["faces"], key=lambda f: -f['quality'])
        n_false_negatives = faces[0]["quality_details"]["n_false_negative"]
        if n_false_negatives > 0:
            print("Removing ",n_false_negatives," false negatives (probably wrong person) from ",person_id)
            person["faces"] = faces[:-n_false_negatives]


def print_best_image(people):
    for person_id, person in people.items():
        faces = sorted(person["faces"], key=lambda f: -f['quality'])
        print("Best image for recognition for "+person_id+" is "+faces[0]["image_id"]+"; having quality:", faces[0]["quality"])


def decorate_image_with_metrics(ax, image, face):
    image_height, image_width, unused = image.shape
    h, w = image_height / 5, image_width / 2   # text should be unaffected by image size (that may be uneven)
    # ax.text(0, 0, "{0:.2f}".format(face["quality"]), fontsize=10, color='red', verticalalignment='bottom')
    ax.text(0, 0,   "{0:.2f}".format(face["quality_details"]["avg_own"]), color='red', verticalalignment='top')
    ax.text(w, 0,   "{0:.2f}".format(face["quality_details"]["avg_oth"]), color='green', verticalalignment='top')
    ax.text(0, h,   "{}/{}".format(face["quality_details"]["n_false_negative"], face["quality_details"]["n_false_positive"]), color='blue', verticalalignment='top')
    ax.text(w, h,   "{0:.2f}".format(face["quality_details"]["avg_others_like_me"]), color='white', verticalalignment='top')


def plot_persons_faces(person):
    faces = sorted(person["faces"], key=lambda f: -f['quality'])    # best quality first
    faces = faces[0:18] + faces[-18:]   # Just some, not all
    num_cols = int(np.sqrt(len(faces))-(1e-7)) + 1
    print("Plotting face for ", person["person_id"])
    plt.figure(person["person_id"], figsize=(num_cols, num_cols))
    for i, face in enumerate(faces):
        image = io.imread(face["path"])
        ax = plt.subplot(num_cols, num_cols, i + 1)
        ax.set_axis_off()
        decorate_image_with_metrics(ax, image, face)
        plt.imshow(image)

    plt.subplots_adjust(wspace=0.1, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.draw()  # plt.show()
    return None

def plot_images(people):
    # for person in people.values():
    #     plot_persons_faces(person)
    people_arr = list(people.values())
    plot_persons_faces(people_arr[1])
    plot_persons_faces(people_arr[5])
    plot_persons_faces(people_arr[11])
    plot_persons_faces(people_arr[12])
    plot_persons_faces(people_arr[13])


people = load_dict(intermediate_file)
#filter_dataset_errors(people)
calculate_qualities(people)

print_best_image(people)

plot_images(people)
plt.show()


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


