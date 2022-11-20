def hog(number_of_ets=9):
    import numpy as np
    import pgm_to_matrix
    import glob
    preimages = pgm_to_matrix.convert("Check_list")
    images = np.zeros((len(glob.glob('/home/uncookie/PycharmProjects/face_rec/Check_list/*')), 255))
    hist_compare_images = np.zeros((40*number_of_ets, 255))
    cross = 0
    for i in range(len(preimages)):
        for j in range(112):
            for k in range(92):
                images[i][preimages[i][j, k] - 1] += 1
    for i in range(40):
        compare_images = pgm_to_matrix.convert(f'Data_base/s{i + 1}')
        for m in range(len(compare_images)):
            for r in range(112):
                for k in range(92):
                    hist_compare_images[m + cross][compare_images[m][r, k] - 1] += 1
        cross += number_of_ets

    answers = []
    for i in range(len(preimages)):
        distances = np.zeros((1, 40))
        cross = 0
        for k in range(len(hist_compare_images)):
            distances[0][cross] += np.sqrt(np.sum(np.square(np.array(images[i])/255 - np.array(hist_compare_images[k])/255)))
            if (k + 1) % number_of_ets == 0:
                cross += 1
        answers.append(np.unravel_index(np.argmin(distances, axis=None), distances.shape)[1] + 1)
    return answers, images[0]


def answers_for_methods():
    import os
    opened = []
    for i in range(40):
        opened.append(0)
    for filename in os.listdir('Check_list'):
        with open(os.path.join('Check_list', filename), 'rb') as pgmf:
            filename = filename.replace('newfile', '')
            filename = filename.replace('.pgm', '')
            for i in range(len(filename)):
                if filename[i] == '_':
                    opened[int(filename[i+1:]) - 1] = int(int(filename[:i]))
                    break
    return opened


def random(used_dots=400, number_of_ets=9):
    import numpy as np
    import pgm_to_matrix
    import random
    preimages = pgm_to_matrix.convert("Check_list")
    dots = []
    amount_of_dots = used_dots
    vec_compare_images = []
    for i in range(amount_of_dots):
        dots.append([random.randrange(112), random.randrange(92)])
    images = []
    for i in range(len(preimages)):
        vector = []
        for j in range(amount_of_dots):
            vector.append(preimages[i][dots[j][0], dots[j][1]])
        images.append(vector)
    for i in range(40):
        compare_images = pgm_to_matrix.convert(f'Data_base/s{i + 1}')
        for k in range(len(compare_images)):
            vec = []
            for m in range(amount_of_dots):
                vec.append(compare_images[k][dots[m][0], dots[m][1]])
            vec_compare_images.append(vec)
    answers = []
    for i in range(len(images)):
        distances = np.zeros((1, 40))
        cross = 0
        for j in range(len(vec_compare_images)):
            distances[0][cross] += np.sqrt(np.sum(np.square(np.array(images[i]) / 255 - np.array(vec_compare_images[j]) / 255)))
            if (j + 1) % number_of_ets == 0:
                cross += 1
        answers.append(np.unravel_index(np.argmin(distances, axis=None), distances.shape)[1] + 1)
    return answers, dots


def scale(percent=14, reference = 9):
    import numpy as np
    import pgm_to_matrix
    import cv2
    preimages = pgm_to_matrix.convert("Check_list")
    scaled_compare_images = []
    scale_percent = percent
    thing = preimages[0]
    width = int(thing.shape[1] * scale_percent / 100)
    height = int(thing.shape[0] * scale_percent / 100)
    dim = (width, height)
    for i in range(len(preimages)):
        preimages[i] = np.array(cv2.resize(preimages[i], dim, interpolation=cv2.INTER_LINEAR)).flatten()
    for i in range(40):
        compare_images = pgm_to_matrix.convert(f'Data_base/s{i + 1}')
        for j in range(len(compare_images)):
            scaled_compare_images.append(np.array(cv2.resize(compare_images[j], dim, interpolation=cv2.INTER_LINEAR)).flatten())
    answers = []
    for i in range(len(preimages)):
        distances = np.zeros((1, 40))
        cross = 0
        for j in range(len(scaled_compare_images)):
            distances[0][cross] += np.sqrt(np.sum(np.square(np.array(preimages[i]) / 255 - np.array(scaled_compare_images[j]) / 255)))
            if (j + 1) % reference == 0:
                cross += 1
        answers.append(np.unravel_index(np.argmin(distances, axis=None), distances.shape)[1] + 1)
    return answers, preimages[0]



