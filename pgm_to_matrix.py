def convert(path):
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    massive_of_images = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'rb') as pgmf:
            im = plt.imread(pgmf)
            im = np.matrix(im)
            massive_of_images.append(im)

    return massive_of_images
