import pickle
import gzip
import os.path
import random
import numpy as np
import cv2

target_path = "data/mnist_expanded_rotation.pkl.gz"

def expand():
    print("Expanding the MNIST training set")

    if os.path.exists(target_path):
        print("The expanded training set already exists.  Exiting.")
    else:
        with gzip.open("data/mnist.pkl.gz", 'rb') as f:
            training_data, validation_data, test_data = pickle.load(f,encoding='latin1')
        expanded_training_pairs = []
        rotation_mat = []
        degrees = [-8,-4,0,4,8]
        for degree in degrees:
            rotation_mat.append(cv2.getRotationMatrix2D((14,14),degree,1))
        j = 0 # counter
        for x, y in zip(training_data[0], training_data[1]):
            expanded_training_pairs.append((x, y))
            image = np.reshape(x, (28, 28))
            j += 1
            if j % 1000 == 0: print("Expanding image number", j)
            for mat in rotation_mat:
                new_img = cv2.warpAffine(image, mat, (28,28))
                expanded_training_pairs.append((np.reshape(new_img, 784), y))

        random.shuffle(expanded_training_pairs)
        expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
        print("Saving expanded data. This may take a few minutes.")
        with gzip.open(target_path, "w") as f:
            pickle.dump((expanded_training_data, validation_data, test_data), f)


if __name__ == "__main__":
    expand()