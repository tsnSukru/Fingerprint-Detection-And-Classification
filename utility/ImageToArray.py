import numpy as np
import cv2
import os

"""
this class is for converting images into numerical arrays before artificial neural networks is trained 
"""
def findSubFolders(folder_path):
    try:
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        return subfolders
    except Exception as e:
        print(f"Hata: {e}")
        return []


def convertToArray(inputBasePath, outputBasePath):
    image_width = 224
    image_height = 224
    color = 1
    classes = findSubFolders(inputBasePath)

    os.chdir(inputBasePath)

    X = []
    Y = []

    i = 0
    for class1 in classes:
        os.chdir(class1)
        for files in os.listdir('./'):
            img = cv2.imread(files, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (image_width, image_height))
            X.append(img)
            Y.append(class1)
            i = i + 1
        os.chdir('..')

    X = np.array(X).reshape(-1, image_width, image_height, color)
    Y = np.array(Y)

    os.chdir(outputBasePath)
    np.save(str(image_width) + 'x' + str(image_height) + '_images', X)
    np.save(str(image_width) + 'x' + str(image_height) + '_labels', Y)

    return (outputBasePath + "\\" + str(image_width) + 'x' + str(image_height) + '_images.npy'), (
            outputBasePath + "\\" + str(image_width) + 'x' + str(image_height) + '_labels.npy')


if __name__ == '__main__':
    convertToArray(inputBasePath=r"write your fingerprint database path here",
                   outputBasePath=r"write here where you want to save the output arrays")
