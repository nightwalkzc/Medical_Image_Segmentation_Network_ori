import cv2
import numpy as np
from PIL import Image


def get_largest_connected(img):

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, markers = cv2.connectedComponents(gray, connectivity=8)
    cnt = np.bincount(markers.flatten())

    k, num = 1, -1
    cnt[cnt == cnt.max(initial=0)] = -100

    for idx in range(len(cnt)):
        if cnt[idx] > num:
            k = idx
            num = cnt[idx]

    markers[markers == k] = 255
    markers[markers < 255] = 0

    return markers
    # return Image.fromarray(markers).convert('L')


def get_connectArea(path):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, markers = cv2.connectedComponents(gray, connectivity=8)
    return markers


def get_contours(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return img, contours, hierarchy


def get_center(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    centers = []

    contours, cnt = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        M = cv2.moments(cont)
        if M['m00'] <= 0.:
            continue
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        centers.append((center_x, center_y))

    return centers


def fill(img):

    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return Image.fromarray(img).convert('L')
