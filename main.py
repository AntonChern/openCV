import math

import numpy as np
import cv2 as cv

name = "river"


def orb_features():
    img = cv.imread(name + '.jpg', 0)
    orb = cv.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 0, 255), flags=0)
    cv.imwrite('orb.jpg', img2)


def sift_features():
    img = cv.imread(name + '.jpg')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img)
    cv.imwrite('sift_keypoints.jpg',img)


def canny_edges():
    img = cv.imread(name + '.jpg',0)
    edges = cv.Canny(img,100,200)
    cv.imwrite('canny_edges.jpg', edges)


def gray_scale():
    image = cv.imread(name + '.jpg')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite('gray.jpg', gray)


def hsv():
    image = cv.imread(name + '.jpg')
    dst = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imwrite('hsv.jpg', dst)


def reflecting():
    image = cv.imread(name + '.jpg')
    reflecting = cv.flip(image, 0)
    cv.imwrite('reflecting.jpg', reflecting)


def flip():
    image = cv.imread(name + '.jpg')
    flip_image = cv.flip(image, 1)
    cv.imwrite('flip.jpg', flip_image)


def rotate45():
    image = cv.imread(name + '.jpg')
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))
    cv.imwrite('rotated45.jpg', rotated)


def rotate(degrees, center):
    image = cv.imread(name + '.jpg')
    (h, w) = image.shape[:2]
    M = cv.getRotationMatrix2D(center, degrees, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))
    cv.imwrite('rotated.jpg', rotated)


def shifting():
    img = cv.imread(name + '.jpg')
    h, w = img.shape[:2]
    translation_matrix = np.float32([[1, 0, 10], [0, 1, 0]])
    dst = cv.warpAffine(img, translation_matrix, (w, h))
    cv.imwrite('shifting.jpg', dst)


def increase_brightness(value):
    img = cv.imread(name + '.jpg')
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    cv.imwrite('brightness.jpg', img)


def contrast():
    img = cv.imread(name + '.jpg')
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    mv = cv.split(img_yuv)
    img_y = mv[0].copy()

    temp = cv.bilateralFilter(mv[0], 9, 50, 50)
    exp = np.power(2, (128.0 - (255 - temp).astype(np.float32)) / 128.0)
    temp = (255 * np.power(img_y.flatten() / 255.0, exp.flatten())).astype(np.uint8)
    temp = temp.reshape(img_y.shape)

    dst = img.copy()

    img_y[img_y == 0] = 1
    for k in range(3):
        val = temp / img_y
        val1 = img[:, :, k].astype(np.int32) + img_y.astype(np.int32)
        val2 = (val * val1 + img[:, :, k] - img_y) / 2
        dst[:, :, k] = val2.astype(np.int32)
    cv.imwrite('contrast.jpg', dst)


def gamma_correction():
    img = cv.imread(name + '.jpg')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mean = np.mean(img_gray)
    gamma = math.log10(0.5) / math.log10(mean / 255)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    res = cv.LUT(img, gamma_table)
    cv.imwrite('gamma_restored.jpg', res)


def histogram_equalization():
    img = cv.imread(name + '.jpg', 0)
    equ = cv.equalizeHist(img)
    cv.imwrite('histogram_equalization.jpg', equ)


def white_balance():
    img = cv.imread(name + '.jpg')

    r, g, b = cv.split(img)
    r_avg = cv.mean(r)[0]
    g_avg = cv.mean(g)[0]
    b_avg = cv.mean(b)[0]

    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg

    r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv.merge([b, g, r])
    cv.imwrite('balance_img.jpg', balance_img)


def threshold():
    img = cv.imread(name + '.jpg')
    ret, binary = cv.threshold(img, 175, 255, cv.THRESH_BINARY)
    cv.imwrite('thresholding.jpg', binary)


def find_contours():
    img = cv.imread(name + '.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv.imwrite('contours.jpg', img)


def filter_contours():
    img = cv.imread(name + '.jpg')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)

    sobel = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
    cv.imwrite('filter_contours.jpg', sobel)


def do_blur():
    img = cv.imread(name + '.jpg')
    img_rst = cv.blur(img, (20, 20))
    cv.imwrite('blur.jpg', img_rst)


def highPassFiltering(img, size):
    h, w = img.shape[0:2]
    h1,w1 = int(h/2), int(w/2)
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0
    return img


def highPassFourier():
    gray = cv.imdecode(np.fromfile(name + '.jpg', dtype=np.uint8), 1)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    img_dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(img_dft)

    dft_shift = highPassFiltering(dft_shift, 200)

    idft_shift = np.fft.ifftshift(dft_shift)
    ifimg = np.fft.ifft2(idft_shift)
    ifimg = np.abs(ifimg)
    cv.imwrite('highPassFourier.jpg', np.int8(ifimg))


def lowPassFiltering(img, size):
    h, w = img.shape[0:2]
    h1,w1 = int(h/2), int(w/2)
    img2 = np.zeros((h, w), np.uint8)
    img2[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1
    img3=img2*img
    return img3


def lowPassFourier():
    gray = cv.imdecode(np.fromfile(name + '.jpg', dtype=np.uint8), 1)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    h, w = gray.shape

    for i in range(3000):
        x = np.random.randint(0, h)
        y = np.random.randint(0, w)
        gray[x, y] = 255

    img_dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(img_dft)

    dft_shift = lowPassFiltering(dft_shift, 200)

    idft_shift = np.fft.ifftshift(dft_shift)
    ifimg = np.fft.ifft2(idft_shift)
    ifimg = np.abs(ifimg)
    cv.imwrite('lowPassFourier.jpg', np.int8(ifimg))


def erode():
    img = cv.imread(name + '.jpg', 1)

    kernel = np.ones((5, 5), 'uint8')

    erode_img = cv.erode(img, kernel, iterations=1)
    cv.imwrite('erode.jpg', erode_img)


def dilate():
    img = cv.imread(name + '.jpg', 1)

    kernel = np.ones((5, 5), 'uint8')

    dilate_img = cv.dilate(img, kernel, iterations=1)
    cv.imwrite('dilate.jpg', dilate_img)


if __name__ == "__main__":
    orb_features()
    sift_features()
    canny_edges()
    gray_scale()
    hsv()
    reflecting()
    flip()
    rotate45()
    rotate(degrees=38, center=(130, 50))
    shifting()
    increase_brightness(100)
    contrast()
    gamma_correction()
    histogram_equalization()
    white_balance()
    threshold()
    find_contours()
    filter_contours()
    do_blur()
    highPassFourier()
    lowPassFourier()
    erode()
    dilate()