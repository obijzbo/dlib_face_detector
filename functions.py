import cv2
import numpy as np

def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return (startX, startY, w, h)


def detectCartoon1(imagePath):
    img_before = cv2.imread(imagePath)
    img_after = 0

    gray = cv2.GaussianBlur(img_before, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    img_after = cv2.Laplacian(gray, cv2.CV_64F)
    img_after = cv2.convertScaleAbs(img_after)
    return np.mean(img_after)



def detectCartoon2(imagePath):
    img_before = cv2.imread(imagePath)
    img_after = 0

    for i in range(1, 31, 2):
        img_after = cv2.bilateralFilter(img_before, i, i*2, i/2)

    img_after = cv2.cvtColor(img_after, cv2.COLOR_HSV2BGR_FULL)
    img_before = cv2.cvtColor(img_before, cv2.COLOR_HSV2BGR_FULL)

    return np.mean(img_before - img_after)



def is_cartoon(imagepath, threshold: float = 0.98) -> bool:
    # real people images:  79.5%     all: 2149 | cartoon: 440 | real: 1709
    # cartoon images:      59.4%     all: 481 | cartoon: 286 | real: 195

    # read and resize image
    img = cv2.imread(str(imagepath))
    img = cv2.resize(img, (1024, 1024))

    # blur the image to "even out" the colors
    color_blurred = cv2.bilateralFilter(img, 6, 250, 250)

    # compare the colors from the original image to blurred one.
    diffs = []
    for k, color in enumerate(('b', 'r', 'g')):
        # print(f"Comparing histogram for color {color}")
        real_histogram = cv2.calcHist(img, [k], None, [256], [0, 256])
        color_histogram = cv2.calcHist(color_blurred, [k], None, [256], [0, 256])
        diffs.append(cv2.compareHist(real_histogram, color_histogram, cv2.HISTCMP_CORREL))

    return sum(diffs) / 3 > threshold




def is_cartoon_color_count(imagepath, threshold: float = 0.3) -> bool:
    #real:     71%
    #cartoon:  80%
    # Much slower due to CPU overhead of color counting. Recommend multiprocess pool if using

    img = cv2.imread(str(imagepath))
    img = cv2.resize(img, (1024, 1024))
    # img = cv2.bilateralFilter(img, 6, 250, 250)

    # Find count of each color
    color_count = {}
    for row in img:
        for item in row:
            value = tuple(item)
            if value not in color_count:
                color_count[value] = 1
            else:
                color_count[value] += 1

    # Identify the percent of the image that uses the top 512 colors
    most_common_colors = sum([x[1] for x in sorted(color_count.items(), key=lambda pair: pair[1], reverse=True)[:512]])
    return (most_common_colors / (1024 * 1024)) > threshold

def show_img(image_path, bool):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(264,264))
    if bool == True:
        cv2.imshow("Cartoon", img)
        cv2.waitKey(0)
    elif bool == False:
        cv2.imshow("Not Cartoon", img)
        cv2.waitKey(0)
