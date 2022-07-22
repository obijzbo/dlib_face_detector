import os
import cv2
from functions import is_cartoon, show_img, is_cartoon_color_count

root_dir = os.getcwd()
image_folder = "DATA/is_cartoon"
# cartoon = 0
# not_cartoon = 0

try:
    print("[INFO] Loading images ......")
    image_list = os.listdir(f"{root_dir}/{image_folder}")
    for image_var in image_list:
        image_directory = f"{root_dir}/{image_folder}/{image_var}"
        if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
            show_img(image_directory,is_cartoon(image_directory))
            # if is_cartoon(image_directory) == True:
            #     cartoon+=1
            #     cv2.imshow("Cartoon", image_directory)
            #     cv2.waitKey(0)
            # elif is_cartoon(image_directory) == False:
            #     not_cartoon+=1
            #     cv2.imshow("Not Cartoon", image_directory)
            #     cv2.waitKey(0)
    # print("Number of cartoon image : "+cartoon)
    # print("Number of non cartoon image : "+not_cartoon)
    for image_var in image_list:
        image_directory = f"{root_dir}/{image_folder}/{image_var}"
        if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
            show_img(image_directory,is_cartoon_color_count(image_directory))

except Exception as e:
    print(f"Error : {e}")