import imutils
import time
import dlib
import cv2
import os
import json
# construct the argument parser and parse the arguments
from functions import convert_and_trim_bb
from configs.config_ml_model import STATIC

root_dir = os.getcwd()
# image_folder = "DATA/is_face"
model = "MODEL/mmod_human_face_detector.dat"
evaluation_dir = "evaluation"
evaluation_folder = os.path.basename(STATIC)

# initialize true positive, true negative, false positive, false negative
TP = 0
FP = 0
TN = 0
FN = 0

# load dlib's CNN face detector
print("[INFO] loading CNN face detector...")
detector = dlib.cnn_face_detection_model_v1(model)


try:
	print("[INFO] Loading images ......")
	# image_list = os.listdir(f"{root_dir}/{image_folder}")
	# for image_var in image_list:
	image_folders = os.listdir(f"{root_dir}/{STATIC}")
	for image_folder in image_folders:
		image_list = os.listdir(f"{root_dir}/{STATIC}/{image_folder}")
		path = os.path.basename(image_folder)
		# print(path)
		# image_directory = f"{root_dir}/{image_folder}/{image_var}"
		for image_var in image_list:
			image_directory = f"{root_dir}/{STATIC}/{image_folder}/{image_var}"
			if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True or image_directory.endswith(".jpeg") == True or image_directory.endswith(".JPEG") == True or image_directory.endswith(".png") == True or image_directory.endswith(".PNG") == True:
				# print(image_directory)
				image = cv2.imread(image_directory)
				image = imutils.resize(image, width=600)
				rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				# perform face detection using dlib's face detector
				start = time.time()
				print("[INFO[ performing face detection with dlib...")
				results = detector(rgb)

				if results:
					if path=="Face":
						TP+=1
					elif path=="Not Face":
						FP+=1
				else:
					if path=="Face":
						FN+=1
					elif path=="Not Face":
						TN+=1
				end = time.time()
				print("[INFO] face detection took {:.4f} seconds".format(end - start))
				# convert the resulting dlib rectangle objects to bounding boxes,
				# then ensure the bounding boxes are all within the bounds of the
				# input image
				boxes = [convert_and_trim_bb(image, r.rect) for r in results]
				# loop over the bounding boxes
				for (x, y, w, h) in boxes:
					# draw the bounding box on our image
					cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
				# show the output image
				cv2.imshow("Output", image)
				cv2.waitKey(0)

	confusion_matrix = [[TP, TN],
						[FP, FN]]
	classification_accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
	error_rate = (1 - (TP + TN) / (TP + TN + FP + FN)) * 100
	print("Classification accuracy : "+str(classification_accuracy))
	print("Error rate : "+str(error_rate))
	print("Confusion matrix : ")
	print(" TN "+str(TN)+" | "+" FP "+str(FP))
	print(" ")
	print(" FN "+str(FN)+" | "+" TP "+str(TP))
	confusion_matrix_dict = {
		"Total": TP+TN+FP+FN,
		"TP": TP,
		"TN": TN,
		"FP": FP,
		"FN": FN,
		"Classification Accuracy": classification_accuracy,
		"Error Rate": error_rate
	}
	folder_path = f"{root_dir}/{evaluation_dir}/{evaluation_folder}"
	try:
		# Create target Directory
		os.mkdir(folder_path)
		print("Directory ", folder_path, " Created ")

	except FileExistsError:
		print("Directory ", folder_path, " already exists")
	i = 1
	file_path = f"{root_dir}/{evaluation_dir}/{evaluation_folder}/{i}.txt"
	i = os.path.basename(file_path)
	if os.path.isfile(file_path):
		i=i+1
	file_path = f"{root_dir}/{evaluation_dir}/{evaluation_folder}/{i}.txt"
	try:
		# Create target Directory
		open(file_path, 'a').close
		print("Directory ", file_path, " Created ")

	except FileExistsError:
		print("Directory ", file_path, " already exists")

	with open( file_path, "w") as file:
		file.write(json.dumps(confusion_matrix_dict))


except Exception as e:
	print(f"Error : {e}")