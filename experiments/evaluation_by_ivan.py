import os
import numpy as np
import cv2

imgPath = "./imgDir/"  # path to extracted features
lblPath = "./labelDir/"  # path to GT labels
outputFile = 'results.txt'  # output txt file

label = os.listdir(lblPath)
label.sort()
imgs = os.listdir(imgPath)
imgs.sort()

num_of_classes = 3

epsilon = 0.000001

def label_generation(original_label, grayscale_values): #generation of maps of labels (0,1,2,3)
    if len(grayscale_values) == 4:
        crop_of_label = np.copy(original_label)
        zero = crop_of_label == grayscale_values[1]
        crop_of_label[zero] = 0
        one = crop_of_label == grayscale_values[2]
        crop_of_label[one] = 1
        two = crop_of_label == grayscale_values[3]
        crop_of_label[two] = 2
        trash = crop_of_label ==grayscale_values[0]
        crop_of_label[trash] = 3
    else:
        crop_of_label = np.copy(original_label)
        zero = crop_of_label == grayscale_values[0]
        crop_of_label[zero] = 0
        one = crop_of_label == grayscale_values[1]
        crop_of_label[one] = 1
        two = crop_of_label == grayscale_values[2]
        crop_of_label[two] = 2
    return crop_of_label

num_of_img = len(label)
result_file = open(outputFile, "w")
for i in range(0, num_of_img):
    start = [0, 0]

    loaded_lbl = cv2.imread(lblPath + label[i], 0)  # labels load
    print("Label: " + label[i] + " was loaded")
    result_file.write("Label: " + label[i] + " was loaded\n")
    sort_values = sorted(set(loaded_lbl.ravel()))
    loaded_lbl_generated = label_generation(loaded_lbl, sort_values)
    loaded_lbl_generated = loaded_lbl_generated.ravel()

    loaded_img = cv2.imread(imgPath + imgs[i], 0)  # imgs load
    print("Img: " + imgs[i] + " was loaded")
    result_file.write("Img: " + imgs[i] + " was loaded\n")
    sort_values = sorted(set(loaded_img.ravel()))
    loaded_img_generated = label_generation(loaded_img, sort_values)
    loaded_img_generated = loaded_img_generated.ravel()

    correct_org = 0
    total = 0
    for j in range (0, len(loaded_img_generated)): # prochazim pixel po pixelu (ano, je to prasarna)
        if loaded_lbl_generated[j] != 3:
            total += 1
            if loaded_img_generated[j] == loaded_lbl_generated[j]:
                correct_org += 1

    print("-------------------")
    print("Corect: " + str(correct_org))
    print("Total: " + str(total))
    print("Acc: " + str(correct_org / (total + epsilon)))
    print("-------------------")
    result_file.write("-------------------\n")
    result_file.write("Corect: " + str(correct_org) + "\n")
    result_file.write("Total: " + str(total) + "\n")
    result_file.write("Acc: " + str(correct_org / (total + epsilon)) + "\n")
    result_file.write("-------------------\n")
result_file.close()