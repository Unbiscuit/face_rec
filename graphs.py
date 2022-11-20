from os import listdir, path
from shutil import move
from random import randint, choice, sample
import matplotlib.pyplot as plt
import numpy as np
import cv2
import methods


try:
    """accuracies_1 = []
    accuracies_2 = []
    accuracies_3 = []
    accuracies_4 = []
    accuracies_5 = []
    accuracies_6 = []
    accuracies_7 = []
    accuracies_8 = []
    accuracies_9 = []"""
    accuracies = []
    for number_of_ets in range(1, 10):
        truths = 0
        for i in range(1, 41):
            reference = i
            for j in range(1, 11):
                number_of_image = j
                for k in range(1, 41):
                    if k != reference:
                        move(f'Data_base/s{k}/{number_of_image}.pgm', f'used_images/s{k}')
                    else:
                        move(f'Data_base/s{k}/{number_of_image}.pgm', 'Check_list')
                for k in range(1, 41):
                    for file_name in sample(listdir(f'Data_base/s{k}'), 9 - number_of_ets):
                        move(path.join(f'Data_base/s{k}', file_name), f'used_images/s{k}')

                answer = methods.hog(number_of_ets)

                if answer[0][0] == reference:
                    truths += 1

                for k in range(1, 41):
                    if k != reference:
                        move(f'used_images/s{k}/{number_of_image}.pgm', f'Data_base/s{k}')
                    else:
                        move(f'Check_list/{number_of_image}.pgm', f'Data_base/s{k}')
                for k in range(1, 41):
                    for file_name in sample(listdir(f'used_images/s{k}'), 9 - number_of_ets):
                        move(path.join(f'used_images/s{k}', file_name), f'Data_base/s{k}')
        accuracies.append(truths/400)
        """if number_of_ets == 1:
            accuracies_1.append(truths/400)
        if number_of_ets == 2:
            accuracies_2.append(truths/400)
        if number_of_ets == 3:
            accuracies_3.append(truths/400)
        if number_of_ets == 4:
            accuracies_4.append(truths/400)
        if number_of_ets == 5:
            accuracies_5.append(truths/400)
        if number_of_ets == 6:
            accuracies_6.append(truths/400)
        if number_of_ets == 7:
            accuracies_7.append(truths/400)
        if number_of_ets == 8:
            accuracies_8.append(truths/400)
        if number_of_ets == 9:
            accuracies_9.append(truths/400)"""
except:
    for i in range(1, 41):
        if i != reference:
            move(f'used_images/s{i}/{number_of_image}.pgm', f'Data_base/s{i}')
        else:
            move(f'Check_list/{number_of_image}.pgm', f'Data_base/s{i}')
    for k in range(1, 41):
        for file_name in sample(listdir(f'used_images/s{k}'), 9 - number_of_ets):
            move(path.join(f'used_images/s{k}', file_name), f'Data_base/s{k}')


plt.figure(figsize=(18,9), dpi=80)
#plt.plot(np.arange(10, 101, 10), np.array(accuracies_1), color='tab:red', label='number of references = 1')
#plt.plot(np.arange(10, 101, 10), np.array(accuracies_2), color='tab:green', label='number of references = 2')
#plt.plot(np.arange(10, 101, 10), np.array(accuracies_3), color='tab:blue', label='number of references = 3')
#plt.plot(np.arange(10, 101, 10), np.array(accuracies_4), color='tab:orange', label='number of references = 4')
#plt.plot(np.arange(10, 101, 10), np.array(accuracies_5), color='tab:purple', label='number of references = 5')
#plt.plot(np.arange(10, 101, 10), np.array(accuracies_6), color='tab:brown', label='number of references = 6')
#plt.plot(np.arange(10, 16, 1), np.array(accuracies_7), color='tab:pink', label='number of references = 7')
#plt.plot(np.arange(10, 16, 1), np.array(accuracies_8), color='tab:gray', label='number of references = 8')
#plt.plot(np.arange(10, 16, 1), np.array(accuracies_9), color='tab:olive', label='number of references = 9')
plt.plot(np.arange(1, 10, 1), np.array(accuracies), color='tab:red')

plt.title("dispersion of accuracy", fontsize=16)
plt.xlabel("number of references", fontsize=14)
plt.ylabel("accuracy", fontsize=14)
plt.xticks(np.arange(1, 10, 1))
plt.grid(axis='both', alpha=.3)
#plt.legend()

plt.gca().spines["top"].set_alpha(0.0)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)
plt.gca().spines["left"].set_alpha(0.3)
plt.show()


