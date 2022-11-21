from os import listdir
from shutil import move
from random import randint, choice
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np
import cv2
import methods
from pgm_to_matrix import convert


def move_files_for_work(first_path, second_path, third_path):
    if i != reference:
        move(first_path, second_path)
    else:
        move(first_path, third_path)


def move_files_after_work(first_path, second_path, third_path):
    if i != reference:
        move(first_path, third_path)
    else:
        move(second_path, third_path)


def get_answer(answer_1, answer_2, answer_3):
    if answer_1[0][0] == answer_2[0][0]:
        answer = answer_1
    if answer_1[0][0] == answer_3[0][0]:
        answer = answer_1
    if answer_2[0][0] == answer_3[0][0]:
        answer = answer_2
    if (answer_1[0][0] != answer_2[0][0]) and (answer_2[0][0] != answer_3[0][0]):
        answer = answer_3
    if answer_1[0][0] == answer_2[0][0] == answer_3[0][0]:
        answer = answer_1

    return answer


def get_original_found_images(path_to_origin, path_to_answer):
    with cbook.get_sample_data(path_to_origin) as image_file:
        image_1 = plt.imread(image_file)
    with cbook.get_sample_data(path_to_answer) as image_file:
        image_2 = plt.imread(image_file)

    return image_1, image_2


def get_input_output_hist(path):
    result_matrix = convert(path)
    results_hist = np.zeros((1, 255))
    for j in range(112):
        for k in range(92):
            results_hist[0][result_matrix[0][j, k] - 1] += 1
    input_hist = []
    output_hist = []
    for i in range(len(answer_2[1])):
        for j in range(int(answer_2[1][i])):
            input_hist.append(i)
        for k in range(int(results_hist[0, i])):
            output_hist.append(i)

    return input_hist, output_hist


def get_resized_input_output(path_to_origin, path_to_answer):
    input_img = cv2.imread(path_to_origin, cv2.IMREAD_UNCHANGED)
    output_img = cv2.imread(path_to_answer, cv2.IMREAD_UNCHANGED)
    scale_percent = 25
    width = int(input_img.shape[1] * scale_percent / 100)
    height = int(input_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_input = cv2.resize(input_img, dim, interpolation=cv2.INTER_LINEAR)
    resized_output = cv2.resize(output_img, dim, interpolation=cv2.INTER_LINEAR)

    return resized_input, resized_output


if __name__ == '__main__':
    try:
        plt.ion()
        fig, axes = plt.subplots(4, 2)
        while True:
            reference = randint(1, 40)
            number_of_image = randint(1, 10)

            for i in range(1, 41):
                base_path = f'Data_base/s{i}/{number_of_image}.pgm'
                path_for_not_used_files = f'used_images/s{i}'
                path_for_input_file = 'Check_list'
                move_files_for_work(base_path, path_for_not_used_files, path_for_input_file)

            answer_1 = methods.random()
            answer_2 = methods.hog()
            answer_3 = methods.scale()
            answer = get_answer(answer_1, answer_2, answer_3)

            name_of_result = choice(listdir(f'Data_base/s{answer[0][0]}'))
            path_to_origin = f'/home/uncookie/PycharmProjects/face_rec/Check_list/{number_of_image}.pgm'
            path_to_answer = f'/home/uncookie/PycharmProjects/face_rec/Data_base/s{answer[0][0]}/{name_of_result}'

            origin, result = get_original_found_images(path_to_origin, path_to_answer)

            path_of_answer = f"Data_base/s{answer[0][0]}"
            input_hist, output_hist = get_input_output_hist(path_of_answer)
            resized_input, resized_output = get_resized_input_output(path_to_origin, path_to_answer)

            pts = np.array(answer_1[1])

            axes[0, 0].imshow(origin, cmap='gray')
            axes[0, 0].set_title('Исходное изображение')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(result, cmap='gray')
            axes[0, 1].set_title('Эталон')
            axes[0, 1].axis('off')

            axes[1, 0].cla()
            axes[1, 0].hist(input_hist, bins=20, edgecolor='black')
            axes[1, 0].set_title('Гистограмма исходного изображения')

            axes[1, 1].cla()
            axes[1, 1].hist(output_hist, bins=20, edgecolor='black', color='red')
            axes[1, 1].set_title('Гистограмма эталона')

            axes[2, 0].imshow(resized_input, cmap='gray')
            axes[2, 0].set_title('Скалированное исходное изображение')
            axes[2, 0].axis('off')

            axes[2, 1].imshow(resized_output, cmap='gray')
            axes[2, 1].set_title('Скалированный эталон')
            axes[2, 1].axis('off')

            axes[3, 0].cla()
            axes[3, 0].imshow(origin, cmap='gray')
            axes[3, 0].scatter(pts[:, 1], pts[:, 0], marker='s', color="blue", s=1)
            axes[3, 0].set_title('Точки на исходном изображении')
            axes[3, 0].axis('off')

            axes[3, 1].cla()
            axes[3, 1].imshow(result, cmap='gray')
            axes[3, 1].scatter(pts[:, 1], pts[:, 0], marker='s', color="red", s=1)
            axes[3, 1].set_title('Точки на эталоне')
            axes[3, 1].axis('off')

            fig.set_figwidth(12)
            fig.set_figheight(6)
            fig.canvas.manager.set_window_title('Lab1')

            fig.tight_layout()
            fig.canvas.draw()

            for i in range(1, 41):
                path_for_unused_images = f'used_images/s{i}/{number_of_image}.pgm'
                path_for_answered_file = f'Check_list/{number_of_image}.pgm'
                destination_path = f'Data_base/s{i}'
                move_files_after_work(path_for_unused_images, path_for_answered_file, destination_path)
            fig.canvas.flush_events()
    except:
        for i in range(1, 41):
            path_for_unused_images = f'used_images/s{i}/{number_of_image}.pgm'
            path_for_answered_file = f'Check_list/{number_of_image}.pgm'
            destination_path = f'Data_base/s{i}'
            move_files_after_work(path_for_unused_images, path_for_answered_file, destination_path)
