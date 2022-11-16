import methods
import create_check_list
import matplotlib.pyplot as plt
import numpy as np

accuracies = []
epochs = 2
for j in range(epochs):
    create_check_list.create_check_list()
    answers = methods.scale()
    checker = methods.answers_for_methods()
    for i in range(len(checker)):
        if checker[i] == 0:
            checker.pop(i)
    counter = 0
    for m in range(len(answers)):
        if answers[m] == checker[m]:
            counter += 1
    accuracies.append(counter/40)


fig, ax = plt.subplots()
plt.title("dispersion of accuracy", fontsize=16)
plt.xlabel("iteration", fontsize=14)
plt.ylabel("accuracy", fontsize=14)
plt.text(1, max(accuracies), f'mean = {np.array(accuracies).mean().round(decimals=3)}', fontsize=13, fontweight='bold')

ax.plot(np.arange(1, epochs + 1), np.array(accuracies))
plt.show()


