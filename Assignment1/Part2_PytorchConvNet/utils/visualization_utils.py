import numpy as np
import matplotlib.pyplot as plt

def pltClasses(categorized_data, title):
    data = np.ones((10,10,1,28,28))
    for i in range(10):
        num_samples = np.minimum(10, len(categorized_data[i]))
        data[i][:num_samples] = np.squeeze(np.array(categorized_data[i][:num_samples]), axis=(1,))

    for i in range(10):
        for j in range(10):
            plt_idx = j * 10 + i + 1
            ax = plt.subplot(10, 10, plt_idx)
            plt.imshow(data[i,j,0]+0.13087, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            if i == 0 and j == 0:
                plt.title(title, size=14)

    plt.show()