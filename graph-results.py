# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Graph results of test
def graph_test(fname, title):
    dir = './results/' + fname + '.csv'
    df = pd.read_csv(dir, header=None, names=['test', 'time', 'sacc', 'uacc'])
    x = np.arange(len(df.index))
    plt.plot(x, df['sacc'] * 100, label='Per-Sample Accuracy')
    plt.plot(x, df['uacc'] * 100, label='Per-User Accuracy')
    plt.xticks(x, [n.replace(" ", "\n") for n in df['test']])
    plt.ylabel('Accuracy (%)')
    plt.title(title + " (N=500)")
    plt.legend()
    plt.savefig('./figures/' + fname + '.png', bbox_inches='tight')
    plt.savefig('./figures/' + fname + '.pdf', bbox_inches='tight')
    plt.clf()

# Graph added noise
graph_test('added-noise', 'Accuracy vs. Added Noise')

# Graph reduced dimensions
graph_test('reduced-dimensions', 'Accuracy vs. Reduced Dimensions')

# Graph reduced FPS
graph_test('reduced-fps', 'Accuracy vs. Reduced FPS')

# Graph added noise
graph_test('reduced-precision', 'Accuracy vs. Reduced Precision')
