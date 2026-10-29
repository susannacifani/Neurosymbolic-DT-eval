import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def pad_list(lst, l):
    if len(lst) < l:
        num_to_pad = l - len(lst)
        padded_list = lst + [lst[-1]] * num_to_pad
        return padded_list
    else:
        return lst

def plot(dir_name, num_exp, formula = None, max_reward=100, type='rewards'):
    if type == 'rewards':
        results = []
        for exp in range(num_exp):
            with open("{}/train_rewards_{}.txt".format(dir_name, exp), "r") as f:
                lines = f.readlines()
            lines = [float(line.strip()) for line in lines]
            lines = pad_list(lines, 10000)
            lines = np.convolve(lines, np.ones(100)/100, mode='valid')
            results.append(lines)
        results = np.array(results)
        df = None
        df = pd.DataFrame(results).melt()
        sns.lineplot(x="variable", y="value", data=df)
        plt.title("{}".format(formula[2]))
        plt.axhline(y=max_reward, color='r', linestyle='--')
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.savefig(dir_name+"/Rewards_seaborn_plot.png")
        plt.clf()
    
    elif type == 'image_accuracy':
        results = []
        for exp in range(num_exp):
            with open("{}/image_classification_accuracy_{}.txt".format(dir_name, exp), "r") as f:
                lines = f.readlines()
            lines = [float(line.strip()) for line in lines]
            lines = pad_list(lines, 10000)
            lines = np.convolve(lines, np.ones(100)/100, mode='valid')
            results.append(lines)
        results = np.array(results)
        df = None
        df = pd.DataFrame(results).melt()
        sns.lineplot(x="variable", y="value", data=df)
        plt.title("{}".format(formula[2]))
        plt.xlabel("Episodes")
        plt.ylabel("Image classification accuracy")
        plt.savefig(dir_name+"/Image_accuracy_seaborn_plot.png")
        plt.clf()

    elif type == 'sequence_accuracy':
        results = []
        for exp in range(num_exp):
            with open("{}/sequence_classification_accuracy_{}.txt".format(dir_name, exp), "r") as f:
                lines = f.readlines()
            lines = [float(line.strip()) for line in lines]
            lines = pad_list(lines, 10000)
            lines = np.convolve(lines, np.ones(100)/100, mode='valid')
            results.append(lines)
        results = pad_list(results, 10000)
        results = np.array(results)
        df = None
        df = pd.DataFrame(results).melt()
        sns.lineplot(x="variable", y="value", data=df)
        plt.title("{}".format(formula[2]))
        plt.xlabel("Episodes")
        plt.ylabel("Sequence classification accuracy")
        plt.savefig(dir_name+"/Sequence_accuracy_seaborn_plot.png")
        plt.clf()