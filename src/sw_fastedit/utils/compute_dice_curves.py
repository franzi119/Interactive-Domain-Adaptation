import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-i", "--input_dir", required=True, help="Base folder for dice npy files")
    parser.add_argument("-o", "--output_dir", required=True, help="All the dice plots will be stored here")
    args = parser.parse_args()
    return args

def read_npys(args):
    input_dir = args.input_dir
    npy_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".npy")]
    dice_scores = np.array([np.load(f) for f in npy_files])
    return dice_scores

def compute_mean_std(dice_scores):
    mean_dice = np.mean(dice_scores, axis=0)
    std_dice = np.std(dice_scores, axis=0)  

    return mean_dice, std_dice

def plot_dice_scores(mean_dice, std_dice, args):
    x_values = np.arange(1, len(mean_dice) + 1)  # Assuming x-axis is from 1 to the length of mean_dice

    plt.plot(x_values, mean_dice, label='Mean Dice')
    plt.fill_between(x_values, mean_dice - std_dice, mean_dice + std_dice, alpha=0.2, label='Standard Deviation')

    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Mean and Standard Deviation of Dice Scores')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'mean_std_dice.pdf'))




def main():
    args = parse_args()
    dice_scores = read_npys(args)
    mean_dice, std_dice = compute_mean_std(dice_scores)
    plot_dice_scores(mean_dice, std_dice, args)


if __name__ == "__main__":
    main()