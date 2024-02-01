import argparse
import os
import nibabel as nib
import numpy as np
import json
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--nifti_path", required=False, help="Path to the ground-truth label in nifti format")
    parser.add_argument("--json_path", required=False, help="Path to the json containing the click coordinates")
    parser.add_argument("--output_path", required=False, help="Path to the output directory with all the json files")
    parser.add_argument("--dice_to_val_clicks_path", required=False, help="Path to store a scatter plot with the dice scores and number of valid clicks as well as the dice_scores and predictions directories")


    args = parser.parse_args()
    return args

def read_clicks(input_dir):
    with open(input_dir, 'r') as f:
        json_data = json.load(f)
    tumor = json_data['tumor']
    background = json_data['background']
    return tumor, background

def read_dice_scores(input_dir):
    dice_scores = {}
    npy_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".npy")]
    for f in npy_files:
        fn = f.split('/')[-1].replace('.npy', '')
        last_dice = np.load(f)[-1]
        dice_scores[fn] = round(last_dice, 2)
    return dice_scores 


def read_nifti(input_dir):
    img = nib.load(input_dir)
    a = np.array(img.dataobj)
    return a

def compute_num_valid_clicks(tumor, nifti_np):
    val_clicks = sum([nifti_np[nifti_np.shape[0] - c[0], c[1], c[2]] for c in tumor]) # need be flipped because of Slicer / MONAILabel interplay when saving clicks
    return val_clicks, len(tumor)




def main():
    args = parse_args()
    markers = ['o', 'x', '*', 'v']
    colors = ['r', 'g', 'b', 'y']
    if args.output_path is not None:
        json_dicts = {}

        # Load all JSON data into a list
        all_json_data = []
        for i in np.arange(1, 5):
            json_path = os.path.join(args.output_path, f'disks_{i}', 'predictions', 'num_val_clicks.json')
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                all_json_data.append(json_data)
                json_dicts[i] = json_data

        # Find common keys
        common_keys = set.intersection(*(set(d.keys()) for d in all_json_data))

        # Plotting
        x_ticks = list(common_keys)

        for key in x_ticks:
                for i, value in enumerate([json_dicts[j][key] for j in np.arange(1, 5)]):
                    if key == x_ticks[-1]:
                        plt.scatter(np.ones(1) * x_ticks.index(key), [value], marker=markers[i], color=colors[i], label=f'A{i+1}')
                    else:
                        plt.scatter(np.ones(1) * x_ticks.index(key), [value], marker=markers[i], color=colors[i])


        # Customize the plot
        plt.ylabel('% of valid clicks')
        plt.xticks(np.arange(len(x_ticks)), labels=x_ticks, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, 'num_val_clicks_all_annotators.pdf'))
    

    if args.dice_to_val_clicks_path is not None:
        for i in np.arange(1, 5):
            dice_scores = read_dice_scores(os.path.join(args.dice_to_val_clicks_path, f'disks_{i}', 'dice_scores'))
            num_val_clicks_fn = os.path.join(args.dice_to_val_clicks_path, f'disks_{i}', 'predictions', 'num_val_clicks.json')
            with open(num_val_clicks_fn, 'r') as f:
                num_val_clicks = json.load(f)

            common_keys = set(dice_scores.keys()) & set(num_val_clicks.keys())

            result_list = [(key, (dice_scores[key], num_val_clicks[key])) for key in common_keys]


            x_values = [entry[1][0] for entry in result_list]
            y_values = [entry[1][1] for entry in result_list]  # Using the first part of each tuple for the y-axis

            # Linear regression
            coefficients = np.polyfit(x_values, y_values, 1)
            polynomial = np.poly1d(coefficients)
            plt.plot(np.arange(0, 1, 0.001), polynomial(np.arange(0, 1, 0.001)), color=colors[i-1], label=f'A{i}', linestyle='--')

            mean_x = np.median(x_values)
            mean_y = np.median(y_values)
            plt.axvline(x=mean_x, color=colors[i-1], linestyle='dotted', label=f'Mean A{i}')
            plt.axhline(y=mean_y, color=colors[i-1], linestyle='dotted')

            plt.scatter(x_values, y_values, marker=markers[i-1], color=colors[i-1])
            plt.xlabel('Dice Score')
            plt.ylabel('% of valid clicks')
            plt.legend()
            plt.tight_layout()

        plt.savefig(os.path.join(args.dice_to_val_clicks_path, 'dice_to_num_val_clicks.pdf'))


    if args.output_path is not None or args.dice_to_val_clicks_path is not None:
        exit()





    json_files = [os.path.join(args.json_path, f) for f in os.listdir(args.json_path) if f.endswith(".json")]
    all_val_clicks, sum_all_clicks = {}, {}
    for json_file in json_files:
        tumor, background = read_clicks(json_file)
        nifti_file = json_file.split('/')[-1].replace('clicks.json', 'label.nii.gz')
        nifti_path = os.path.join(args.nifti_path, nifti_file.split('_label')[0], nifti_file)
        nifti_np = read_nifti(nifti_path)
        val_clicks, all_clicks = compute_num_valid_clicks(tumor, nifti_np)
        print(nifti_file, f'Valid Clicks: {val_clicks} / All Clicks: {all_clicks}')
        all_val_clicks[nifti_file.split('_label')[0]] = int(val_clicks) / int(all_clicks)
        sum_all_clicks[nifti_file.split('_label')[0]] = int(all_clicks)
    print('[SUMMARY]', round(100 * (sum(all_val_clicks.values()) / sum(sum_all_clicks.values())), 4), '% of all clicks are valid.')

    data = np.array(list(all_val_clicks.values()))

    # Create a histogram
    plt.hist(data, bins='auto', edgecolor='black', alpha=0.7)

    # Add a vertical line for the mean
    plt.axvline(x=np.mean(data), color='r', linestyle='--', label='Mean')

    # Add labels and legend
    plt.xlabel('Number of Valid Clicks per Volume')
    plt.xlim(0, 10)
    plt.yticks([])  # Remove y-axis ticks
    plt.legend()
    plt.savefig(os.path.join(args.nifti_path, 'mean_num_val_clicks.pdf'))
    with open(os.path.join(args.nifti_path, 'num_val_clicks.json'), 'w') as json_file:
        json.dump(all_val_clicks, json_file)





if __name__ == "__main__":
    main()
