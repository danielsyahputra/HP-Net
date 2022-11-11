import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

def pickle2list(pickle_file):
    att_outputs = []
    with open(pickle_file, "rb") as f:
        while True:
            try:
                att_dict = pickle.load(f)
            except:
                break
            att_outputs.append(att_dict)
    return att_outputs

def att_plot(model_name, att_dict, plot_mode) -> None:
    img_filename = att_dict["file_name"]
    input_img = cv2.imread(f"data/pa100k/{img_filename}")
    att_size = (input_img.shape[1], input_img.shape[0])
    att_level_num = len(att_dict) - 1
    att_channel_num = 8

    # Plot initial image
    plt.subplot(att_level_num + 1, att_channel_num, 1)
    plt.imshow(input_img)
    plt.axis("off")

    # Color map
    color_map = np.uint8([[250], [180], [120], [60], [0]])
    plt.subplot(att_level_num + 1, att_channel_num, 2)
    plt.imshow(cv2.resize(color_map, att_size))
    plt.axis("off")

    # Plot attention
    for att_idx in range(att_level_num):
        for channel_idx in range(att_channel_num):
            if model_name == "HP":
                att_pm = att_dict[f"AF{att_idx+1}"]
                att = np.uint8(255 * cv2.resize(att_pm[channel_idx], att_size) / np.max(att_pm))
            else:
                att = np.uint8(255 * cv2.resize(att_dict[model_name][channel_idx], att_size) / np.max(att_dict[model_name]))
            
            plt.subplot(att_level_num + 1, att_channel_num, (att_idx + 1) * 8 + channel_idx + 1)
            plt.imshow(att)
            plt.axis("off")

    if plot_mode == "img_show":
        plt.axis("off")
        plt.show()
    elif plot_mode == "img_save":
        folder_name = f"results/attention/{model_name}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        plt.axis("off")

        plt.savefig(f"{folder_name}/{img_filename[:-4]}.png")


if __name__=="__main__":
    model_name = "HP"
    outputs = pickle2list("results/att_output_AF1.pkl")
    for output in outputs:
        att_plot(model_name, att_dict=output, plot_mode="img_save")