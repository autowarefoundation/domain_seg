#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import os
import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from Models.inference.domain_seg_infer import DomainSegNetworkInfer


def make_visualization(prediction):

    # Creating visualization object
    shape = prediction.shape
    row = shape[0]
    col = shape[1]
    vis_predict_object = np.zeros(
        (row, col, 3), 
        dtype = "uint8"
    )

    # Assigning background colour
    vis_predict_object[:, :, 0] = 255
    vis_predict_object[:, :, 1] = 93
    vis_predict_object[:, :, 2] = 61

    # Getting foreground object labels
    foreground_lables = np.where(prediction == 1.0)

    # Assigning foreground objects colour
    vis_predict_object[foreground_lables[0], foreground_lables[1], 0] = 28
    vis_predict_object[foreground_lables[0], foreground_lables[1], 1] = 148
    vis_predict_object[foreground_lables[0], foreground_lables[1], 2] = 255
            
    return vis_predict_object

def main(): 

    parser = ArgumentParser()

    parser.add_argument(
        "-p", 
        "--model_checkpoint_path", 
        dest = "model_checkpoint_path", 
        help = "Path to Pytorch checkpoint file to load model dict."
    )
    parser.add_argument(
        "-i", 
        "--input_image_dirpath", 
        dest = "input_image_dirpath", 
        help = "Path to input image directory which will be processed by DomainSeg."
    )
    parser.add_argument(
        "-o",
        "--output_image_dirpath",
        dest = "output_image_dirpath",
        help = "Path to output image directory where visualizations will be saved",
        required = True
    )

    args = parser.parse_args()
    # Arranging I/O dirs
    input_image_dirpath = args.input_image_dirpath
    output_image_dirpath = args.output_image_dirpath
    if (not os.path.exists(output_image_dirpath)):
        os.makedirs(output_image_dirpath)

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = DomainSegNetworkInfer(checkpoint_path = model_checkpoint_path)
    print("DomainSeg model loaded.")
  
    # Transparency factor
    alpha = 0.5

    # # Reading input image
    # print("Reading image...")
    # input_image_filepath = args.input_image_filepath
    # frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # image_pil = Image.fromarray(image)
    # image_pil = image_pil.resize((640, 320))

    # # Run inference and create visualization
    # print("Running inference and creating visualization...")
    # prediction = model.inference(image_pil)
    # vis_obj = make_visualization(prediction)

    # # Resize and display visualization
    # vis_obj = cv2.resize(
    #     vis_obj,
    #     (frame.shape[1], frame.shape[0])
    # )
    # image_vis_obj = cv2.addWeighted(
    #     vis_obj, alpha,
    #     frame, 1 - alpha,
    #     0
    # )
    # cv2.imshow("Prediction objects", image_vis_obj)
    # cv2.waitKey(0)

    # Process through input image dir
    for filename in sorted(os.listdir(input_image_dirpath)):
        if (filename.endswith((".png", ".jpg", ".jpeg"))):

            # Fetch image
            input_image_filepath = os.path.join(
                input_image_dirpath, filename
            )
            img_id = filename.split(".")[0].zfill(3)

            print(f"Reading Image: {input_image_filepath}")
            frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize((640, 320))

            # Inference
            prediction = model.inference(image_pil)

            # Visualization
            vis_obj = make_visualization(prediction)

            # Postprocess
            vis_obj = cv2.resize(
                vis_obj,
                (frame.shape[1], frame.shape[0])
            )
            image_vis_obj = cv2.addWeighted(
                vis_obj, alpha,
                frame, 1 - alpha,
                0
            )

            output_image_filepath = os.path.join(
                output_image_dirpath,
                f"{img_id}_result.png"
            )
            # image_vis_obj.save(output_image_filepath)
            cv2.imwrite(output_image_filepath, image_vis_obj)
        else:
            print(f"Skipping non-image file: {filename}")
            continue


if __name__ == "__main__":
    main()
# %%