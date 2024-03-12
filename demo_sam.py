# --------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------

import os
import cv2
import time
import numpy as np
import argparse
from torchvision import transforms
import torch

from models.build_sam import sam_model_registry


# ---------------------------- Basic parser parameter ----------------------------
parser = argparse.ArgumentParser(description=("Runs automatic mask generation on an input image or directory of images, "
                                              "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
                                              "as well as pycocotools if saving in RLE format."),
                                              )

parser.add_argument("--input", type=str, required=True,
                    help="Path to either a single input image or folder of images.",
                    )

parser.add_argument("--output", type=str, required=True,
                    help=("Path to the directory where masks will be output. Output will be either a folder "
                          "of PNGs per image or a single json with COCO-style masks."),
                    )

parser.add_argument("--model-type", type=str, required=True,
                    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
                    )

parser.add_argument("--checkpoint", type=str, required=True,
                    help="The path to the SAM checkpoint to use for mask generation.",
                    )

parser.add_argument("--device", type=str, default="cuda",
                    help="The device to run generation on.")

parser.add_argument("--convert-to-rle", action="store_true",
                    help=("Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
                          "Requires pycocotools."),
                    )

parser.add_argument("--show", action="store_true",
                    help=("To show the segmentation results on the input image."),
                    )


def main(args):
    # Build the SAM
    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model.to(args.device)

    # load an image
    sample_image_np = cv2.imread("data/images/ex1.jpg")
    sample_image_np = cv2.cvtColor(sample_image_np, cv2.COLOR_BGR2RGB)
    original_size = (sample_image_np.shape[0], sample_image_np.shape[1])
    sample_image_tensor = transforms.ToTensor()(sample_image_np)

    # bboxes of the sample
    bboxes = [[236.8000,  82.8916, 325.1200, 441.4458],]
    
    # convert the bboxes into the point prompts
    input_points = torch.as_tensor(bboxes)  # [bs, 4], bs = 1
    input_labels = torch.tensor([[2, 3]])   # top-left, bottom-right
    
    batched_input = [
        {'image': sample_image_tensor.to(args.device),
         'original_size': original_size,
         'boxes': input_points.to(args.device),
         'point_labels': input_labels.to(args.device)},
    ]

    start_time = time.time()
    outputs = model(batched_input, False)
    # [bs, C, H, W]
    mask = outputs[0]['masks'].cpu().numpy().astype(np.uint8)
    mask = mask[0, 0]
    print(" - Infer time: {:3f} s".format(time.time() - start_time))
    
    # ----------- visualize masks -------------
    if args.show:
        masked_image_np = cv2.cvtColor(sample_image_np, cv2.COLOR_RGB2BGR)
        color = [(np.random.randint(255), np.random.randint(255), np.random.randint(255))]
        # [H, W] -> [H, W, 1]         
        mask = np.repeat(mask[..., None], 3, axis=-1)
        mask_rgb = mask * color * 0.6
        inv_alph_mask = (1 - mask * 0.6)
        masked_image_np = (masked_image_np * inv_alph_mask +  mask_rgb).astype(np.uint8)
        cv2.imshow("masked image", masked_image_np)
        cv2.waitKey(0)

    # save the results
    os.makedirs("outputs/sam/", exist_ok=True)
    masked_image_np = masked_image_np.copy().astype(np.uint8)
    cv2.imwrite("outputs/sam/result.png", masked_image_np)


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(12)

    main(args)
