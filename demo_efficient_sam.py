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

from models.build_efficient_sam import efficient_sam_model_registry


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
    # Build the EfficientSAM model.
    model = efficient_sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model.to(args.device)

    # load an image
    sample_image_np = cv2.imread("data/images/ex2.jpg")
    sample_image_np = cv2.cvtColor(sample_image_np, cv2.COLOR_BGR2RGB)
    sample_image_tensor = transforms.ToTensor()(sample_image_np)

    # bboxes of the sample
    bboxes = [[  1.2800, 141.6533,  74.2400, 421.5467],
              [ 85.7600, 194.5600, 296.9600, 474.4533],
              [ 80.6400, 293.5467, 481.2800, 634.8800],
              [408.3200,   1.7067, 633.6000, 638.2933],
              [281.6000,   5.1200, 435.2000, 636.5867],
              [171.5200,  22.1867, 280.3200, 250.8800],
              [ 87.0400,  71.6800, 198.4000, 300.3733],
              [ 72.9600,  90.4533, 131.8400, 235.5200],
              [355.8400,   0.0000, 422.4000, 145.0667],
              [408.3200,  35.8400, 439.0400, 162.1333]]
    num_queries = len(bboxes)
    
    # convert the bboxes into the point prompts
    input_points = torch.as_tensor(bboxes).unsqueeze(0)      # [bs, num_queries, 4], bs = 1
    input_points = input_points.view(-1, num_queries, 2, 2)  # [bs, num_queries, num_pts, 2]
    input_labels = torch.tensor([2, 3])  # top-left, bottom-right
    input_labels = input_labels[None, None].repeat(1, num_queries, 1) # [bs, num_queries, num_pts]

    start_time = time.time()
    predicted_logits, predicted_iou = model(
        sample_image_tensor[None, ...].to(args.device),
        input_points.to(args.device),
        input_labels.to(args.device),
    )
    print(" - Infer time: {:3f} s".format(time.time() - start_time))

    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    ) # [bs, num_queries, num_candidate_masks, img_h, img_w]

    masks = torch.ge(predicted_logits, 0).cpu().detach().numpy()
    masks = masks[0, :, 0, :, :]  # [num_queries, img_h, img_w]

    # ----------- visualize masks -------------
    if args.show:
        masked_image_np = cv2.cvtColor(sample_image_np, cv2.COLOR_RGB2BGR)
        for i in range(num_queries):
            mask = masks[i]
            color = [(np.random.randint(255), np.random.randint(255), np.random.randint(255))]
            # [H, W] -> [H, W, 1]         
            mask = np.repeat(mask[..., None], 3, axis=-1)
            mask_rgb = mask * color * 0.6
            inv_alph_mask = (1 - mask * 0.6)
            masked_image_np = (masked_image_np * inv_alph_mask +  mask_rgb).astype(np.uint8)
        cv2.imshow("masked image", masked_image_np)
        cv2.waitKey(0)

    # save the results
    os.makedirs("outputs/efficient_sam/", exist_ok=True)
    masked_image_np = masked_image_np.copy().astype(np.uint8)
    cv2.imwrite("outputs/efficient_sam/result.png", masked_image_np)


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(12)

    main(args)
