# Segment-Anything Demo

Please refer to the [README](./checkpoints/README.md) to prepare the pretrained checkpoints.

# Run demo
## 1. Segment Anything
For SAM, you can refer to the following command to perform segmentation with bbox-prompts. However, there are yet
some unknown bugs in `demo_sam.py`.

```Shell
python demo_sam.py --mode vit_t --checkpoint ./checkpoints/sam/sam_vit_b_01ec64.pth --device cuda --show

```

You also can refer to the following command to perfrom automatic mask generation.

```Shell
python amg_sam.py --mode vit_t --checkpoint ./checkpoints/sam/sam_vit_b_01ec64.pth --device cuda --show

```


## 2. Efficient Segment Anything
For Efficient SAM, you can refer to the following command to perform efficient segmentation with bbox-prompts.

```Shell
python demo_efficient_sam.py --mode vit_t --checkpoint ./checkpoints/efficient_sam/efficient_sam_vitt.pt --device cuda --show

```