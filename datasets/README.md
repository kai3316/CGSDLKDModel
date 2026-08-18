# Dataset

The image and label files are not included in this repository. Provide your own YOLO-format dataset and point the config files at its root.

## Directory layout (YOLO format)

```
<dataset root>/
    images/
        train/    2106 images (navel orange)
        val/      603 images
        test/     306 images
    labels/
        train/    one .txt per image, YOLO box format
        val/
        test/
```

## Navel orange orchard dataset

3015 images collected in the experimental orchard of Jiangxi Agricultural University (28.7593N, 115.8416E), split 7:2:1 into 2106 / 603 / 306. Single class. See navel_orange.yaml.

## Public cross-dataset benchmarks (transferability evaluation)

- MinneApple: Hani et al., "MinneApple: A Benchmark Dataset for Apple Detection and Segmentation", https://github.com/nicolaihaeni/MinneApple
- Mango dataset-1: Gu, "Mango dataset-1 (Guiqi, Jinhuang, Tainong, Aomang)", Mendeley Data, doi:10.17632/hmf98f9drg.1

Prepare mango.yaml / minneapple.yaml in the same format and set data: accordingly when training.

## Training settings (from the paper)

| Setting | Value |
|---------|-------|
| Input size | 640x640 |
| Epochs | 200 |
| Batch size | 16 |
| Optimizer | SGD, lr 0.01 |
| Workers | 8 |
| Augmentation | square crop, horizontal flip, brightness/contrast, Gaussian blur (train only) |
