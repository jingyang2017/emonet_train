# Estimation of continuous valence and arousal levels from faces in naturalistic conditions, Nature Machine Intelligence 2021

Unofficial implementation of the paper _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021.

** Code is partly forked/copied from the official code of [emonet](https://github.com/face-analysis/emonet)**

# Traing and evaluation
## step1:
`python train.py --nclassses 5 `
## step2:
`python train.py --nclassses 5 --kd --kd_w 0.3 --path step1_model_path`

