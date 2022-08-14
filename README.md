# Unofficial implementation of paper: Estimation of continuous valence and arousal levels from faces in naturalistic conditions
Code is partly forked/copied from the official code of [emonet](https://github.com/face-analysis/emonet)

# Training and evaluation
## step1:
`python train.py --nclassses 5 `
## step2:
`python train.py --nclassses 5 --kd --kd_w 0.3 --path step1_model_path`

# pretrained model 
`https://github.com/jingyang2017/emonet_test/tree/master/ibug/emotion_recognition/emonet/weights`
## References
\[1\] Toisoul, Antoine, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos, and Maja Pantic. "[Estimation of continuous valence and arousal levels from faces in naturalistic conditions.](https://rdcu.be/cdnWi)" _Nature Machine Intelligence_ 3, no. 1 (2021): 42-50.

