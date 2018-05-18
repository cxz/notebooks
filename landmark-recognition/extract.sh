python3 -m cirtorch.examples.extract \
    --network-path ../retrieval-SfM-120k_resnet101_gem_contrastive_m0.85_adam_lr1.0e-06_wd1.0e-04_nnum5_qsize2000_psize20000_bsize5_imsize/model_best.pth.tar \
    --gpu-id '0' --datasets 'reco' --image-size 300  --multiscale
