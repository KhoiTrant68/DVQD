# accelerate launch --multi_gpu train.py --base src/configs/stage1/dual_feat_imagenet.yml   --max_epochs 6 --mode feat --with_tracking
# accelerate launch --multi_gpu train.py --base src/configs/stage1/dual_dynamic_entropy_imagenet.yml   --max_epochs 6 --mode entropy --with_tracking
# accelerate launch --multi_gpu train.py --base src/configs/stage1/dual_fixed_entropy_imagenet.yml   --max_epochs 6 --mode entropy --with_tracking
# accelerate launch --multi_gpu train.py --base src/configs/stage1/triple_feat_imagenet.yml    --max_max_epochs 6 --mode feat --with_tracking
accelerate launch --multi_gpu train.py --base src/configs/stage1/triple_dynamic_entropy_imagenet.yml --max_epochs 6 --mode entropy --with_tracking
# accelerate launch --multi_gpu train.py --base src/configs/stage1/triple_fixed_entropy_imagenet.yml   --max_epochs 6 --mode entropy --with_tracking
