# python train_pretrain.py --gpus 1 --seed 2054 --fold 0 --epochs 10 \
#     --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
#     --model_name efficientvit_l2.r256_in1k --backbone_lr 1e-4 --lr 1e-3 --pos_weight 1 --weight_decay 0.01 \
#     --pos_upsample_rate 1 --logdir efficientvit_l2_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3054 --fold 0 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --backbone_lr 1e-4 --lr 1e-3 --pos_weight 1 --weight_decay 0.01 \
    --resume_dir efficientvit_l2_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 2 --neg_downsample_rate 10 --logdir efficientvit_l2_finetune_pos2x_neg10div_sz128_ep30_bs64


# python train_pretrain.py --gpus 1 --seed 2055 --fold 1 --epochs 10 \
#     --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
#     --model_name efficientvit_l2.r256_in1k --backbone_lr 1e-4 --lr 1e-3 --pos_weight 1 --weight_decay 0.01 \
#     --pos_upsample_rate 1 --logdir efficientvit_l2_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3055 --fold 1 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --backbone_lr 1e-4 --lr 1e-3 --pos_weight 1 --weight_decay 0.01 \
    --resume_dir efficientvit_l2_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 2 --neg_downsample_rate 10 --logdir efficientvit_l2_finetune_pos2x_neg10div_sz128_ep30_bs64


# python train_pretrain.py --gpus 1 --seed 2056 --fold 2 --epochs 10 \
#     --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
#     --model_name efficientvit_l2.r256_in1k --backbone_lr 1e-4 --lr 1e-3 --pos_weight 1 --weight_decay 0.01 \
#     --pos_upsample_rate 1 --logdir efficientvit_l2_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3056 --fold 2 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --backbone_lr 1e-4 --lr 1e-3 --pos_weight 1 --weight_decay 0.01 \
    --resume_dir efficientvit_l2_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 2 --neg_downsample_rate 10 --logdir efficientvit_l2_finetune_pos2x_neg10div_sz128_ep30_bs64


# python train_pretrain.py --gpus 1 --seed 2057 --fold 3 --epochs 10 \
#     --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
#     --model_name efficientvit_l2.r256_in1k --backbone_lr 1e-4 --lr 1e-3 --pos_weight 1 --weight_decay 0.01 \
#     --pos_upsample_rate 1 --logdir efficientvit_l2_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3057 --fold 3 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --backbone_lr 1e-4 --lr 1e-3 --pos_weight 1 --weight_decay 0.01 \
    --resume_dir efficientvit_l2_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 2 --neg_downsample_rate 10 --logdir efficientvit_l2_finetune_pos2x_neg10div_sz128_ep30_bs64


# python train_pretrain.py --gpus 1 --seed 2058 --fold 4 --epochs 10 \
#     --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
#     --model_name efficientvit_l2.r256_in1k --backbone_lr 1e-4 --lr 1e-3 --pos_weight 1 --weight_decay 0.01 \
#     --pos_upsample_rate 1 --logdir efficientvit_l2_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3058 --fold 4 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --backbone_lr 1e-4 --lr 1e-3 --pos_weight 1 --weight_decay 0.01 \
    --resume_dir efficientvit_l2_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 2 --neg_downsample_rate 10 --logdir efficientvit_l2_finetune_pos2x_neg10div_sz128_ep30_bs64