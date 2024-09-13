# python train_pretrain.py --gpus 1 --seed 2034 --fold 0 --epochs 30 \
#     --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
#     --model_name resnetrs50	 --drop_path_rate 0.3 --lr 1e-3 --pos_weight 1 \
#     --pos_upsample_rate 1 --logdir resnetrs50_rocstar_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3034 --fold 0 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --drop_path_rate 0.3 --backbone_lr 5e-4 --lr 1e-3 --pos_weight 1 \
    --resume_dir resnetrs50_rocstar_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 10 --neg_downsample_rate 10 --logdir resnetrs50_rocstar_finetune_pos10x_neg10div_sz128_ep30_bs64


# python train_pretrain.py --gpus 1 --seed 2035 --fold 1 --epochs 30 \
#     --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
#     --model_name resnetrs50	 --drop_path_rate 0.3 --lr 1e-3 --pos_weight 1 \
#     --pos_upsample_rate 1 --logdir resnetrs50_rocstar_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3035 --fold 1 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --drop_path_rate 0.3 --backbone_lr 5e-4 --lr 1e-3 --pos_weight 1 \
    --resume_dir resnetrs50_rocstar_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 10 --neg_downsample_rate 10 --logdir resnetrs50_rocstar_finetune_pos10x_neg10div_sz128_ep30_bs64


# python train_pretrain.py --gpus 1 --seed 2036 --fold 2 --epochs 30 \
#     --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
#     --model_name resnetrs50	 --drop_path_rate 0.3 --lr 1e-3 --pos_weight 1 \
#     --pos_upsample_rate 1 --logdir resnetrs50_rocstar_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3036 --fold 2 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --drop_path_rate 0.3 --backbone_lr 5e-4 --lr 1e-3 --pos_weight 1 \
    --resume_dir resnetrs50_rocstar_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 10 --neg_downsample_rate 10 --logdir resnetrs50_rocstar_finetune_pos10x_neg10div_sz128_ep30_bs64


# python train_pretrain.py --gpus 1 --seed 2037 --fold 3 --epochs 30 \
#     --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
#     --model_name resnetrs50	 --drop_path_rate 0.3 --lr 1e-3 --pos_weight 1 \
#     --pos_upsample_rate 1 --logdir resnetrs50_rocstar_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3037 --fold 3 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --drop_path_rate 0.3 --backbone_lr 5e-4 --lr 1e-3 --pos_weight 1 \
    --resume_dir resnetrs50_rocstar_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 10 --neg_downsample_rate 10 --logdir resnetrs50_rocstar_finetune_pos10x_neg10div_sz128_ep30_bs64


# python train_pretrain.py --gpus 1 --seed 2038 --fold 4 --epochs 30 \
#     --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
#     --model_name resnetrs50	 --drop_path_rate 0.3 --lr 1e-3 --pos_weight 1 \
#     --pos_upsample_rate 1 --logdir resnetrs50_rocstar_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3038 --fold 4 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --drop_path_rate 0.3 --backbone_lr 5e-4 --lr 1e-3 --pos_weight 1 \
    --resume_dir resnetrs50_rocstar_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 10 --neg_downsample_rate 10 --logdir resnetrs50_rocstar_finetune_pos10x_neg10div_sz128_ep30_bs64