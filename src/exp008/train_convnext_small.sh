# python train_pretrain.py --gpus 1 --seed 2024 --fold 0 --epochs 30 \
#     --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
#     --model_name convnext_small.fb_in1k --drop_path_rate 0.3 --lr 1e-3 --pos_weight 1 \
#     --pos_upsample_rate 1 --logdir convnext_small_rocstar_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3024 --fold 0 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --drop_path_rate 0.3 --backbone_lr 5e-4 --lr 1e-3 --pos_weight 1 \
    --resume_dir convnext_small_rocstar_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 10 --neg_downsample_rate 10 --logdir convnext_small_rocstar_finetune_pos10x_neg10div_sz128_ep30_bs64

python train_pretrain.py --gpus 1 --seed 2025 --fold 1 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --model_name convnext_small.fb_in1k	 --drop_path_rate 0.3 --lr 1e-3 --pos_weight 1 \
    --pos_upsample_rate 1 --logdir convnext_small_rocstar_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3025 --fold 1 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --drop_path_rate 0.3 --backbone_lr 5e-4 --lr 1e-3 --pos_weight 1 \
    --resume_dir convnext_small_rocstar_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 10 --neg_downsample_rate 10 --logdir convnext_small_rocstar_finetune_pos10x_neg10div_sz128_ep30_bs64


python train_pretrain.py --gpus 1 --seed 2026 --fold 2 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --model_name convnext_small.fb_in1k	 --drop_path_rate 0.3 --lr 1e-3 --pos_weight 1 \
    --pos_upsample_rate 1 --logdir convnext_small_rocstar_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3026 --fold 2 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --drop_path_rate 0.3 --backbone_lr 5e-4 --lr 1e-3 --pos_weight 1 \
    --resume_dir convnext_small_rocstar_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 10 --neg_downsample_rate 10 --logdir convnext_small_rocstar_finetune_pos10x_neg10div_sz128_ep30_bs64


python train_pretrain.py --gpus 1 --seed 2027 --fold 3 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --model_name convnext_small.fb_in1k	 --drop_path_rate 0.3 --lr 1e-3 --pos_weight 1 \
    --pos_upsample_rate 1 --logdir convnext_small_rocstar_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3027 --fold 3 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --drop_path_rate 0.3 --backbone_lr 5e-4 --lr 1e-3 --pos_weight 1 \
    --resume_dir convnext_small_rocstar_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 10 --neg_downsample_rate 10 --logdir convnext_small_rocstar_finetune_pos10x_neg10div_sz128_ep30_bs64


python train_pretrain.py --gpus 1 --seed 2028 --fold 4 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --model_name convnext_small.fb_in1k	 --drop_path_rate 0.3 --lr 1e-3 --pos_weight 1 \
    --pos_upsample_rate 1 --logdir convnext_small_rocstar_pretrain_sz128_ep30_bs64

python train_finetune.py --gpus 1 --seed 3028 --fold 4 --epochs 30 \
    --batch_size 64 --image_size 128 --num_workers 24 --precision 16-mixed \
    --drop_path_rate 0.3 --backbone_lr 5e-4 --lr 1e-3 --pos_weight 1 \
    --resume_dir convnext_small_rocstar_pretrain_sz128_ep30_bs64 \
    --pos_upsample_rate 10 --neg_downsample_rate 10 --logdir convnext_small_rocstar_finetune_pos10x_neg10div_sz128_ep30_bs64