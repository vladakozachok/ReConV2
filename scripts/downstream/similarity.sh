CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config cfgs/transfer/base/finetune_similarity.yaml \
    --finetune_model \
    --exp_name $2 \
    --ckpts $3 \
    --seed $RANDOM