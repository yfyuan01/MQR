# The name of experiment
name=VLBart

output=snap/QueryRewrite/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/queryrewrite.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 1 \
        --num_workers 4 \
        --backbone 'facebook/bart-base' \
        --output $output ${@:2} \
        --load snap/pretrain/VLBart/Epoch30 \
        --num_beams 5 \
        --batch_size 64 \
        --valid_batch_size 64 \
        --dump_path result1.pkl \
        