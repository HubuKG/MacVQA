# The name of experiment
name=VQAv2_Our_G5

output=snap/$name


PYTHONPATH=$PYTHONPATH:./src \
torchrun --nproc_per_node=$1 \
    --master_port 29500 \
    src/vqacl_comp.py \
        --distributed --multiGPU \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 3e-5 \
        --epochs 3 \
        --num_workers 4 \
        --backbone '/MAC-T5/models/t5-base'\
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 100 \
        --from_scratch \
        --comp_cate G5
