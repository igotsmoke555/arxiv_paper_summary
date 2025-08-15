export WANDB_MODE=disabled

train_data="\
    ../example_data/bge_train_sample_score.jsonl"

# set large epochs and small batch size for testing
num_train_epochs=4
per_device_train_batch_size=4

# set num_gpus to 2 for testing
num_gpus=4

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path BAAI/bge-base-zh-v1.5 \
    --cache_dir $HF_HUB_CACHE \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size 16 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval '为这个句子生成表示以用于检索相关文章：' \
    --query_instruction_format '{}{}' \
    --knowledge_distillation True \
"

training_args="\
    --output_dir ./test_encoder_only_base_bge-base-zh-v1.5 \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../../ds_stage0.json \
    --logging_steps 1 \
    --save_steps 80 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.embedder.encoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
