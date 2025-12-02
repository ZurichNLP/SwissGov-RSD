for lr in 1e-5 3e-5 5e-5 8e-5; do
    python -m encoders.finetuning.train_modernbert \
        --model_name_or_path answerdotai/ModernBERT-large \
        --output_dir out/ModernBERT-large-rsd_lr${lr} \
        --learning_rate ${lr} \
        --num_train_epochs 4 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --per_device_eval_batch_size 16 \
        --logging_steps 25 \
        --evaluation_strategy steps \
        --eval_steps 25 \
        --save_strategy steps \
        --load_best_model_at_end True \
        --metric_for_best_model eval_loss \
        --save_total_limit 1
done
