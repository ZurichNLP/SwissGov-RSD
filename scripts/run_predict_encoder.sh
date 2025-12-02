for type in diffalign; do
  for model in sentence-transformers/LaBSE; do
    CUDA_VISIBLE_DEVICES=0,1 python -m scripts.predict_encoder $model --type $type --test_data rsd --split full
  done
done

# for type in finetuned; do
#   for model in out/ModernBERT-large-de-projected; do
#     CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m scripts.predict_encoder $model --type $type
#   done
# done
