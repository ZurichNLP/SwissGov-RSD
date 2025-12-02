# Original llama models
#python -m scripts.predict_llama meta-llama/Llama-3.2-1B-Instruct
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m scripts.predict_llama meta-llama/Llama-3.1-8B-Instruct

# Fine-tuned llama models
# python -m scripts.predict_llama meta-llama/Llama-3.2-1B-Instruct --adapter out/Llama-3.2-1B-Instruct-rsd
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m scripts.predict_llama meta-llama/Llama-3.1-8B-Instruct --adapter out/Llama-3.1-8B-Instruct-rsd


#python -m scripts.predict_llama meta-llama/Llama-3.1-8B-Instruct --adapter out/Llama-3.2-1B-Instruct-rsd
