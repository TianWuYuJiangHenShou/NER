export CUDA_VISIBLE_DEVICES=0
python main.py --model-name MODEL_NAME --vocab token2idx.json tag2idx.json --dropout 0.5
