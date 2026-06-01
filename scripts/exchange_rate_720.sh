export CUDA_VISIBLE_DEVICES=0

model_name=SARAF

for seed in 2021 0 42
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id exchange_rate_720_96 \
      --model $model_name \
      --data exchange_rate \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 96 \
      --learning_rate 0.0001 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --itr 1 \
      --seed $seed \
      --topm 2
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id exchange_rate_720_192 \
      --model $model_name \
      --data exchange_rate \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 192 \
      --learning_rate 0.001 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --itr 1 \
      --seed $seed \
      --topm 2
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id exchange_rate_720_336 \
      --model $model_name \
      --data exchange_rate \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 336 \
      --learning_rate 0.0001 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --itr 1 \
      --seed $seed \
      --time_aware_weight 0.3 \
      --topm 3
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id exchange_rate_720_720 \
      --model $model_name \
      --data exchange_rate \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 720 \
      --learning_rate 0.0001 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --itr 1 \
      --seed $seed \
      --topm 10
done