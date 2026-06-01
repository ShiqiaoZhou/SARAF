export CUDA_VISIBLE_DEVICES=0

model_name=SARAF

for seed in 2021 0 42
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_720_96 \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 96 \
      --learning_rate 0.001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --topm 20


    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_720_192 \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 192 \
      --learning_rate 0.001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --topm 10

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_720_336 \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 336 \
      --learning_rate 0.0001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --topm 20

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_720_720 \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 720 \
      --learning_rate 0.0001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --topm 10

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_720_96 \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 96 \
      --learning_rate 0.01 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --topm 10

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_720_192 \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 192 \
      --learning_rate 0.001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --topm 10

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_720_336 \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 336 \
      --learning_rate 0.001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --topm 20

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_720_720 \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 720 \
      --learning_rate 0.0001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.5 \
      --topm 2

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_720_96 \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 96 \
      --learning_rate 0.001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.1 \
      --topm 20

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_720_192 \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 192 \
      --learning_rate 0.0001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.1 \
      --topm 20

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_720_336 \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 336 \
      --learning_rate 0.0001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.1 \
      --topm 20

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_720_720 \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 720 \
      --learning_rate 0.001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.3 \
      --topm 20

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_720_96 \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 96 \
      --learning_rate 0.001\
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.9 \
      --topm 10

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_720_192 \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 192 \
      --learning_rate 0.001\
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.3 \
      --topm 20

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_720_336 \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 336 \
      --learning_rate 0.0001\
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.9 \
      --topm 20

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_720_720 \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len 720 \
      --label_len 48 \
      --pred_len 720 \
      --learning_rate 0.001\
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.5 \
      --topm 5
done