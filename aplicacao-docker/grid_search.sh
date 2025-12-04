
#!/bin/bash

# Hiperparâmetros

TIPO_DADOS=("AP-COV" "Autoencoder" "SUB-IT")
OVERSAMPLE=(true false)
# Rede Neural
H_LAYER_SIZE=(16 32)
H_LAYER_NUM=(1 2)
REGULARIZER=("L1" "L2")
REGULARIZER_LAMBDA=(1e-2 1e-3)
EPOCHS=(1 5 15)
# Regressão Logística
PENALTY=("l1" "l2")
C=(0.01 0.1 1.0 10)
MAX_ITER=(1 5 15)

# logs e resultados
mkdir -p logs
RESULTS_FILE_RN="results_rn.csv"
RESULTS_FILE_RL="results_rl.csv"
echo "tipo_dados", "oversample", "h_layer_size", "h_layer_num", "regularizer","regularizer_lambda", "epochs", "f1", "loss" > "$RESULTS_FILE_RN"
echo "tipo_dados", "oversample", "penalty", "c", "max_iter", "f1", "loss" > "$RESULTS_FILE_RL"

# Grid search da rede neural
for td in "${TIPO_DADOS[@]}"; do
  for os in "${OVERSAMPLE[@]}"; do
    for hl_size in "${H_LAYER_SIZE[@]}"; do
      for hl_num in "${H_LAYER_NUM[@]}"; do
        for reg in "${REGULARIZER[@]}"; do
          for reg_lambda in "${REGULARIZER_LAMBDA[@]}"; do
            for ep in "${EPOCHS[@]}"; do

               if [[ "$td" == "AP-COV" ]]; then
                  n_var=24
               elif [[ "$td" == "Autoencoder" ]]; then
                  n_var=30
               elif [[ "$td" == "SUB-IT" ]]; then
                  n_var=40
               fi
               # Arquivos de logs
               LOG_FILE="logs/td${td}_os${os}_hls${hl_size}_hln${hl_num}_r${reg}_rl${reg_lambda}_ep${ep}.log"

               echo "Avaliando com hiperparâmetros td=$td, os=$os, hls=$hl_size, hln=$hl_num, r=$reg, rl = $reg_lambda, ep=$ep"

               # Rodar Flower
               flwr run . local-deployment --stream --run-config  "num-server-rounds=5 algoritmo='Rede Neural' tipo_dados='$td' oversample=$os hidden_layer_size=$hl_size hidden_layer_num=$hl_num regularizer='$reg' regularizer_lambda=$reg_lambda epochs=$ep n_variaveis=$n_var" > "$LOG_FILE" 2>&1

               # Recuperar métricas finais
               F1=$(grep "F1-MACRO PÓS-AGREGAÇÃO DA RODADA =" "$LOG_FILE" | tail -1 | cut -d'=' -f2 | awk '{print $1}')
               LOSS=$(grep "LOSS PÓS-AGREGAÇÃO DA RODADA =" "$LOG_FILE" | tail -1 | cut -d'=' -f2 | awk '{print $1}')
               echo "$F1 e $LOSS"

               # Salvar no arquivo csv
               echo "$td, $os, $hl_size, $hl_num, $reg, $reg_lambda, $ep, $F1, $LOSS" >> "$RESULTS_FILE_RN"
            done
          done
        done
      done
    done
  done
done

# Grid search da regressão logística
for td in "${TIPO_DADOS[@]}"; do
  for os in "${OVERSAMPLE[@]}"; do
    for p in "${PENALTY[@]}"; do
      for c in "${C[@]}"; do
        for mi in "${MAX_ITER[@]}"; do

               if [[ "$td" == "AP-COV" ]]; then
                  n_var=24
               elif [[ "$td" == "Autoencoder" ]]; then
                  n_var=30
               elif [[ "$td" == "SUB-IT" ]]; then
                  n_var=40
               fi

               # Arquivos de logs
               LOG_FILE="logs/td${td}_os${os}_p${p}_c${c}_mi${mi}.log"

               echo "Avaliando com hiperparâmetros td=$td, os=$os, p=$p, c=$c, mi=$mi"

               # Rodar Flower
               flwr run . local-deployment --stream --run-config "num-server-rounds=5 algoritmo='Regressão Logística' tipo_dados='$td' oversample=$os penalty='$p' C=$c max_iter=$mi n_variaveis=$n_var"> "$LOG_FILE" 2>&1

               # Recuperar métricas finais
               F1=$(grep "F1-MACRO PÓS-AGREGAÇÃO DA RODADA =" "$LOG_FILE" | tail -1 | cut -d'=' -f2 | awk '{print $1}')
               LOSS=$(grep "LOSS PÓS-AGREGAÇÃO DA RODADA =" "$LOG_FILE" | tail -1 | cut -d'=' -f2 | awk '{print $1}')
               echo "$F1 e $LOSS"

               # Salvar no arquivo csv
               echo "$td, $os, $p, $c, $mi, $F1, $LOSS" >> "$RESULTS_FILE_RL"
        done
      done
    done
  done
done
