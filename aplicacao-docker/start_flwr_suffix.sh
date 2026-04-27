#!/bin/bash

# Garantia: pelo menos 1 sufixo deve ser passado!
if [[ $# -lt 1 ]]; then
    echo "Uso: $0 <SUFIXO1> <SUFIXO2> ... <SUFIXON>"
    exit 1
fi

# Cria a rede (ignorar erro se já existe)
docker network create --driver bridge flwr-network 2>/dev/null

# Inicia o Superlink (único)
docker run --rm \
    -p 9091:9091 -p 9092:9092 -p 9093:9093 \
    --network flwr-network \
    --name superlink \
    --detach \
    flwr/superlink:1.18.0 \
    --insecure \
    --isolation \
    process

# Inicia o Serverapp (único)
docker run --rm \
    -e PYTHONHASHSEED=0 -e TF_DETERMINISTIC_OPS=1  -e TF_ENABLE_ONEDNN_OPTS=0 \
    --network flwr-network \
    --name serverapp \
    --detach \
    flwr_serverapp:0.0.1 \
    --insecure \
    --serverappio-api-address superlink:9091

# Para cada sufixo passado (um hospital)
BASE_PORT=9094  # Porta base para supernode
INDEX=1

for SUFIXO in "$@"; do
    SN_PORT=$((BASE_PORT + INDEX - 1))
    DATASET_PATH_TRAIN="/home/erick/dados_TCGA/fl_conj_treino_${SUFIXO}.csv"
    DATASET_PATH_VALID="/home/erick/dados_TCGA/fl_conj_valid_${SUFIXO}.csv"
    DATASET_PATH_TEST="/home/erick/dados_TCGA/fl_conj_test_${SUFIXO}.csv"
    
    # Testa se o arquivo existe
    if [[ ! -f "${DATASET_PATH_TRAIN}" || ! -f "${DATASET_PATH_VALID}" || ! -f "${DATASET_PATH_TEST}" ]]; then
        echo "Aviso: Arquivo(s) não encontrado(s):"
        [[ ! -f "${DATASET_PATH_TRAIN}" ]] && echo "  Faltando: ${DATASET_PATH_TRAIN}"
        [[ ! -f "${DATASET_PATH_VALID}"  ]] && echo "  Faltando: ${DATASET_PATH_VALID}"
        [[ ! -f "${DATASET_PATH_TEST}"  ]] && echo "  Faltando: ${DATASET_PATH_TEST}"
        INDEX=$((INDEX + 1))
        continue
    fi


    # Inicia o supernode do hospital
    docker run --rm \
        -e PYTHONHASHSEED=0 -e TF_DETERMINISTIC_OPS=1  -e TF_ENABLE_ONEDNN_OPTS=0 \
        -p ${SN_PORT}:${SN_PORT} \
        --network flwr-network \
        --name supernode-${SUFIXO} \
        --detach \
        flwr/supernode:1.18.0 \
        --insecure \
        --superlink superlink:9092 \
        --clientappio-api-address 0.0.0.0:${SN_PORT} \
        --isolation process

    # Inicia o clientapp do hospital
    docker run --rm \
        -e PYTHONHASHSEED=0 -e TF_DETERMINISTIC_OPS=1 -e CLIENT_ID=${INDEX} \
        -v ${DATASET_PATH_TRAIN}:/mnt/fl_conj_treino_cliente.csv \
        -v ${DATASET_PATH_VALID}:/mnt/fl_conj_valid_cliente.csv \
        -v ${DATASET_PATH_TEST}:/mnt/fl_conj_test_cliente.csv \
        --network flwr-network \
        --name client-${SUFIXO} \
        --detach \
        flwr_clientapp:0.0.1 \
        --insecure \
        --clientappio-api-address supernode-${SUFIXO}:${SN_PORT}

    INDEX=$((INDEX + 1))
done

echo "Ambiente FLWR iniciado para sufixos:" "$@"
