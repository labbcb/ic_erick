#!/bin/bash

# Número de clientes/hospitais
N_CLIENTES=${1:-2}  # Default: 2, ou defina via argumento

# Cria a rede (ignora erro se já existe)
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
    --network flwr-network \
    --name serverapp \
    --detach \
    flwr_serverapp:0.0.1 \
    --insecure \
    --serverappio-api-address superlink:9091

# Para cada cliente/hospital
BASE_PORT=9094  # Porta base para supernode
for ((i=1; i<=N_CLIENTES; i++)); do
    SN_PORT=$((BASE_PORT + i - 1))  # Ex: 9094, 9095, ...
    DATASET_PATH="/dados_simulacao/fl_conj_treino_c${i}.csv"
    
    # Inicia o supernode do cliente
    docker run --rm \
        -p ${SN_PORT}:${SN_PORT} \
        --network flwr-network \
        --name supernode-${i} \
        --detach \
        flwr/supernode:1.18.0 \
        --insecure \
        --superlink superlink:9092 \
        --clientappio-api-address 0.0.0.0:${SN_PORT} \
        --isolation process

    # Inicia o clientapp do cliente
    docker run --rm \
        -v ${DATASET_PATH}:/mnt/fl_conj_treino_cliente.csv \
        --network flwr-network \
        --name client-${i} \
        --detach \
        flwr_clientapp:0.0.1 \
        --insecure \
        --clientappio-api-address supernode-${i}:${SN_PORT}
done

echo "Ambiente FLWR com $N_CLIENTES clientes iniciado."
