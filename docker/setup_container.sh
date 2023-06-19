API_HOST=(`env | grep IPUOF_VIPU_API_HOST | awk -F '=' '{print $2}'`)
API_PARTITION=(`env | grep IPUOF_VIPU_API_PARTITION_ID | awk -F '=' '{print $2}'`)
echo 'API HOST: ' $API_HOST
echo 'API PARTITION:' $API_PARTITION

docker run \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --net=host \
    --cap-add=IPC_LOCK \
    --device=/dev/infiniband/ \
    -e IPUOF_VIPU_API_HOST=$API_HOST \
    -e IPUOF_VIPU_API_PARTITION_ID=$API_PARTITION \
    --ipc=host \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    --privileged \
    --ulimit stack=67108864 \
    -ti \
    -v `pwd`:/workspace \
    graphcore/pytorch:3.2.0 \
    bash -c \
    'DEBIAN_FRONTEND=noninteractive apt-get update; 
    DEBIAN_FRONTEND=noninteractive apt-get install -y libboost-all-dev;
    cd /workspace; make; pip install -r requirements.txt; /bin/bash;'
