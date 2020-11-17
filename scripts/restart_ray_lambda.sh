#!/bin/bash
ROOT_DIR=$(dirname $(dirname $(realpath -s $0)))
echo $ROOT_DIR
MY_IPADDR=10.20.41.115
source $ROOT_DIR/env/bin/activate
echo $MY_IPADDR

ray stop
sleep 3
NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=enp179s0f0 ray start --head --object-manager-port=8076 --resources='{"machine":1}' --object-store-memory=32359738368
sleep 2

echo "=> node $i"
ssh -o StrictHostKeyChecking=no -i /home/hao.zhang/.ssh/arion.pem hao.zhang@10.20.41.120 "source $ROOT_DIR/env/bin/activate; ray stop; NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=enp179s0f0 ray start --address='$MY_IPADDR:6379' --object-manager-port=8076 --resources='{\"machine\":1}' --object-store-memory=32359738368";
wait

#sleep 5
#for node in ${OTHERS_IPADDR[@]}; do
# echo "=> $node"
# ssh -o StrictHostKeyChecking=no $node PATH=$PATH:/home/ubuntu/anaconda3/bin:/home/ubuntu/anaconda3/condabin, ray start --redis-address=$MY_IPADDR:6379 --object-manager-port=8076 --resources=\'{\"node\":1}\' --object-store-memory=34359738368 &
#done
#wait
