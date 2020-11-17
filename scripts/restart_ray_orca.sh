#!/bin/bash

ROOT_DIR=$(dirname $(realpath -s $0))
MY_IPADDR=$(hostname -i)
source $ROOT_DIR/env/bin/activate
echo $MY_IPADDR

ray stop
sleep 3
ray start --head --object-manager-port=8076 --resources='{"machine":1}' --object-store-memory=32359738368
sleep 2

for i in {1..2}
do
  echo "=> node $i"
  ssh -o StrictHostKeyChecking=no h$i.ray-dev3.BigLearning.orca.pdl.cmu.edu "source $ROOT_DIR/env/bin/activate; ray stop; ray start --address='$MY_IPADDR:6379' --object-manager-port=8076 --resources='{\"machine\":1}' --object-store-memory=32359738368";
done
wait

#sleep 5
#for node in ${OTHERS_IPADDR[@]}; do
# echo "=> $node"
# ssh -o StrictHostKeyChecking=no $node PATH=$PATH:/home/ubuntu/anaconda3/bin:/home/ubuntu/anaconda3/condabin, ray start --redis-address=$MY_IPADDR:6379 --object-manager-port=8076 --resources=\'{\"node\":1}\' --object-store-memory=34359738368 &
#done
#wait
