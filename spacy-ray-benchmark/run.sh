#!/bin/bash

ROOT_DIR=$(dirname $(realpath -s $0))
MY_IPADDR=$(hostname -i)
source $ROOT_DIR/env/bin/activate
echo $MY_IPADDR

cd spacy-ray-benchmark/tmp/experiments/en-ent-wiki
ray stop --force
sleep 3
ray start --head --object-manager-port=8076 --object-store-memory=25359738368
sleep 2

for i in {1..15}
do
  echo "=> node $i"
  ssh -o StrictHostKeyChecking=no h$i.ray-dev-16.BigLearning.orca.pdl.cmu.edu "source $ROOT_DIR/env/bin/activate; cd $ROOT_DIR/spacy-ray-benchmark/tmp/experiments/en-ent-wiki; ray stop --force; sleep 3; ray start --address='$MY_IPADDR:6379' --object-manager-port=8076 --object-store-memory=25359738368; sleep 2"
done
wait

sleep 10

python3 -m spacy ray train configs/default.cfg --n-workers 16 --gpu-id 16 --paths.train corpus/train.spacy --paths.dev corpus/test.spacy --address=$MY_IPADDR:6379