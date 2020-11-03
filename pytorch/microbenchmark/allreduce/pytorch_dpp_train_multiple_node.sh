#!/bin/bash
python /users/hzhang2/projects/ray-scalable-ml-design/distributed-pytorch-stack/comm-lib-microbenchmarks/main.py /datasets/BigLearning/jinlianw/imagenet_raw/ \
	--epochs 1 \
	-a vgg19 \
	-b 64 \
	--world-size 3 \
	--dist-url="file:///users/hzhang2/projects/ray-scalable-ml-design/distributed-pytorch-stack/comm-lib-microbenchmarks/sharedfile" \
	--rank $1 \
