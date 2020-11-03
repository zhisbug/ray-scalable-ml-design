#!/bin/bash
python /users/hzhang2/projects/ray-scalable-ml-design/distributed-pytorch-stack/comm-lib-microbenchmarks/main.py /datasets/BigLearning/jinlianw/imagenet_raw/ \
	-a vgg19 \
	-b 64
