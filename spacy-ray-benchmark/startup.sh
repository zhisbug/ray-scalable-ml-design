ROOT_DIR=$(dirname $(realpath -s $0))


virtualenv --python=python3.6 env
wait

source $ROOT_DIR/env/bin/activate

pip3 install -U spacy-nightly --pre
pip3 install -e spacy-ray-benchmark
pip3 install -U cupy-cuda110
