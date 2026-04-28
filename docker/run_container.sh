proj_dir=$(cd $(dirname $0);cd ..;pwd)
echo proj_dir $proj_dir

docker run --rm -it --gpus all --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --name mprg_hybrid_pruning \
    -v $proj_dir/files:/workspace/files \
    mprg/hybrid_pruning:20260216 \
    bash
