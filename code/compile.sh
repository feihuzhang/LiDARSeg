#export PYTHONPATH=/home/feihu/Desktop/MinknowskiEngine/examples/pointnet/libs:$PYTHONPATH
if true; then
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/feihu/anaconda1912/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/feihu/anaconda1912/etc/profile.d/conda.sh" ]; then
        . "/home/feihu/anaconda1912/etc/profile.d/conda.sh"
    else
        export PATH="/home/feihu/anaconda1912/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export LD_LIBRARY_PATH="/home/feihu/anaconda1912/lib:/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"
export LD_INCLUDE_PATH="/home/feihu/anaconda1912/include:/usr/local/cuda-10.0/include:/home/feihu/anaconda1912/lib/python3.7/site-packages/torch/lib/include:$LD_INCLUDE_PATH"
export PYTHONPATH=/media/feihu/Storage/kitti_point_cloud/SparseConv/build/lib:${PYTHONPATH}
fi
cd libs
rm -rf build
python setup.py build
cp -r build/lib* build/lib

