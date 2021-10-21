export PYTHONPATH=/home/feihu/Desktop/SparseConvNet/examples/pointnet/libs:$PYTHONPATH
if true; then
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/feihu/anaconda1910/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/feihu/anaconda1910/etc/profile.d/conda.sh" ]; then
        . "/home/feihu/anaconda1910/etc/profile.d/conda.sh"
    else
        export PATH="/home/feihu/anaconda1910/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export LD_LIBRARY_PATH="/home/feihu/anaconda1910/lib:/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"
export LD_INCLUDE_PATH="/home/feihu/anaconda1910/include:/usr/local/cuda-10.0/include:/home/feihu/anaconda1910/lib/python3.7/site-packages/torch/lib/include:$LD_INCLUDE_PATH"
export PYTHONPATH=/home/feihu/Desktop/SparseConvNet/build/lib:${PYTHONPATH}
fi
CUDA_VISIBLE_DEVICES=0,1 python train.py --lr_decay=0.04 --learning_rate=0.1 --optimizer='SGD' --model pointnet2_part_seg_msg --normal --log_dir sparse2

