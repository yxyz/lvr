#当下载CUDA工具包 + 580版本计算库（兼容550驱动，且是列表中较新稳定版）速度慢时，可以直接使用下面的命令安装
sudo dpkg -i nvidia-cuda-toolkit_11.5.1-1ubuntu1_amd64.deb libnvidia-compute-580_580.95.05-0ubuntu0.22.04.1_amd64.deb
#如果安装时提示 “依赖未满足”，执行下面的命令自动补全依赖：
sudo apt-get -f install