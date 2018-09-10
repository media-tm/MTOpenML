# nvidia-gpu驱动

## 1. Nvidia-CUDA®驱动

### 1.1 CUDA®概述
CUDA®是由NVIDIA开发的用于图形处理单元（GPU）上的通用计算的并行计算平台和编程模型。 借助CUDA，开发人员可以通过利用GPU的强大功能大大加速计算应用程序。在GPU加速的应用程序中，工作负载的连续部分在CPU上运行 - 它针对单线程性能进行了优化 - 而应用程序的计算密集型部分并行运行在数千个GPU核心上。 使用CUDA时，开发人员使用流行语言（如C，C ++，Fortran，Python和MATLAB）编程，并通过几个基本关键字形式的扩展来表达并行性。NVIDIA的CUDA工具包提供了开发GPU加速应用程序所需的一切。 CUDA工具包包括GPU加速库，编译器，开发工具和CUDA运行时。

请访问以下网址了解CUDA®概述、下载、安装和入门。
- [CUDA® 工具包 9.0](https://developer.nvidia.com/cuda-downloads)
- [CUDA® DOC](https://docs.nvidia.com/cuda/)
- [CUDA®  Quick Start Guide](https://docs.nvidia.com/cuda/archive/9.0/cuda-quick-start-guide/index.html)
- [CUDA® Doc 9.0](https://docs.nvidia.com/cuda/archive/9.0/)
- [cuDNN® SDK](https://docs.nvidia.com/deeplearning/sdk/index.html)
- [cuDNN® Install](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

| Meta Package | Purpose |
| ------ | ------ |
| cuda | Installs all CUDA Toolkit and Driver packages. Handles upgrading to the next version of the cuda package when it's released. |
| cuda-9-0 | Installs all CUDA Toolkit and Driver packages. Remains at version 9.0 until an additional version of CUDA is installed. |
| cuda-toolkit-9-0 | Installs all CUDA Toolkit packages required to develop CUDA applications. Does not include the driver. |
| cuda-runtime-9-0 | Installs all CUDA Toolkit packages required to run CUDA applications, as well as the Driver packages. |
| cuda-libraries-9-0 | Installs all runtime CUDA Library packages. |
| cuda-libraries-dev-9-0 | Installs all development CUDA Library packages. |
| cuda-drivers | Installs all Driver packages. Handles upgrading to the next version of the Driver packages when they're released. |


### 1.2 CUDA®安装注意事项

**Installing NVIDIA Graphics Drivers**
- NVIDIA graphics driver R375 or newer for CUDA 8
- NVIDIA graphics driver R384 or newer for CUDA 9
- NVIDIA graphics driver R390 or newer for CUDA 9.2

特别说明： 官方文档非常完善包含了所有细节，如果你安装失败，安装文档就是远方的灯塔。先看看深度学习框架支持那些版本，再安装以提高效率。

CUDA和cuDNN的版本选择  
mxnet-2018     : Download cuDNN v7.1.4 (May 16, 2018), for CUDA 9.0  
tensorflow-2018: Download cuDNN v7.1.4 (May 16, 2018), for CUDA 9.0


### 1.3 ubuntu安装CUDA® 9.0 + cuDNN 7.1
**CUDA®前提条件** 
- CUDA-capable GPU
- A supported version of Linux with a gcc compiler and toolchain
- NVIDIA CUDA Toolkit (available at http://developer.nvidia.com/cuda-downloads)

**CUDA®安装步骤**  
- 使用包安装，参见[官方CUDA®安装指导文档#package-manager-installation](https://docs.nvidia.com/cuda/archive/9.0/cuda-installation-guide-linux/index.html#package-manager-installation)  
```sh
$ sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
$ sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
$ /var/lib/apt/lists/*cuda*Packages | grep "Package:"
$ sudo apt-get update
$ sudo apt-get install cuda-9-0
$ sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_9.0.xx_amd64.deb
$ sudo apt-get update   # update patch
$ sudo apt-get upgrade cuda-9-0

```
- 使用runfile安装，需要禁用Nouveau驱动，参见[官方CUDA®安装指导文档#runfile](https://docs.nvidia.com/cuda/archive/9.0/cuda-installation-guide-linux/index.html#runfile)


**CUDA®升级**
```sh
$ cat /var/lib/apt/lists/*cuda*Packages | grep "Package:"      # available CUDA® packages
$ sudo apt-get install cuda           # Ubuntu
$ sudo apt-get install cuda-drivers   # Ubuntu
```

**CUDA®环境设置**
参见[官方CUDA®安装指导文档#post-installation-actions](https://docs.nvidia.com/cuda/archive/9.0/cuda-installation-guide-linux/index.html#post-installation-actions)
```sh
$ sudo gedit ~/.bashrc
$ export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
$ export CUDA_HOME=/usr/local/cuda-9.0 
$ export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
$ export PATH=${CUDA_HOME}/bin:${PATH}
$ sudo ldconfig
$ watch -n 1 nvidia-smi
$ cuda-install-samples-9.0.sh <dir>
$ cat /proc/driver/nvidia/version
```

**cuDNN® 安装**
windows环境安装参见，[官方cuDNN®安装指导文档#install-windows](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows)  
Linux环境安装参见，[官方cuDNN®安装指导文档#install-linux](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux)

ubuntu使用deb时的安装命令
```sh
# 安装cuDNN®运行时库
$ sudo dpkg -i libcudnn7_7.0.3.11-1+cuda9.0_amd64.deb
# 安装cuDNN®开发库
$ sudo dpkg -i libcudnn7-dev_7.0.3.11-1+cuda9.0_amd64.deb
# 安装cuDNN®示例代码
$sudo dpkg -i libcudnn7-doc_7.0.3.11-1+cuda9.0_amd64.deb
```

**cuDNN® 验证**
```sh
$ $cp -r /usr/src/cudnn_samples_v7/ $HOME
$ cd  $HOME/cudnn_samples_v7/mnistCUDNN
$ make clean && make
$ ./mnistCUDNN
```

## 2. 删除 Nvidia-CUDA® 驱动
```sh
$ sudo apt-get --purge remove cuda
$ sudo apt autoremove
$ sudo apt-get clean
```

## 参考文献  
[1 Ubuntu 16.04安装NVIDIA驱动](https://blog.csdn.net/cosmoshua/article/details/76644029)
[2 Ubuntu 16.04 + CUDA 8.0 + cuDNN v5.1 + TensorFlow(GPU support)安装配置详解](http://www.cnblogs.com/wangduo/p/7383989.html)
