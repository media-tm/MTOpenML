## 1. 高频用法
### 1.1 列出已安装的包
$ pip freeze or pip list

### 1.2 卸载包
$ pip uninstall <包名> 
$ pip uninstall -r requirements.txt

### 1.3 升级包
$ pip install -U <包名>
$ pip install <包名> --upgrade

### 1.4 显示包所在的目录
pip show -f <包名>

### 1.5 搜索包
pip search <搜索关键字>

### 1.6 查询可升级的包
pip list -o

## 2. 低频用法
## 2.1 指定单次安装源
阿里镜像源：https://mirrors.aliyun.com/pypi/simple
科大镜像源：http://pypi.mirrors.ustc.edu.cn/simple/
$ pip install <包名> -i https://mirrors.aliyun.com/pypi/simple

## 2.2 安装pip
$ sudo easy_install pip  #安装
$ pip install -U pip     #升级

## 2.3 下载程序包(不安装)
$ pip install <包名> -d <目录>

## 2.4 wheel打包
pip wheel <包名>