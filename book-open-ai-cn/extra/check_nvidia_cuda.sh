echo -e "\r\n pre-installation checking"
echo "\r\n check step[1/4]: Verify You Have a CUDA-Capable GPU"
lspci | grep -i nvidia

echo -e "\r\n check step[2/4]: Verify You Have a Supported Version of Linux, 32bit or 64bit"
uname -m && cat /etc/*release

echo -e "\r\n check step[3/4]: Verify the System Has gcc Installed"
gcc --version

echo -e "\r\n check step[4/4]: Verify the System has the Correct Kernel Headers and Development Packages Installed"
uname -r
sudo apt-get install linux-headers-$(uname -r)
