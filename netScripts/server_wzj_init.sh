apt-get update
echo 'installing vim......'
apt-get install vim
cp /home/wzz/env_settings/vimrc.bk ~/.vimrc
echo 'installing tkinter......'
apt-get install tcl-dev tk-dev python3-tk
echo 'installing screen......'
apt-get install screen
echo 'installing bypy......'
pip install bypy
apt-get install aria2
echo 'Set the locale......'
apt-get clean && apt-get install -y locales
locale-gen en_US.UTF-8
cp /home/wzz/env_settings/bashrc.bk ~/.bashrc
cp /home/wzz/env_settings/profile.bk ~/.profile
source ~/.bashrc

