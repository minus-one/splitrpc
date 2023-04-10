sudo modprobe nvidia-peermem
sudo /data/azk68/gdrcopy/insmod.sh
sudo ifconfig enp219s0f0 192.168.25.1/8 mtu 9000 up
sudo ip neigh add 192.168.25.2 lladdr 0c:42:a1:10:41:18 dev enp219s0f0
