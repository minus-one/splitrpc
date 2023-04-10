#! /bin/zsh

ifname=$1
ip neigh change 192.168.25.1 lladdr b8:ce:f6:cc:6a:52 dev ${ifname} 
ip neigh change 192.168.25.2 lladdr 0c:42:a1:10:41:18 dev ${ifname} 
ip neigh change 192.168.25.3 lladdr 02:43:67:f7:31:3b dev ${ifname}   
