// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <algorithm>
#include <iostream>

static int initUdpSock(struct sockaddr_in *si_me, bool timeout=true) 
{
  int s;
  //create a UDP socket
  if ((s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1) {
    std::cerr << "socket failed\n";
    return -1;
  }
  
  //bind socket to port
  if (bind(s, (struct sockaddr *)si_me, sizeof(*si_me)) == -1) {
    perror("Bind failed. Error: ");
    return -1;
  }

  int buffsize = 2*1024*1024;
  if(setsockopt(s, SOL_SOCKET, SO_RCVBUF, &buffsize, sizeof(buffsize)) < 0) {
    perror("Error in setting sock opt SO_RCVBUF");
    return -1;
  }
  if(setsockopt(s, SOL_SOCKET, SO_SNDBUF, &buffsize, sizeof(buffsize)) < 0) {
    perror("Error in setting sock opt SO_SNDBUF");
    return -1;
  }
  if(timeout) {
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 10; //10us
    if (setsockopt(s, SOL_SOCKET, SO_RCVTIMEO,&tv,sizeof(tv)) < 0) {
      perror("Error in setting sock opt SO_RCVTIMEO");
      return -1;
    }
  }
  return s;
}

static 
int getData(int udp_sock, struct sockaddr_in *si_other, void* data, size_t len_in_bytes)
{  
  socklen_t slen = sizeof(*si_other); 
  //size_t recv_len = 0;

  return recvfrom(udp_sock, data, size_t(len_in_bytes), 0, (struct sockaddr *)si_other, &slen);

  //int tries = 1;
  //int64_t seg_len = 0;

  //while(recv_len < len_in_bytes) {
  //  seg_len = 0;
  //  if ((seg_len = recvfrom(udp_sock, data, size_t(len_in_bytes - recv_len), 0, (struct sockaddr *)si_other, &slen)) == -1) {
  //    tries--;
  //  } else { 
  //    TRACE_PRINTF("recvfrom returns with seg_len: %ld\n", seg_len);
  //    recv_len += seg_len;
  //    data = (void*)((char*)data + seg_len);
  //  }
  //  if(tries == 0)
  //    break;
  //}
  ////if(recv_len != len_in_bytes) {
  ////  std::cout<< "Warning! Recevied " << recv_len << " bytes instead of " << len_in_bytes << "\n";
  ////}
  //return recv_len;
}

static void sendData(int udp_sock, struct sockaddr_in *si_other, void *outputData, size_t len, const size_t seg_size)
{
  //si_other->sin_port = htons(port);
  socklen_t slen = sizeof(*si_other); 
  //now reply the client with the same data
  while(len > 0) {
    size_t curr_seg_size = std::min(seg_size, len);
    if (sendto(udp_sock, outputData, curr_seg_size, MSG_CONFIRM, (struct sockaddr *)si_other, slen) == -1) {
      perror("sendto failed: ");
    }
    len -= curr_seg_size;
    outputData = (void*)((char*)outputData + seg_size);
  }
}
