// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "eth_common.h"
#include "config_utils.h"

// RPC framework packet headers
struct p2p_rpc_hdr {
  uint8_t pkt_hdr_t[UDP_HEADER_LEN];
  uint64_t req_token;
  int64_t sock_perf_header;
  uint16_t seq_num;
  uint8_t _pad[64 - (UDP_HEADER_LEN + sizeof(uint64_t) + sizeof(int64_t) + sizeof(uint16_t))];
  //uint16_t func_id;
} __attribute__((packed));

static const size_t RPC_HEADER_LEN = sizeof(struct p2p_rpc_hdr);
static const size_t RPC_HEADER_TAIL_LEN = RPC_HEADER_LEN - UDP_HEADER_LEN;
static const size_t MAX_MTU = RPC_MTU + RPC_HEADER_LEN;

static inline struct eth_hdr* get_eth_header(struct p2p_rpc_hdr *hdr) {
  return reinterpret_cast<struct eth_hdr *>(hdr->pkt_hdr_t);
}

static inline struct ipv4_hdr* get_ip_header(struct p2p_rpc_hdr *hdr) {
  return reinterpret_cast<struct ipv4_hdr *> (hdr->pkt_hdr_t + sizeof(struct eth_hdr));
}

static inline struct udp_hdr* get_udp_header(struct p2p_rpc_hdr *hdr) {
  return reinterpret_cast<struct udp_hdr *> (hdr->pkt_hdr_t + sizeof(ipv4_hdr) + sizeof(struct eth_hdr));
}

static inline uintptr_t get_req_token(struct p2p_rpc_hdr *hdr) {
  return static_cast<uintptr_t>(hdr->req_token);
}

static inline uint16_t get_seq_num(struct p2p_rpc_hdr *hdr) {
  return static_cast<uint16_t>(hdr->seq_num);
}

static inline void* get_req_tail(struct p2p_rpc_hdr *hdr) {
  return (void*)((uint8_t*)hdr + UDP_HEADER_LEN); 
}

static inline void set_sockperf_header(struct p2p_rpc_hdr *hdr) {
  hdr->sock_perf_header &= 0xFFFFFFFF00000000; 
}

static inline void add_ip_udp_len(struct p2p_rpc_hdr *hdr, uint16_t pkt_len) {
  struct ipv4_hdr *ip_h = get_ip_header(hdr);
  ip_h->tot_len = htons(sizeof(ipv4_hdr) + sizeof(udp_hdr) + pkt_len); 
  struct udp_hdr *udp_h = get_udp_header(hdr);
  udp_h->len = htons(sizeof(udp_hdr) + pkt_len);
}

static inline void add_udp_cksum(struct p2p_rpc_hdr *hdr) {
  struct udp_hdr *udp_h = get_udp_header(hdr);
  udp_h->check = ipv4_pseudo_csum(get_ip_header(hdr));
}

// Checksum needs to be calculated after calling this
static inline void swap_eth_hdr(struct p2p_rpc_hdr *pkt_hdr) {
  struct eth_hdr *eth_h = get_eth_header(pkt_hdr);
  // Swap the MAC addr to do the echo
  struct mac_addr tmp_mac;
  tmp_mac = eth_h->src_mac_;
  eth_h->src_mac_ = eth_h->dst_mac_;
  eth_h->dst_mac_ = tmp_mac;

  // Swap IP and UDP
  struct ipv4_hdr *ip_h = get_ip_header(pkt_hdr);
  struct udp_hdr *udp_h = get_udp_header(pkt_hdr);
  uint32_t tmp_ip = ip_h->src_ip;
  ip_h->src_ip = ip_h->dst_ip;
  ip_h->dst_ip = tmp_ip;
  ip_h->check = 0;
  uint16_t tmp_port = udp_h->src_port;
  udp_h->src_port = udp_h->dst_port;
  udp_h->dst_port = tmp_port;
  udp_h->check = 0;
}
