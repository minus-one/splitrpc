// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "p2p_rpc.h"

// Pre-created rpc-hdr for a connection
struct p2p_rpc_conn_info {
  /*
     int nic_port, queue_id;     // App
     int src_port, dst_port;     // L4
     uint32_t s_ip, d_ip;        // L3
     uint8_t s_mac[6], r_mac[6]; // L2
  */
  struct p2p_rpc_hdr hdr_template;
};

static inline p2p_rpc_conn_info*
alloc_conn_info()
{
  struct p2p_rpc_conn_info *new_conn = new struct p2p_rpc_conn_info; 
  std::memset(new_conn, 0, sizeof(struct p2p_rpc_conn_info));
  return new_conn;
}

static inline void 
release_conn_info(struct p2p_rpc_conn_info* conn)
{
  delete conn;
}

static inline p2p_rpc_conn_info* 
init_conn_info(
    int src_port, int dst_port, 
    const char *src_ip_str, const char *dst_ip_str, 
    mac_addr src_mac, mac_addr dst_mac)
{
  struct p2p_rpc_conn_info* new_conn = alloc_conn_info();
  struct eth_hdr *eth_h = (struct eth_hdr *)(&(new_conn->hdr_template.pkt_hdr_t[0]));
  gen_eth_header_template(eth_h, &src_mac, &dst_mac);

  struct ipv4_hdr *ip_h = (struct ipv4_hdr *)((uint8_t *) eth_h + sizeof(struct eth_hdr));
  gen_ipv4_header_template(ip_h, ipv4_from_str(src_ip_str), ipv4_from_str(dst_ip_str));

  struct udp_hdr *udp_h = (struct udp_hdr *)((uint8_t *) ip_h + sizeof(struct ipv4_hdr));
  gen_udp_header_template(udp_h, src_port, dst_port);

  return new_conn;
}

static inline p2p_rpc_conn_info* 
initialize_src_conn_info(std::string src_uri)
{
  std::stringstream ss_tmp(src_uri);
  std::string src_mac_str, src_ip_str, src_port_str;

  std::getline(ss_tmp, src_mac_str, ',');
  std::getline(ss_tmp, src_ip_str, ',');
  std::getline(ss_tmp, src_port_str, ',');
  
  mac_addr src_mac = mac_from_string(src_mac_str);
  int server_port = std::stoi(src_port_str);

  //spdlog::info("SERVER: MAC: {}, IP: {}, Port: {}",
  //             mac_to_string(src_mac), src_ip_str, src_port);

  return init_conn_info(server_port, 0,
                     src_ip_str.c_str(), "0.0.0.0",
                     src_mac, src_mac);
}

// Parses the rpc_hdr of a received request, and updates the conn_info
static inline void
set_conn_info(struct p2p_rpc_conn_info *existing_conn, p2p_rpc_hdr *recv_hdr)
{
  // Copy the non-eth portions
  memcpy(get_req_tail(&existing_conn->hdr_template), get_req_tail(recv_hdr), RPC_HEADER_TAIL_LEN);
  // Set the dst port
  get_udp_header(&existing_conn->hdr_template)->dst_port = get_udp_header(recv_hdr)->src_port;
  // Set the dst IP
  get_ip_header(&existing_conn->hdr_template)->dst_ip = get_ip_header(recv_hdr)->src_ip;
  // Set the dst MAC
  get_eth_header(&existing_conn->hdr_template)->dst_mac_ = get_eth_header(recv_hdr)->src_mac_;
}
