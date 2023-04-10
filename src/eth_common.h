// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <string>
#include <sstream>
#include <cstring>

struct mac_addr {
  uint8_t addr[6];
}__attribute__((packed));

struct eth_hdr {
  struct mac_addr dst_mac_;
  struct mac_addr src_mac_;
  uint16_t eth_type_;
} __attribute__((packed));

// In network-byte-order
struct ipv4_hdr {
  uint8_t ihl : 4;
  uint8_t version : 4;
  uint8_t ecn : 2;
  uint8_t dscp : 6;
  uint16_t tot_len;
  uint16_t id;
  uint16_t frag_off;
  uint8_t ttl;
  uint8_t protocol;
  uint16_t check;
  uint32_t src_ip;
  uint32_t dst_ip;
} __attribute__((packed));

struct udp_hdr {
  uint16_t src_port;
  uint16_t dst_port;
  uint16_t len;
  uint16_t check;
} __attribute__((packed));

static constexpr int UDP_HEADER_LEN = sizeof(eth_hdr) + sizeof(ipv4_hdr) + sizeof(udp_hdr);
static constexpr int ethHdrBits = UDP_HEADER_LEN * 8;
static constexpr uint16_t ETHER_TYPE_IPV4 = 0x0800;

static inline void 
gen_eth_header_template(eth_hdr *eth_header,
  const mac_addr *src_mac, const mac_addr *dst_mac)
{
  eth_header->src_mac_ = *src_mac;
  eth_header->dst_mac_ = *dst_mac;
  eth_header->eth_type_ = ntohs(ETHER_TYPE_IPV4);
}

/// Format the IPv4 header for a UDP packet. All value arguments are in
/// host-byte order. 
static inline void 
gen_ipv4_header_template(ipv4_hdr *ipv4_hdr_, 
    uint32_t src_ip, uint32_t dst_ip) 
{
  ipv4_hdr_->version = 4;
  ipv4_hdr_->ihl = 5;
  ipv4_hdr_->ecn = 1;  // ECT => ECN-capable transport
  ipv4_hdr_->dscp = 0;
  ipv4_hdr_->tot_len = 
    htons(sizeof(ipv4_hdr) + sizeof(udp_hdr)); // Add payload size to this, in request processing
  ipv4_hdr_->id = htons(0);
  ipv4_hdr_->frag_off = htons(0);
  ipv4_hdr_->ttl = 128;
  ipv4_hdr_->protocol = IPPROTO_UDP;
  ipv4_hdr_->src_ip = htonl(src_ip);
  ipv4_hdr_->dst_ip = htonl(dst_ip);
  ipv4_hdr_->check = 0;
}

static inline void
gen_udp_header_template(udp_hdr *udp_h, int src_port, int dst_port)
{
  udp_h->len = htons(sizeof(ipv4_hdr) + sizeof(udp_hdr));
  udp_h->src_port = htons(src_port);
  udp_h->dst_port = htons(dst_port);
  udp_h->check = 0;
}

/*
 * Compute checksum of IPv4 pseudo-header.
 */
static inline uint16_t ipv4_pseudo_csum(struct ipv4_hdr *ip)
{
  /* Header used for pseudo-checksum calculation
  struct psd_hdr
  {
    uint32_t src_addr;
    uint32_t dst_addr;
    uint8_t zero;
    uint8_t proto;
    uint16_t len;
  } __attribute__((packed));
  */

  //uint16_t psd_hdr_compressed[8];
  uint32_t psd_hdr_compressed[4];
  *(uint32_t*)(psd_hdr_compressed + 0) = ip->src_ip;
  *(uint32_t*)(psd_hdr_compressed + 1) = ip->dst_ip;
  *(uint8_t*)(psd_hdr_compressed + 2) = 0;
  *((uint8_t*)psd_hdr_compressed + 9) = ip->protocol;
  *(uint16_t*)((uint8_t*)psd_hdr_compressed + 10) = htons(
      ntohs(ip->tot_len) - (int)(ip->version * 4u));

  uint32_t sum = 0;
  uint16_t size = 8 * sizeof(uint16_t);
  uint16_t *p = (uint16_t*)psd_hdr_compressed;
  while (size > 1) {
    sum += *p;
    size -= sizeof(uint16_t);
    p++;
  }
  if (size) {
    sum += *((const uint8_t *) p);
  }

  // Fold 32-bit @x one's compliment sum into 16-bit value.
  sum = (sum & 0x0000FFFF) + (sum >> 16);
  sum = (sum & 0x0000FFFF) + (sum >> 16);
  return (uint16_t) sum;
}

/// Get the host-byte-order IPv4 address from a human-readable IP string
static inline uint32_t ipv4_from_str(const char* ip) {
  uint32_t addr = 0;
  int ret = inet_pton(AF_INET, ip, &addr);  // addr is in network-byte order
  if(ret != 1)
    printf("inet_pton() failed for %s\n", ip);
  return ntohl(addr);
}

/// Convert a network-byte-order IPv4 address to a human-readable IP string
static inline std::string ipv4_to_string(uint32_t ipv4_addr) {
  char str[INET_ADDRSTRLEN];
  const char* ret = inet_ntop(AF_INET, &ipv4_addr, str, sizeof(str));
  if(ret != str)
    printf("inet_ntop failed");
  str[INET_ADDRSTRLEN - 1] = 0;  // Null-terminate
  return str;
}

static inline mac_addr mac_from_string(const std::string mac_str) {
  std::stringstream ss_tmp(mac_str);
  mac_addr ret;
  for(int i = 0 ; i < 6; i++) {
    std::string tok;
    std::getline(ss_tmp, tok, ':');
    ret.addr[i] = static_cast<uint8_t>(std::stoi(tok, nullptr, 16));
  }
  return ret;
}

static inline std::string mac_to_string(mac_addr m) {
  char m_str[20];
  std::snprintf(m_str, 20, "%02hhx:%02hhx:%02hhx:%02hhx:%02hhx:%02hhx",
      m.addr[0], m.addr[1],
      m.addr[2], m.addr[3],
      m.addr[4], m.addr[5]);
  return std::string(m_str);
}
