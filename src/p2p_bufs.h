// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <stdlib.h>

// Has to be < 2^16
const uint16_t MAX_BI_SIZE = 1024;
const uint16_t MAX_WI_SIZE = 1024;

// These are fixed sized buf-ptrs containing p2p-rpc 
// headers or buf-ptrs for dpdk mbufs
struct p2p_hbufs {
  uintptr_t burst_items[MAX_BI_SIZE];
  uint16_t num_items;
}__attribute__((packed));

/*
 * p2p_bufs is an array of <ptr-to-buf, size-of-buf>.
 */
struct p2p_bufs {
  uintptr_t burst_items[MAX_BI_SIZE];
  size_t item_size[MAX_BI_SIZE];
  uint16_t num_items;
}__attribute__((packed));

// sk_buf contains a list of mbufs and/or app-bufs ptrs
// If it is zero-copy, there will be no o_buf
// On RX-path: i_buf -> mbuf, o_buf -> app-buf
// On TX-path: i_buf -> app_buf, o_buf -> mbuf
class p2p_sk_buf {
  public:
    uintptr_t i_buf[MAX_BI_SIZE];
    uintptr_t o_buf[MAX_BI_SIZE];
    size_t len[MAX_BI_SIZE - 1];
    // FIXME: Keeping it like this for appropriate alignment purposes
    size_t num_items;
}__attribute__((packed));

static void
print_bufs(struct p2p_bufs *bufs)
{
  printf("==========================================================================\n");
  printf("BUF: %p, num_items: %d\n", (void*)bufs, bufs->num_items);
  for(size_t i = 0 ; i < bufs->num_items ; i++) {
    printf("i: %ld, bi_ptr: %p, bi_size: %ld\n", i, (void*)bufs->burst_items[i], bufs->item_size[i]);
  }
  printf("==========================================================================\n");
}

static void
print_skb(p2p_sk_buf *skb)
{
  printf("==========================================================================\n");
  printf("SKB: %p, num_items: %ld\n", (void*)skb, skb->num_items);
  for(size_t i = 0 ; i < skb->num_items ; i++) {
    printf("i: %ld, i_buf: %p, o_buf: %p, len: %ld\n", i, (void*)skb->i_buf[i], (void*)skb->o_buf[i], skb->len[i]);
  }
  printf("==========================================================================\n");
}
