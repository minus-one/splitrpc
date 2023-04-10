// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "p2p_rpc_tring.h"
#include "debug_utils.h"

#include "gpu_rdma_common.h"

enum APP_RR_STATUS {FREE, RX_COMPLETE, WORK_COMPLETE, TX_COMPLETE };

class p2p_rpc_bf_wi 
{
  public:
  uint64_t token; // Pointer to itself - used for wr_id
  size_t idx;     // Index in the RrPool


  uint8_t *local_req_stub, *local_resp_stub;
  
  struct ibv_sge *write_sge, *read_sge;
  struct ibv_send_wr *write_wr, *read_wr;

  // Client side details
  sockaddr_in *si_client;
  void *rpc_rr;

  // States
  uint32_t wi_state;

  void print_wi()
  {
    TRACE_PRINTF("wi: %p, idx: %ld, write_addr: %p, read_addr: %p, notif_addr: %p\n",
        (void*)this, idx, (void*)write_wr[0].wr.rdma.remote_addr, (void*)read_wr->wr.rdma.remote_addr, (void*)write_wr[1].wr.rdma.remote_addr);
  }
};

class P2pRpcBfRrPool {
  private:
    // Used to collect the info about the memory
    //buf_mon_rr buf_mon_msg; 
    struct r_context *rctx;

    P2pRpcTring *req_pool, *resp_pool;

    int pi_idx;
  public:
    // FIXME: This shouldn't be public. Ideally we should just use the get_next() APIs
    int pool_size;

    size_t req_size, resp_size;
    struct ibv_mr *req_mr, *resp_mr;

    struct ibv_mr *notify_mr;
    //struct ibv_mr *notify_mr;
    p2p_rpc_bf_wi *bf_wi_pool;

    inline p2p_rpc_bf_wi* get_bf_wi(int idx) {
      if(idx >= 0 && idx < pool_size)
        return &bf_wi_pool[idx];
      return NULL;
    }

    inline p2p_rpc_bf_wi* get_next() {
      p2p_rpc_bf_wi *next_bf_wi = &bf_wi_pool[pi_idx];
      next_bf_wi->write_wr[0].wr.rdma.remote_addr = (uint64_t)req_pool->get_next(req_size); 
      next_bf_wi->read_wr->wr.rdma.remote_addr = (uint64_t)resp_pool->get_next(resp_size); 
      pi_idx = (pi_idx + 1) % pool_size;
      return next_bf_wi; 
    }

    P2pRpcBfRrPool(struct r_context *_rctx, int _pool_size, buf_mon_rr _buf_mon_details)
    {
      rctx = _rctx;
      pool_size = _pool_size;
      buf_mon_rr buf_mon_msg = _buf_mon_details;
      req_size = buf_mon_msg.req_size;
      resp_size = buf_mon_msg.resp_size;
      req_pool = new P2pRpcTring((void*)buf_mon_msg.req_buf_attr.address, buf_mon_msg.req_buf_attr.length);
      resp_pool = new P2pRpcTring((void*)buf_mon_msg.resp_buf_attr.address, buf_mon_msg.resp_buf_attr.length);

      req_mr = rdma_buffer_alloc(rctx->pd, pool_size * req_size, IBV_ACCESS_LOCAL_WRITE);
      resp_mr = rdma_buffer_alloc(rctx->pd, pool_size * resp_size, IBV_ACCESS_LOCAL_WRITE);
      memset(req_mr->addr, 0, pool_size * buf_mon_msg.req_size);
      memset(resp_mr->addr, 0, pool_size * buf_mon_msg.resp_size);

      notify_mr = rdma_buffer_alloc(rctx->pd, sizeof(uint32_t), IBV_ACCESS_LOCAL_WRITE);
      *(uint32_t*)(notify_mr->addr) = APP_RR_STATUS::RX_COMPLETE;

      bf_wi_pool = new p2p_rpc_bf_wi[pool_size];

      TRACE_PRINTF("==================================================================================\n");
      TRACE_PRINTF("Setting up P2pRpcBfRrPool: %d items...\n", pool_size);
      for(int i = 0 ; i < pool_size; i++) {
        p2p_rpc_bf_wi *new_bf_wi = &bf_wi_pool[i];
        new_bf_wi->token = (uint64_t)(void*)(new_bf_wi);
        new_bf_wi->idx = i;
        new_bf_wi->si_client = new struct sockaddr_in;
        new_bf_wi->wi_state = APP_RR_STATUS::FREE; 

        new_bf_wi->write_sge = new struct ibv_sge[2];
        new_bf_wi->write_wr = new struct ibv_send_wr[2];
        new_bf_wi->read_sge = new struct ibv_sge;
        new_bf_wi->read_wr = new struct ibv_send_wr;

        // Setup the WRITE work-items
        struct ibv_sge *sge = new_bf_wi->write_sge;
        new_bf_wi->local_req_stub = (i * req_size) + (uint8_t*)req_mr->addr;
        sge[0].addr = (uint64_t)(new_bf_wi->local_req_stub);
        //sge[0].addr = ((uint64_t) req_mr->addr);
        sge[0].length = req_size;
        sge[0].lkey = req_mr->lkey;

        bzero(&new_bf_wi->write_wr[0], sizeof(struct ibv_send_wr));
        new_bf_wi->write_wr[0].wr_id = new_bf_wi->token;
        new_bf_wi->write_wr[0].sg_list = &sge[0];
        new_bf_wi->write_wr[0].num_sge = 1;
        new_bf_wi->write_wr[0].opcode = IBV_WR_RDMA_WRITE;
        new_bf_wi->write_wr[0].send_flags = IBV_SEND_SIGNALED;
        new_bf_wi->write_wr[0].wr.rdma.rkey = buf_mon_msg.req_buf_attr.stag.remote_stag;
        new_bf_wi->write_wr[0].next = &new_bf_wi->write_wr[1];

        // Setup the NOTIFY work-items
        uint64_t state_offset = (uint64_t)(i * sizeof(uint32_t));
        sge[1].addr = ((uint64_t) notify_mr->addr);
        sge[1].length = sizeof(uint32_t); 
        sge[1].lkey = notify_mr->lkey;
        bzero(&new_bf_wi->write_wr[1], sizeof(struct ibv_send_wr));
        new_bf_wi->write_wr[1].wr_id = new_bf_wi->token;
        new_bf_wi->write_wr[1].sg_list = &sge[1];
        new_bf_wi->write_wr[1].num_sge = 1;
        new_bf_wi->write_wr[1].opcode = IBV_WR_RDMA_WRITE;
        new_bf_wi->write_wr[1].send_flags = IBV_SEND_SIGNALED;
        new_bf_wi->write_wr[1].wr.rdma.rkey = buf_mon_msg.state_buf_attr.stag.remote_stag;
        new_bf_wi->write_wr[1].wr.rdma.remote_addr = buf_mon_msg.state_buf_attr.address + state_offset; 

        // Setup the READ work-items
        sge = new_bf_wi->read_sge;
        new_bf_wi->local_resp_stub =  (i * resp_size) + (uint8_t*)resp_mr->addr;
        sge->addr = (uint64_t)(new_bf_wi->local_resp_stub);
        //sge->addr = ((uint64_t) resp_mr->addr);
        sge->length = resp_size;
        sge->lkey = resp_mr->lkey;
        
        bzero(new_bf_wi->read_wr, sizeof(struct ibv_send_wr));
        new_bf_wi->read_wr->wr_id = new_bf_wi->token;
        new_bf_wi->read_wr->sg_list = sge;
        new_bf_wi->read_wr->num_sge = 1;
        new_bf_wi->read_wr->opcode = IBV_WR_RDMA_READ;
        new_bf_wi->read_wr->send_flags = IBV_SEND_SIGNALED;
        new_bf_wi->read_wr->wr.rdma.rkey = buf_mon_msg.resp_buf_attr.stag.remote_stag;

        // MUTABLES that will be set by calling P2pRpcTring
        new_bf_wi->write_wr[0].wr.rdma.remote_addr = 0; 
        new_bf_wi->read_wr->wr.rdma.remote_addr = 0; 

        TRACE_PRINTF("bf_rr_idx: %d, l_req_stub: %p, l_resp_stub: %p, r_state: %p, r_req_stub: %p, r_resp_stub: %p\n",
            i, (void*)new_bf_wi->local_req_stub, (void*)new_bf_wi->local_resp_stub, (void*) new_bf_wi->write_wr[1].wr.rdma.remote_addr,
            (void*)new_bf_wi->write_wr[0].wr.rdma.remote_addr, (void*)new_bf_wi->read_wr->wr.rdma.remote_addr); 
      }
      pi_idx = 0;
    }

    ~P2pRpcBfRrPool() 
    {
      for(int i = 0 ; i < pool_size; i++) {
        delete bf_wi_pool[i].si_client;
        delete bf_wi_pool[i].write_sge;
        delete bf_wi_pool[i].read_sge;
        delete bf_wi_pool[i].read_wr;
        delete bf_wi_pool[i].write_wr;
      }
      delete bf_wi_pool;
      delete req_pool;
      delete resp_pool;
      rdma_buffer_free(req_mr);
      rdma_buffer_free(resp_mr);   
      rdma_buffer_free(notify_mr);
    }
};
