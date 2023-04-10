// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <algorithm>
#include <signal.h>


#include "time_utils.h"
#include "stats_utils.h"
#include "config_utils.h"

#include "g_utils.cuh"

#include "ort_app.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


int
main() {
////////////////////////// Application initializations

  OrtFacade *AppServer;
  std::string model_path = getDatasetBasePath() + std::string("/models/") + get_ort_model_name();
  TRACE_PRINTF("OrtAppInit: Model: %s\n", model_path.c_str());

  cudaStream_t work_stream;
  cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking);

  AppServer = new OrtFacade(get_cuda_device_id(), work_stream);
  AppServer->loadModel(model_path);
  AppServer->printModelInfo();

  std::vector<void*>d_data;
  AppServer->setup_io_binding(d_data, d_data);

  AppServer->predict_with_io_binding();

  //AppServer->load_data_and_predict(app_ctx->h_stub->req, app_ctx->h_stub->resp, app_ctx->curr_batch_size);

//////////////////////// Cleanup

  printf("Run was successful\n");

  delete AppServer;

  return 0;
}
