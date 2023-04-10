# splitrpc
A {Control + Data} Path Splitting RPC Stack for ML Inference Serving

## Pre-requisites
There are a whole bunch of dependencies for SplitRPC to work. Please look at REQUIREMENTS.md file for list.
Some of them need to be downloaded, and installed. Usually, the location of the install needs to be specified in one of the conf\_scripts/ files.
The variable names will be self-explanatory for what it represents.

## Organization
- src/ contains the code base and conf\_scripts/ contains the configuration files. They need to be included before building/running the server.
- src/lib\_loadgen contains the load generator using the SplitRPC client/gRPC client. 
- src/apps contains the code for the app and the SplitRPC wrapper (for e.g., for LeNet, it is 'lenet\_rpc\_bench.cc)
- src/grpc\_bench contains the gRPC versions of all the apps
- src/splitrpc\_server contains the SplitRPC server side code for commodity NICs
- src/splitrpc\_rdma\_transport contains the SplitRPC server implemented on the BlueField NIC. 
'gpu\_dpdk\_rdma\_proxy\_handler.cc' implements the SmartNIC side code and 'gpu\_rdma\_batched\_server.cc' implements the host side code.
- transport contains some common DPDK related code that implements some core/common parts of SplitRPC


## To run the server/client
- Please look inside the respective folders inside src/. There are helper scripts to run the server/load generator	
