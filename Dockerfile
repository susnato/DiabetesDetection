FROM nvcr.io/nvidia/tritonserver:23.12-py3

WORKDIR /

COPY ./triton_files /models

# clean and install packages, tree
RUN apt-get -y update && apt-get -y --no-install-recommends install python3 python3-pip && apt-get install -y tree htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/apt/apt.conf.d/docker-clean

RUN tree /models

# transfer the files and run the server
CMD ["/bin/bash", "-c", "tree /models && cat /models/xgb/config.pbtxt && tritonserver --model-repository=/models --cuda-memory-pool-byte-size 0:500000000"]

EXPOSE 8000