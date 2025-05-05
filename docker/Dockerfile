FROM rockylinux:9-minimal AS base

LABEL maintainer="Geon Technologies" \
      description="composite-based components"

RUN microdnf -y update && microdnf -y upgrade

RUN adduser -s /bin/bash composite-dev

FROM base AS build

RUN microdnf install -y --enablerepo=crb \ 
    gcc-toolset-13-gcc-c++ \
    cmake \
    git-core \
    vim \
    gdb \
    fftw-devel \
    openssl-devel \
    meson \
    ninja-build \
    python3-pip \
    numactl-devel \
    gcc-toolset-13-libatomic-devel \
    pciutils \
    elfutils \
    libpcap-devel \
    libbpf-devel \
    libbpf-tools \
    zlib-devel \
    libxdp-devel

FROM build AS dpdk-build

WORKDIR /opt/src
RUN pip3 install pyelftools
RUN git clone --branch v25.03 https://github.com/DPDK/dpdk.git
RUN set -ex; \
    cd dpdk; \
    . /opt/rh/gcc-toolset-13/enable; \
    meson setup \
        --prefix=/opt/dpdk \
        -Dplatform=generic \
        -Dcpu_instruction_set=x86-64-v2 \
        -Duse_hpet=false \
        -Dtests=false \
        build; \
    ninja -C build -j 8; \
    ninja -C build install

FROM build AS project

COPY --from=dpdk-build /opt/dpdk /usr/local
COPY include /opt/composite-comps/include
COPY src /opt/composite-comps/src
COPY CMakeLists.txt /opt/composite-comps
WORKDIR /opt/composite-comps

RUN . /opt/rh/gcc-toolset-13/enable && \
    cmake -B docker-build \
        -DCOMPOSITE_MODULE_RPATH_VALUE=/usr/local/lib64 \
        -DCOMPOSITE_USE_NATS=OFF \
        -DCOMPOSITE_INSTALL=ON \
        -DCMAKE_INSTALL_PREFIX=/opt/usr/local \
        -DCMAKE_BUILD_TYPE=Release;
RUN . /opt/rh/gcc-toolset-13/enable && \
    cmake --build docker-build --target install --parallel 4

FROM base

RUN microdnf install -y libstdc++ vim fftw libatomic numactl

COPY --link --from=project /opt/usr/local /usr/local
COPY --link --from=project /opt/usr/local/bin/composite-cli /usr/local/bin/composite-cli-priv

RUN setcap cap_net_raw+ep /usr/local/bin/composite-cli-priv

COPY examples /usr/local/share/composite/examples
RUN chmod a+r /usr/local/share/composite/examples/*.json

ENV LD_LIBRARY_PATH=/usr/local/lib64
ENV BIN_ARGS=""

USER composite-dev
WORKDIR /home/composite-dev

CMD /usr/local/bin/composite-cli ${BIN_ARGS}
