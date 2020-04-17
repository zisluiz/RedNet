FROM nvidia/cuda:10.0-cudnn7-devel

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository universe && \
    add-apt-repository main && \
    apt-get update

#OpenCV
RUN apt-get -y install build-essential
RUN apt-get -y install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get -y install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
RUN apt-get -y install wget unzip

RUN mkdir /usr/libs
RUN cd /usr/libs && \
    wget -O opencv-3.4.6.zip https://github.com/opencv/opencv/archive/3.4.6.zip && \
    wget -O opencv-contrib-3.4.6.zip https://github.com/opencv/opencv_contrib/archive/3.4.6.zip && \
    unzip opencv-3.4.6.zip && \
    unzip opencv-contrib-3.4.6.zip
RUN apt-get install -y python3.6-dev python3-numpy
RUN cd /usr/libs/opencv-3.4.6 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=/usr/libs/opencv_contrib-3.4.6/modules/ -DPYTHON2_EXECUTABLE=/usr/bin/python2.7 -DPYTHON_INCLUDE_DIR=/usr/include/python2.7 -DPYTHON_INCLUDE_DIR2=/usr/include/x86_64-linux-gnu/python2.7 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7.so -DPYTHON2_NUMPY_INCLUDE_DIRS=/usr/lib/python2.7/dist-packages/numpy/core/include/ -DPYTHON3_EXECUTABLE=/usr/bin/python3.6 -DPYTHON3_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON3_INCLUDE_DIR2=/usr/include/x86_64-linux-gnu/python3.6m -DPYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so -DPYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include/ .. && \
    make -j7 && \
    make install
RUN cd /usr/libs && rm -R opencv-3.4.6 && rm -R opencv_contrib-3.4.6

RUN apt-get install -y libatlas-base-dev liblapack-dev libblas-dev

RUN apt-get update
#RUN apt-get install -y libboost-all-dev

RUN apt-get install -y libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libhdf5-dev
RUN apt-get install -y liblmdb-dev libleveldb-dev libsnappy-dev

RUN apt remove -y --purge cmake
RUN cd /usr/libs && \
    wget -O cmake-3.13.4.tar.gz https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4.tar.gz && \
    tar zxvf cmake-3.13.4.tar.gz && \
    cd cmake-3.13.4 && \
    ./bootstrap && make && \
    make install

RUN export DEBIAN_FRONTEND=noninteractive && apt-get -y install python-pip python-tk && pip install scikit-image protobuf

RUN apt-get install -y libeigen3-dev libflann-dev libvtk7-dev

RUN cd /usr/libs && \
 wget --no-verbose https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz && \
 tar xzf boost_1_65_1.tar.gz && \
 cd boost_1_65_1 && \
 ln -s /usr/local/include/python3.6m /usr/local/include/python3.6 && \
 ./bootstrap.sh --with-python=$(which python3) && \
 ./b2 install && \
 rm /usr/local/include/python3.6 && \
 ldconfig && \
 cd / && rm -rf /usr/libs/boost_1_65_1/

RUN cd /usr/libs && \
    wget -O pcl-1.8.1.tar.gz https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.1.tar.gz && \
    tar zxvf pcl-1.8.1.tar.gz

RUN cd /usr/libs/pcl-pcl-1.8.1 && mkdir build && cd build && cmake .. && make -j2 && make -j2 install
RUN rm -R /usr/libs/pcl-pcl-1.8.1
RUN rm -R /usr/libs/cmake-3.13.4 && rm /usr/libs/opencv-3.4.6.zip && rm /usr/libs/opencv-contrib-3.4.6.zip && rm /usr/libs/pcl-1.8.1.tar.gz && rm /usr/libs/cmake-3.13.4.tar.gz

RUN apt-get -y install python3-pip

COPY requirements.txt /usr/libs/requirements.txt

RUN pip3 install -r /usr/libs/requirements.txt

RUN ln -s /usr/local/lib/libboost_python3.so /usr/local/lib/libboost_python-36.so

RUN echo 'export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python3.6m/"' >  ~/.bashrc_profile
RUN /bin/bash -c "source ~/.bashrc_profile"

#Tensorflow 1.13-gpu
#RUN pip3 install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.14.0-cp36-cp36m-linux_x86_64.whl
#RUN pip3 install --user  tensorflow-gpu
RUN pip3 install --user tensorflow-gpu==1.13.1
#RUN pip3 install scipy

#Pytorch
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.4.1%2Bcu100-cp36-cp36m-linux_x86_64.whl


#RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
#RUN pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
#RUN pip3 install torch torchvision dominate visdom
RUN mkdir -p /root/.cache/torch/checkpoints/
RUN wget -O /root/.cache/torch/checkpoints/vgg16-397923af.pth https://download.pytorch.org/models/vgg16-397923af.pth

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

#COPY caffe/ /usr/libs/caffe/
#RUN cd /usr/libs/caffe && mkdir build && cd build && cmake .. && make

RUN pip3 install scipy==1.2.0 imutils tensorboardX imageio matplotlib scikit-image

RUN cd /usr/libs/ && git clone https://github.com/Algomorph/pyboostcvconverter.git && cd pyboostcvconverter && \
    mkdir build && cd build && cmake -DPYTHON_DESIRED_VERSION:STRING=3.X .. && make && make install

RUN pip3 install psutil nvidia-ml-py3 scikit-learn

RUN /bin/bash -c "source ~/.bashrc_profile"

RUN apt-get -y install gdb

RUN mkdir /usr/app

#COPY MATLAB/ /usr/app/MATLAB/

WORKDIR /usr/app/

#docker build -t obeach:latest -f obeachDockerfile .
#docker run -it --privileged --runtime=nvidia --net=host --ipc=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=unix$DISPLAY --device /dev/dri -v="$HOME/.Xauthority:/root/.Xauthority:rw" -v /home/zis/dev/projects/graph-canny-segm/:/usr/app/graph-canny-segm/ -v /home/zis/dev/projects/cpf_segmentation/:/usr/app/cpf_segmentation/ -v /home/zis/dev/projects/4D_Segmentation/:/usr/app/4D_Segmentation/ -v /home/zis/dev/projects/FCN.tensorflow/:/usr/app/FCN.tensorflow/ -v /home/zis/dev/projects/fusenet-pytorch/:/usr/app/fusenet-pytorch/ -v /home/zis/dev/projects/rgbd-saliency/:/usr/app/rgbd-saliency/ obeach bash
#docker run -it --privileged --runtime=nvidia --net=host --ipc=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=unix$DISPLAY --device /dev/dri -v="$HOME/.Xauthority:/root/.Xauthority:rw" -v /home/zis/dev/code/graph-canny-segm/:/usr/app/graph-canny-segm/ -v /home/zis/dev/code/cpf_segmentation/:/usr/app/cpf_segmentation/ -v /home/zis/dev/code/4D_Segmentation/:/usr/app/4D_Segmentation/ -v /home/zis/dev/code/FCN.tensorflow/:/usr/app/FCN.tensorflow/ -v /home/zis/dev/code/fusenet-pytorch/:/usr/app/fusenet-pytorch/ -v /home/zis/dev/code/DepthAwareCNN/:/usr/app/DepthAwareCNN/ -v /home/zis/dev/code/RedNet/:/usr/app/RedNet/ -v /home/zis/dev/datasets/:/usr/app/datasets/ -v /home/zis/dev/code/obeach/:/usr/app/obeach/ obeach bash

#docker run -it --privileged --runtime=nvidia --net=host --ipc=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=unix$DISPLAY --device /dev/dri -v="$HOME/.Xauthority:/root/.Xauthority:rw" -v /home/zis/dev/code/graph-canny-segm/:/usr/app/graph-canny-segm/ -v /home/zis/dev/code/cpf_segmentation/:/usr/app/cpf_segmentation/ -v /home/zis/dev/code/4D_Segmentation/:/usr/app/4D_Segmentation/ -v /home/zis/dev/code/FCN.tensorflow/:/usr/app/FCN.tensorflow/ -v /home/zis/dev/code/fusenet-pytorch/:/usr/app/fusenet-pytorch/ -v /home/zis/dev/code/test/FCN.tensorflow:/usr/app/test/FCN.tensorflow/ -v /home/zis/dev/code/pytorch-fcn/:/usr/app/pytorch-fcn/ -v /home/zis/dev/code/RedNet/:/usr/app/RedNet/ -v /home/zis/dev/datasets/:/usr/app/datasets/ -v /home/zis/dev/code/obeach/:/usr/app/obeach/ obeach bash
