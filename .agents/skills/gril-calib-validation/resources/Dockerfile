FROM osrf/ros:noetic-desktop-full

RUN apt-get update && apt-get install -y --no-install-recommends \
    libceres-dev python3-dev python3-matplotlib wget \
    ros-noetic-jsk-recognition-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/thirdParty
RUN wget -q https://github.com/Livox-SDK/Livox-SDK/archive/refs/heads/master.tar.gz \
    && tar zxf master.tar.gz \
    && cd Livox-SDK-master/build \
    && cmake .. \
    && make -j4 \
    && make install \
    && rm -rf /root/thirdParty/Livox-SDK-master /root/thirdParty/master.tar.gz

WORKDIR /root/livox_ws/src
RUN wget -q https://github.com/Livox-SDK/livox_ros_driver/archive/refs/tags/v2.6.0.tar.gz \
    && tar zxf v2.6.0.tar.gz \
    && rm v2.6.0.tar.gz \
    && /bin/bash -lc "source /opt/ros/noetic/setup.bash && catkin_init_workspace"

WORKDIR /root/livox_ws
RUN /bin/bash -lc "source /opt/ros/noetic/setup.bash && catkin_make -j4"

WORKDIR /root/catkin_ws
