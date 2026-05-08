FROM osrf/ros:melodic-desktop-full

# Add ubuntu user with same UID and GID as your host system, if it doesn't already exist
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi
# Add sudo support for the non-root user
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install Git
RUN sudo apt install -y git

# Rosdep update
RUN rosdep update

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

################################
## ADD ANY CUSTOM SETUP BELOW ##
################################

RUN sudo apt-get update && \
    sudo apt-get install -y libcdd-dev libgmp-dev libarmadillo-dev && \
    sudo ln -s /usr/include/cdd /usr/include/cddlib

# 2. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Create a virtual environment
RUN uv venv /home/$USERNAME/venv --python 3.9
ENV VIRTUAL_ENV="/home/$USERNAME/venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv pip install defusedxml rospkg netifaces numpy transforms3d scipy scikit-learn pycddlib
RUN sudo apt install ros-melodic-desktop-full --fix-missing

RUN sudo apt install -y ros-melodic-jackal-simulator ros-melodic-jackal-desktop ros-melodic-jackal-navigation


WORKDIR /home/$USERNAME/jackal_ws/src
# RUN git clone https://github.com/tysik/obstacle_detector.git
# RUN git clone https://github.com/LucasWEIchen/lidar_tracking.git
# RUN git clone https://github.com/NKU-MobFly-Robotics/laser-line-segment.git

WORKDIR /home/$USERNAME/jackal_ws
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && catkin_make"
RUN echo "source /home/$USERNAME/jackal_ws/devel/setup.bash" >> ~/.bashrc
