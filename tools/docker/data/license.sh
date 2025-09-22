#!/bin/bash

if [ -z "$ACCEPT_EULA" ]
then
    echo
    echo 'The NVIDIA Software License Agreement (EULA) must be accepted before'
    echo 'Omniverse Kit can start. The license terms for this product can be viewed at'
    echo 'https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/'
    echo
    echo 'Please accept the EULA above by setting the ACCEPT_EULA environment variable.'
    echo 'e.g.: -e "ACCEPT_EULA=Y"'
    echo
    exit 1
else
    echo
    echo 'The NVIDIA Software License Agreement (EULA) must be accepted before'
    echo 'Omniverse Kit can start. The license terms for this product can be viewed at'
    echo 'https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/'
    echo
fi
