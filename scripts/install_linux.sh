#!/bin/bash

# Go to the root of the repo
cd `git rev-parse --show-toplevel`

# Install LTH E-Building
git submodule init
git submodule update

# minerl repo
# Get distribution environment variables
. /etc/lsb-release
export repo="http://ppa.launchpad.net/openjdk-r/ppa/ubuntu"
export repo_check="$repo $DISTRIB_CODENAME main"
export repo_add="$repo main"
if ! grep -q "^deb .*$repo_check" /etc/apt/sources.list /etc/apt/sources.list.d/*; then\
    sudo add-apt-repository ppa:openjdk-r/ppa
else
    echo "openJDK repo exists already."
fi
sudo apt update
sudo apt install openjdk-8-jdk cmake

# install minerl
pip3 install --user scikit-build
pip3 install --upgrade --user minerl

# Download world
python3 scripts/download_world.py