#!/bin/bash
# AVE Framework BOINC Master Server Initialization
set -e

export DEBIAN_FRONTEND=noninteractive

echo "==> Updating Ubuntu Packages..."
apt-get update -y

echo "==> Installing LAMP Stack and GCC Compilation Dependencies..."
apt-get install -y mysql-server apache2 php php-mysql php-cli php-xml php-gd \
    make m4 pkg-config autoconf automake libtool git python3 \
    libmariadb-dev libssl-dev libcurl4-openssl-dev pkg-config python3-mysqldb curl

echo "==> Cloning Berkeley (BOINC) Master Repository..."
cd /root
if [ ! -d "boinc" ]; then
    git clone https://github.com/BOINC/boinc.git
fi

echo "==> Compiling BOINC Daemons (This will take a few minutes)..."
cd boinc
./_autosetup
./configure --disable-client --disable-manager
make -j4

echo "==> BOINC Framework Compilation Successful!"
