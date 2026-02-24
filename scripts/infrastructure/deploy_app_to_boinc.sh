#!/bin/bash
# Script to compile and distribute the BOINC Application
set -e

APP_DIR="/root/projects/ave_alpha_search"
APP_NAME="ave_alpha"
VERSION="1.02"
PLATFORM="x86_64-pc-linux-gnu"
DEST_APP_DIR="$APP_DIR/apps/$APP_NAME/$VERSION/$PLATFORM"

echo "==> Compiling AVE Alpha C++ Physics Engine against BOINC API..."
g++ /root/boinc_alpha_derivation.cpp -o /root/ave_alpha_engine -std=c++11 -O3 \
    -I/root/boinc/api -I/root/boinc/lib -I/root/boinc \
    /root/boinc/api/.libs/libboinc_api.a /root/boinc/lib/.libs/libboinc.a \
    -pthread

echo "==> Registering App in project.xml..."
cd $APP_DIR
# Drop closing tag, insert app, replace closing tag
sed -i '/<\/boinc>/d' project.xml
cat <<EOF >> project.xml
    <app>
        <name>ave_alpha</name>
        <user_friendly_name>AVE Alpha Derivation Engine</user_friendly_name>
    </app>
</boinc>
EOF

echo "==> Initializing DB Registration (xadd)..."
./bin/xadd

echo "==> Creating Application Tree..."
mkdir -p "$DEST_APP_DIR"
cp /root/ave_alpha_engine "$DEST_APP_DIR/${APP_NAME}_${VERSION}_${PLATFORM}"

echo "==> Updating BOINC Versions..."
# Creates the MD5 signatures and database entries for the binary download
yes | ./bin/update_versions

echo "==> Configuring Crontab..."
(crontab -l 2>/dev/null | grep -v "run_daemons"; echo "*/5 * * * * cd /root/projects/ave_alpha_search/html/ops && ./run_daemons") | crontab -

echo "==> Starting BOINC Project Daemons..."
./bin/start

echo "==> Application and Daemons Successfully deployed."
