#!/bin/bash
# BOINC Workunit Generator Initialization
set -e

APP_DIR="/root/projects/ave_alpha_search"
TEMPLATE_DIR="$APP_DIR/templates"

echo "==> Creating Input/Output XML Templates..."
mkdir -p "$TEMPLATE_DIR"

# The input template specifies what files the client downloads
cat <<EOF > "$TEMPLATE_DIR/ave_alpha_in.xml"
<file_info>
    <number>0</number>
</file_info>
<workunit>
    <file_ref>
        <file_number>0</file_number>
        <open_name>in.txt</open_name>
        <copy_file/>
    </file_ref>
    <command_line>--chunk_size 500000 --box_size 100.0</command_line>
</workunit>
EOF

# The output template specifies what the client uploads back
cat <<EOF > "$TEMPLATE_DIR/ave_alpha_out.xml"
<file_info>
    <name><OUTFILE_0/></name>
    <generated_locally/>
    <upload_when_present/>
    <max_nbytes>50000</max_nbytes>
    <url><UPLOAD_URL/></url>
</file_info>
<result>
    <file_ref>
        <file_name><OUTFILE_0/></file_name>
        <open_name>out.txt</open_name>
        <copy_file/>
    </file_ref>
</result>
EOF

echo "==> Generating Sample Input Data..."
mkdir -p "$APP_DIR/download"
echo "Dummy input data for AVE Alpha simulation chunk #1" > "$APP_DIR/download/input_001.txt"

echo "==> Enqueuing the First Workunit (id: alpha_chunk_001)..."
cd "$APP_DIR"
# Sign the input file if needed (bin/dir_hier_path makes the fanout)
# actually, BOINC recommends bin/stage_file
INPUT_PATH=$(bin/dir_hier_path input_001.txt)
echo "Dummy input data for AVE Alpha simulation chunk #1" > "$INPUT_PATH"

./bin/create_work -appname ave_alpha -wu_name alpha_chunk_001 \
    -wu_template templates/ave_alpha_in.xml \
    -result_template templates/ave_alpha_out.xml \
    input_001.txt

echo "==> Workunit Successfully Queued!"
