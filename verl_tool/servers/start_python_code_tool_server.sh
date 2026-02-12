#!/usr/bin/env bash

# Host interface the tool server will bind to.
host="localhost"

# TCP port the tool server will listen on.
port="5500"

# Comma-separated list of tool types to enable for this server instance.
tool_type="python_code"

# Number of worker threads handling requests for each tool type.
workers_per_tool="4"

# Start the tool server in the foreground with the configured parameters.
python -m verl_tool.servers.serve \
  --host "${host}" \
  --port "${port}" \
  --tool_type "${tool_type}" \
  --workers_per_tool "${workers_per_tool}"

