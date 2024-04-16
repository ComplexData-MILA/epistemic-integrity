#!/bin/bash
# Read the PID from the file
PID=$(cat changemyview_pei/llama3-labeling/litellm.pid)

# Kill the process
kill $PID

# Optionally, remove the PID file
rm changemyview_pei/llama3-labeling/litellm.pid