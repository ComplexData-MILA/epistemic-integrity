source changemyview_pei/llama3-labeling/.env
# install litellm 
pip install litellm
pip install 'litellm[proxy]'
# read from the config.yml file for litellm
litellm --model bedrock/meta.llama3-70b-instruct-v1:0 & echo $! > changemyview_pei/llama3-labeling/litellm.pid