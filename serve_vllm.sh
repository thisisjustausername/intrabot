vllm serve Qwen/Qwen3.8-27B-FP8 \
	--max-model-len 65536 \
	--hf-overrides '{"disable_mm_encoder": true}' \
	--max-num-seqs 32 \
	--enable-auto-tool-choice \
	--tool-call-parser qwen3_xml \
	--port 8010
