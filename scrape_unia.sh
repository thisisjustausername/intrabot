#!/usr/bin/env bash

for i in {1..15}; do
	echo "Run $i of 15"
	python3 -m crawl.crawl_everything
done

echo "All 15 runs finished."
