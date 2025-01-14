#!/bin/bash

jupyter notebook \
	--notebook-dir=notebooks \
	--port=8891 \
	--allow-root \
	--ip=0.0.0.0 \
	--no-browser
