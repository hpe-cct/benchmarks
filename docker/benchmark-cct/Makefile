.PHONY: test clean

build: upstream Dockerfile resources/*
	docker build --build-arg http_proxy --build-arg https_proxy -t benchmark-cct .
	touch build

upstream:
	docker build --build-arg http_proxy --build-arg https_proxy -t opencl:runtime nvidia-docker/ubuntu-14.04/opencl/runtime
	touch upstream

test: build
	nvidia-docker run --rm benchmark-cct

clean:
	rm -f build
