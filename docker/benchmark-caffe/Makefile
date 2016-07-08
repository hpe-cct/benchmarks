.PHONY: test clean

build: Dockerfile resources/*
	docker build --build-arg http_proxy --build-arg https_proxy -t benchmark-caffe .
	touch build

upstream:
	docker build --build-arg http_proxy --build-arg https_proxy -t caffe:gpu caffe/docker/standalone/gpu
	touch upstream

test: build
	nvidia-docker run --rm benchmark-caffe

clean:
	rm -f build upstream
