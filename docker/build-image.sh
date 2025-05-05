#!/bin/bash

image=composite-comps
tag=dev-local-dpdk

while getopts i:t: flag
do
  case "${flag}" in
    i) image=${OPTARG};;
    t) tag=${OPTARG};;
  esac    
done

docker build \
  -f docker/Dockerfile \
  -t $image:$tag \
  .
