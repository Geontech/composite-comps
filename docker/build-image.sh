#!/bin/bash

image=composite-comps
os=alpine
tag=dev-local-tmp

while getopts i:s:t: flag
do
  case "${flag}" in
    i) image=${OPTARG};;
    s) os=${OPTARG};;
    t) tag=${OPTARG};;
  esac    
done

docker build \
  -f docker/Dockerfile.$os \
  -t $image:$tag-$os \
  .
