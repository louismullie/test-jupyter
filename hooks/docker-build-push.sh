#!/bin/bash
# make sure to make this file executable:
# chmod +x docker-build-push.sh

IMAGE_NAME="your-image-name"
REGISTRY_URL="your-registry-url"
VERSION=$(git rev-parse --short HEAD)

# Build the Docker image
docker build -t $IMAGE_NAME:$VERSION .

# Tag the image with your registry's URL
docker tag $IMAGE_NAME:$VERSION $REGISTRY_URL/$IMAGE_NAME:$VERSION

# Log in to your Docker registry
echo $DOCKER_PASSWORD | docker login $REGISTRY_URL -u $DOCKER_USERNAME --password-stdin

# Push the image to your Docker registry
docker push $REGISTRY_URL/$IMAGE_NAME:$VERSION