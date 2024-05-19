
mkdir -p ~/.docker/cli-plugins;
BUILD_VERSION=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
echo "BUILD_VERSION: ${BUILD_VERSION}"
curl -Lo ~/.docker/cli-plugins/docker-buildx "https://github.com/docker/buildx/releases/download/${BUILD_VERSION}/buildx-${BUILD_VERSION}.linux-amd64"
chmod +x ~/.docker/cli-plugins/docker-buildx
docker buildx version
