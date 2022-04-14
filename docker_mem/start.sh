# Install bazel
apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
apt update && apt install bazel-4.2.1

# Install benchmark tool
wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.8.0.zip
unzip v2.8.0.zip
WORKDIR tensorflow-2.8.0
python ./configure.py
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model