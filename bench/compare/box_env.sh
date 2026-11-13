# Source on the Linux box before building or running bench/compare:
#   source bench/compare/box_env.sh
# Bend's C backend needs clang >= 14 and the CUDA toolkit headers. The box has
# neither clang nor a CUDA_HOME layout, so this uses a user-local clang 19 from
# `mise install conda:clang@19.1.7` and the distro CUDA 12.4 under /usr.
export BEND_NO_TELEMETRY=1
export CUDA_HOME=/usr CUDA_COMPUTE_CAP=86
export PATH="$(mise where github:bendlang/bend@2.0.40)/bin:$(mise where conda:clang@19.1.7)/bin:$PATH"
export C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export CARGO_TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/target/bench"
