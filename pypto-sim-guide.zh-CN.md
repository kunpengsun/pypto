# PyPTO-Lib 无 NPU 环境 Sim 入门指南

本文提供从创建 Python 环境、下载源码和 submodule，到完成 Sim 编译、运行及精度验证的完整步骤。

**适用范围：Linux x86_64 / aarch64、Bash、可访问 GitHub 和软件包源的 CPU 机器。PTOAS 已安装，本文不包含其安装步骤。**

## 1. 目标与环境要求

这里的 Sim 指 `a2a3sim`、`a5sim` CPU 功能仿真，用于验证程序功能和数值精度。它的执行耗时不能代表 NPU 性能，也不是 `msprof op simulator` 性能分析流程。

- 不需要 NPU、Ascend 驱动、CANN 或 `torch-npu`。
- 运行本文的 Hello World 和 Matmul 不需要下载模型权重。
- 使用 Python 3.10、GCC/G++ 15 和独立 Conda 环境。
- 机器需要已有 Bash、curl 和基本系统工具；其余开发工具在本文中安装。
- 已有的 PTOAS 应匹配主机架构，并能在该 Linux 系统运行。本文固定版本使用 PTOAS 0.57；若使用其 `manylinux_2_34` wheel，宿主系统需要 glibc 2.34 或更新版本。

可以先检查主机：

```bash
uname -m
getconf GNU_LIBC_VERSION
```

**请在同一个 Bash 终端按顺序执行。某一步失败时先解决该错误，再继续。下载代码部分按全新工作目录编写；已有同名目录时，请复用或另选目录，不要直接删除已有工作。**

## 2. 版本与验证范围

为减少不同版本混用导致的问题，本文固定到本次本机验证所用的代码版本。

| 组件 | 版本或来源 |
|---|---|
| PyPTO-Lib | `384553ad7106a50f4ab644911765ba62e1eeebde` |
| PyPTO | `f1bb0860885247ecdcaf2ccddcc45c9f1eb14382` |
| simpler runtime | 由上述 PyPTO 的 `runtime` submodule 固定，短版本 `15f5cbd9` |
| PTO ISA | 由 `pypto/runtime/pto_isa.pin` 固定，短版本 `96ba706c` |
| PTOAS | `0.57`，使用已有安装 |
| Python | `3.10` |
| GCC/G++ | `15` |
| PyTorch | `2.6.0` CPU 版 |

本机已验证 Hello World 的 `a2a3sim`、`a5sim`，以及 Matmul 的 `a2a3sim` 均通过编译、执行和精度比对。**本文的全新 Conda 安装流程尚未在同事的目标机器上执行**，应以第 10 节检查结果作为该机器环境可用的依据。

本文 Python 构建依赖使用本次本机验证的版本组合，不声称复现仓库所有 CI 构建依赖。

## 3. 需要下载哪些内容

| 下载项 | 用途 | 获取方式 |
|---|---|---|
| `pypto-lib` | 示例、模型和精度验证框架 | 单独 clone |
| `pypto` | Python DSL 和编译器 | 单独 clone |
| `simpler` | Sim runtime | `pypto/runtime` submodule |
| `libbacktrace`、`msgpack-c` | PyPTO C/C++ 依赖 | PyPTO submodule |
| `pto-isa` | 仿真所需指令实现和头文件 | runtime 按 `runtime/pto_isa.pin` 自动获取 |
| Python 依赖、编译工具 | 构建和运行 | Conda / pip |
| PTOAS | 编译工具链组件 | 使用已有安装，本文仅配置路径 |

预期目录结构：

```text
~/pypto-sim/
├── .conda/                     # 独立 Python 和编译器环境
├── toolchain-bin/              # gcc-15、g++-15 命令包装
├── activate_sim.sh             # 后续使用的环境激活脚本
├── pypto-lib/
│   ├── examples/
│   └── build_output/           # 运行时生成的编译产物
└── pypto/
    ├── 3rdparty/libbacktrace/  # submodule
    ├── 3rdparty/msgpack-c/     # submodule
    ├── runtime/               # simpler submodule
    │   ├── pto_isa.pin
    │   └── build/pto-isa/      # runtime 自动获取的 PTO ISA
    └── toolchain/versions.env  # PTOAS 版本及校验值
```

不需要再单独 clone 一份 simpler。PTO ISA 由 runtime 在需要时按固定版本自动获取，首次使用时需要联网。

## 4. 准备 Conda

### 4.1 已有 Conda

在能调用 `conda` 的终端执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
```

然后直接进入第 5 节。

### 4.2 没有 Conda

按 Miniforge 官方方式安装到用户目录，无需 sudo。以下命令要求 `$HOME/miniforge3` 尚不存在：

```bash
curl -fL --retry 3 \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
  -o /tmp/Miniforge3.sh

bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

安装来源：[Miniforge 官方说明](https://github.com/conda-forge/miniforge)。

## 5. 创建独立环境并准备 GCC 15

### 5.1 创建环境

可以修改 `SIM_WORKSPACE`，建议路径中不包含空格。

```bash
export SIM_WORKSPACE="$HOME/pypto-sim"
mkdir -p "$SIM_WORKSPACE"

conda create -y --override-channels -c conda-forge \
  -p "$SIM_WORKSPACE/.conda" \
  python=3.10 pip gcc=15 gxx=15 git curl make

conda activate "$SIM_WORKSPACE/.conda"

export PYTHONNOUSERSITE=1
unset PYTHONPATH PTO_ISA_ROOT
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT

python --version
gcc -dumpversion
g++ -dumpversion
```

Python 应为 3.10，GCC/G++ 应为 15。

### 5.2 为 Sim 提供 gcc-15 和 g++-15 命令

Sim 会按名字调用 `gcc-15`、`g++-15`。下面生成包装脚本，让它们使用当前 Conda 环境的编译器。

```bash
mkdir -p "$SIM_WORKSPACE/toolchain-bin"

for compiler in gcc g++; do
  compiler_path="$(command -v "$compiler")"

  case "$compiler_path" in
    "$CONDA_PREFIX"/*) ;;
    *)
      echo "错误：$compiler 没有使用当前 Conda 环境的编译器"
      exit 1
      ;;
  esac

  printf '#!/bin/sh\nexec "%s" "$@"\n' "$compiler_path" \
    > "$SIM_WORKSPACE/toolchain-bin/${compiler}-15"

  chmod +x "$SIM_WORKSPACE/toolchain-bin/${compiler}-15"
done

export PATH="$SIM_WORKSPACE/toolchain-bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

gcc-15 -dumpversion
g++-15 -dumpversion
```

这两个命令都应返回 15 系列版本。包装脚本不会把旧编译器升级为 GCC 15，因此必须先完成上一节的安装与检查。

## 6. 下载代码与全部 submodule

```bash
cd "$SIM_WORKSPACE"

git clone https://github.com/hw-native-sys/pypto-lib.git
git -C pypto-lib checkout 384553ad7106a50f4ab644911765ba62e1eeebde

git clone https://github.com/hw-native-sys/pypto.git
git -C pypto checkout f1bb0860885247ecdcaf2ccddcc45c9f1eb14382

git -C pypto submodule update --init --recursive
git -C pypto submodule status --recursive

export PYPTO_ROOT="$SIM_WORKSPACE/pypto"
export PYPTO_LIB_ROOT="$SIM_WORKSPACE/pypto-lib"
```

固定 commit 后出现 `detached HEAD` 提示是正常的。`submodule status` 中各项前缀应为空格；`-` 表示尚未初始化，`+` 表示当前版本与父仓库固定版本不一致。

如果下载中断，可以重新执行：

```bash
git -C "$PYPTO_ROOT" submodule update --init --recursive
```

仅下载 GitHub 页面上的源码 ZIP 不能代替上述完整的 submodule 初始化过程。

## 7. 安装 Python 依赖

```bash
python -m pip install --upgrade pip

python -m pip install \
  "setuptools==79.0.1" \
  "scikit-build-core==0.12.2" \
  "nanobind==2.12.0" \
  "cmake==4.2.3" \
  "ninja==1.13.0" \
  "numpy==2.2.6" \
  "cloudpickle==3.1.1"

python -m pip install "torch==2.6.0" \
  --index-url https://download.pytorch.org/whl/cpu

python -c 'import torch; print("torch:", torch.__version__); print(torch.randn(2, 2))'
```

使用 CPU 版 PyTorch 即可；不需要安装 torchvision、torchaudio 或 torch-npu。安装来源：[PyTorch 官方历史版本与 CPU 安装命令](https://pytorch.org/get-started/previous-versions/)。

## 8. 配置已有 PTOAS

**只修改下面这一行的路径，不需要在本文流程中安装 PTOAS。** 路径应指向匹配当前机器架构的 PTOAS 0.57 安装目录。

```bash
export PTOAS_ROOT="/实际路径/ptoas-0.57"
```

检查常见安装布局中的可执行文件：

```bash
PTOAS_EXECUTABLE=""

for relative_path in ptoas ptoas.sh bin/ptoas; do
  if [ -f "$PTOAS_ROOT/$relative_path" ] && \
     [ -x "$PTOAS_ROOT/$relative_path" ]; then
    PTOAS_EXECUTABLE="$PTOAS_ROOT/$relative_path"
    break
  fi
done

if [ -z "$PTOAS_EXECUTABLE" ]; then
  echo "错误：PTOAS_ROOT 下没有找到 PTOAS 可执行文件"
  exit 1
fi

"$PTOAS_EXECUTABLE" --version
```

应输出 `ptoas 0.57`。这里直接检查指定目录的可执行文件，避免误用 PATH 中另一版本的 `ptoas`。

## 9. 安装 PyPTO 和 Sim runtime

```bash
# 内存较小的机器可改为 2
export CMAKE_BUILD_PARALLEL_LEVEL=4

python -m pip install \
  --no-build-isolation --no-deps \
  "$PYPTO_ROOT"

# 只构建 a2a3sim 和 a5sim
python -m pip install \
  --no-build-isolation --no-deps \
  --config-settings=build.targets=build_package_sim \
  -e "$PYPTO_ROOT/runtime"
```

`--no-build-isolation` 让构建使用第 7 节安装的工具版本；`--no-deps` 避免安装过程重新选择已准备好的 Python 依赖。`build_package_sim` 指定仅构建仿真目标。

## 10. 保存激活脚本并验证

### 10.1 保存环境激活脚本

以下命令会将前面配置好的实际路径写入脚本：

```bash
cat > "$SIM_WORKSPACE/activate_sim.sh" <<EOF
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SIM_WORKSPACE/.conda"

export SIM_WORKSPACE="$SIM_WORKSPACE"
export PYPTO_ROOT="$PYPTO_ROOT"
export PYPTO_LIB_ROOT="$PYPTO_LIB_ROOT"
export PTOAS_ROOT="$PTOAS_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PYPTO_LIB_ROOT"
export PATH="$SIM_WORKSPACE/toolchain-bin:\$PATH"
export LD_LIBRARY_PATH="$SIM_WORKSPACE/.conda/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export CMAKE_BUILD_PARALLEL_LEVEL=4
export PYPTO_RUNTIME_LOG=error

unset PTO_ISA_ROOT
EOF

source "$SIM_WORKSPACE/activate_sim.sh"
```

`PYTHONPATH` 只加入 pypto-lib。PyPTO 使用刚安装的包，避免源码目录里遗留的旧二进制扩展干扰导入。`PYPTO_RUNTIME_LOG=error` 用于减少 runtime 日志，不会关闭示例的 PASS/FAIL 输出。

### 10.2 检查 Python 依赖

```bash
cd "$PYPTO_LIB_ROOT"

python - <<'PY'
import pypto
import simpler
import simpler_setup
import _task_interface
import torch
from golden import run_jit

print("pypto:", pypto.__file__)
print("simpler:", simpler.__file__)
print("simpler_setup:", simpler_setup.__file__)
print("torch:", torch.__version__)
print("Import checks passed")
PY
```

预期：PyPTO 来自当前 `.conda` 环境的 site-packages，simpler 来自本工作目录的 `pypto/runtime`。

### 10.3 运行三个测试

```bash
# A2/A3：向量运算
python examples/beginner/hello_world.py -p a2a3sim

# A2/A3：矩阵乘法
python examples/beginner/matmul.py -p a2a3sim

# A5：向量运算
python examples/beginner/hello_world.py -p a5sim
```

每个程序应完成以下流程并返回退出码 0：

```text
[RUN] compile ...
[RUN] generate inputs ...
[RUN] compute golden ...
[RUN] runtime ...
[RUN] validate ...
[RUN]   'y' PASS ...
[RUN] PASS (...)
```

输出名因示例不同而变化，Matmul 的输出是 `c`。三个测试均通过，表示该机器已经完成所需的编译、CPU 仿真和精度验证。

## 11. 后续使用

每次打开新终端：

```bash
source "$HOME/pypto-sim/activate_sim.sh"
cd "$PYPTO_LIB_ROOT"

python examples/beginner/hello_world.py -p a2a3sim
```

如果修改了工作目录，调整 `source` 的路径即可。环境脚本保存的是绝对路径，移动目录后需要重新生成；不要把 `.conda` 或激活脚本直接复制给另一台机器使用。

其他示例可以同样传入 `-p a2a3sim` 或 `-p a5sim`。模型入口的平台支持范围不同，应先查看对应脚本的 `--help`；标记 `# ci: no-sim` 的入口不适合作为 Sim 入门测试。

## 12. 常见问题

| 现象 | 检查或处理 |
|---|---|
| 找不到 `gcc-15` / `g++-15` | 重新 source 激活脚本，检查 `toolchain-bin` 和第 5 节的编译器版本。 |
| `GLIBCXX_* not found` | 确认使用当前 Conda 环境的编译器，且 `LD_LIBRARY_PATH` 优先包含该环境的 `lib`。 |
| `GLIBC_* not found` | 检查宿主 glibc 是否满足已有 PTOAS 和其他二进制包要求；Conda 的 libstdc++ 不能替代宿主 glibc。 |
| `ModuleNotFoundError: golden` | 在 pypto-lib 根目录执行，并确认 `PYTHONPATH` 包含该目录。 |
| `ModuleNotFoundError: pypto` | 检查 `command -v python` 是否指向本文环境，以及第 9 节 PyPTO 安装是否成功。 |
| 导入 `pypto.pypto_core` 时缺少符号或类型 | 检查是否给 `PYTHONPATH` 加入了另一份 PyPTO 源码路径；使用第 10 节脚本和匹配的已安装包。 |
| simpler 导入指向已删除的 worktree | 在当前环境执行 `python -m pip uninstall -y simpler`，再执行第 9 节的 runtime 安装命令。 |
| 找不到 PTOAS 或版本不一致 | 按第 8 节检查 `PTOAS_ROOT`，直接调用该目录下的可执行文件查看版本。 |
| PTO ISA 自动获取失败 | 检查 GitHub 网络与代理，恢复网络后重试失败的命令。 |
| 编译时进程被 killed | 检查是否内存不足；将 `CMAKE_BUILD_PARALLEL_LEVEL` 降为 2 或 1 后重试。 |
| 切换源码版本后出现接口不匹配 | 更新 PyPTO submodule，重新读取工具链 pin，再重新安装匹配的 PyPTO 和 runtime。 |

如果需要协助定位问题，建议提供以下输出及失败命令的完整日志：

```bash
uname -m
getconf GNU_LIBC_VERSION
command -v python
python --version
gcc-15 -dumpversion
g++-15 -dumpversion
python -m pip show pypto simpler torch
git -C "$PYPTO_ROOT" rev-parse HEAD
git -C "$PYPTO_ROOT" submodule status --recursive
cat "$PYPTO_ROOT/toolchain/versions.env"
cat "$PYPTO_ROOT/runtime/pto_isa.pin"
```

## 13. 参考资料

- [PyPTO-Lib 固定版本源码](https://github.com/hw-native-sys/pypto-lib/tree/384553ad7106a50f4ab644911765ba62e1eeebde)
- [PyPTO 固定版本源码](https://github.com/hw-native-sys/pypto/tree/f1bb0860885247ecdcaf2ccddcc45c9f1eb14382)
- [PyPTO submodule 定义](https://github.com/hw-native-sys/pypto/blob/f1bb0860885247ecdcaf2ccddcc45c9f1eb14382/.gitmodules)
- [PyPTO 工具链版本定义](https://github.com/hw-native-sys/pypto/blob/f1bb0860885247ecdcaf2ccddcc45c9f1eb14382/toolchain/versions.env)
- [Miniforge 官方说明](https://github.com/conda-forge/miniforge)
- [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)
