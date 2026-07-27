#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly REPOSITORY_ROOT="$(
  cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1
  pwd
)"

cd "${REPOSITORY_ROOT}"

# Tool versions can be overridden explicitly, but these are the supported
# project defaults.
readonly CLANG_VERSION="${CLANG_VERSION:-18}"
readonly GCC_VERSION="${GCC_VERSION:-13}"
readonly CUDA_ARCH="${CUDA_ARCH:-sm_70}"
readonly CUDA_PATH="${CUDA_PATH:-/usr/lib/cuda}"
readonly PYTHON_BIN="${PYTHON_BIN:-python3}"

readonly CLANG_FORMAT_BIN="${CLANG_FORMAT_BIN:-clang-format-${CLANG_VERSION}}"
readonly CLANG_TIDY_BIN="${CLANG_TIDY_BIN:-clang-tidy-${CLANG_VERSION}}"
readonly CLANGXX_BIN="${CLANGXX_BIN:-clang++-${CLANG_VERSION}}"
readonly GXX_BIN="${GXX_BIN:-g++-${GCC_VERSION}}"

FORMAT_FILES=()
CPP_SOURCES=()
CUDA_SOURCES=()

PYBIND11_INCLUDE=""
PYTHON_INCLUDE=""

log() {
  printf '[lint-cpp] %s\n' "$*"
}

fail() {
  if [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
    printf '::error::%s\n' "$*" >&2
  else
    printf '[lint-cpp] ERROR: %s\n' "$*" >&2
  fi

  exit 1
}

group_start() {
  local title="$1"

  if [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
    printf '::group::%s\n' "${title}"
  else
    printf '\n==> %s\n' "${title}"
  fi
}

group_end() {
  if [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
    printf '::endgroup::\n'
  fi
}

require_command() {
  local command_name="$1"

  command -v "${command_name}" >/dev/null 2>&1 ||
    fail "Required command not found: ${command_name}"
}

verify_clang_version() {
  local executable="$1"
  local version_output

  require_command "${executable}"
  version_output="$("${executable}" --version | head -n 1)"

  if ! grep -Eq \
    "version[[:space:]]+${CLANG_VERSION}([.]|[[:space:]]|$)" \
    <<<"${version_output}"; then
    fail \
      "${executable} must be Clang ${CLANG_VERSION}. Detected: ${version_output}"
  fi

  log "${version_output}"
}

verify_gcc_version() {
  local detected_version
  local detected_major

  require_command "${GXX_BIN}"

  detected_version="$("${GXX_BIN}" -dumpfullversion -dumpversion)"
  detected_major="${detected_version%%.*}"

  if [[ "${detected_major}" != "${GCC_VERSION}" ]]; then
    fail \
      "${GXX_BIN} must be GCC ${GCC_VERSION}. Detected: ${detected_version}"
  fi

  log "$("${GXX_BIN}" --version | head -n 1)"
}

load_python_configuration() {
  require_command "${PYTHON_BIN}"

  if ! PYBIND11_INCLUDE="$(
    "${PYTHON_BIN}" -c \
      'import pybind11; print(pybind11.get_include())'
  )"; then
    fail \
      "pybind11 is not installed for ${PYTHON_BIN}. Run: ${PYTHON_BIN} -m pip install 'pybind11>=2.11.1'"
  fi

  PYTHON_INCLUDE="$(
    "${PYTHON_BIN}" -c \
      'import sysconfig; print(sysconfig.get_path("include"))'
  )"

  [[ -d "${PYBIND11_INCLUDE}" ]] ||
    fail "pybind11 include directory does not exist: ${PYBIND11_INCLUDE}"

  [[ -d "${PYTHON_INCLUDE}" ]] ||
    fail "Python include directory does not exist: ${PYTHON_INCLUDE}"
}

collect_files() {
  local search_roots=()

  [[ -d include ]] && search_roots+=(include)
  [[ -d tests/sources ]] && search_roots+=(tests/sources)

  (( ${#search_roots[@]} > 0 )) ||
    fail "Neither include/ nor tests/sources/ exists"

  mapfile -d '' -t FORMAT_FILES < <(
    find "${search_roots[@]}" \
      -type f \
      \( \
        -name '*.c'   -o \
        -name '*.cc'  -o \
        -name '*.cpp' -o \
        -name '*.cxx' -o \
        -name '*.h'   -o \
        -name '*.hh'  -o \
        -name '*.hpp' -o \
        -name '*.hxx' -o \
        -name '*.cu'  -o \
        -name '*.cuh' \
      \) \
      -print0 |
      sort -z
  )

  if [[ -d tests/sources ]]; then
    mapfile -d '' -t CPP_SOURCES < <(
      find tests/sources \
        -type f \
        \( \
          -name '*.cc'  -o \
          -name '*.cpp' -o \
          -name '*.cxx' \
        \) \
        -print0 |
        sort -z
    )

    mapfile -d '' -t CUDA_SOURCES < <(
      find tests/sources \
        -type f \
        -name '*.cu' \
        -print0 |
        sort -z
    )
  fi
}

verify_environment() {
  group_start "Toolchain"

  verify_clang_version "${CLANG_FORMAT_BIN}"
  verify_clang_version "${CLANG_TIDY_BIN}"
  verify_clang_version "${CLANGXX_BIN}"
  verify_gcc_version

  require_command nvcc
  log "$(nvcc --version | tail -n 1)"
  log "$("${PYTHON_BIN}" --version)"

  [[ -d "${CUDA_PATH}" ]] ||
    fail "CUDA toolkit directory does not exist: ${CUDA_PATH}"

  load_python_configuration

  log "pybind11 include: ${PYBIND11_INCLUDE}"
  log "Python include: ${PYTHON_INCLUDE}"
  log "CUDA toolkit: ${CUDA_PATH}"
  log "CUDA architecture: ${CUDA_ARCH}"

  group_end
}

run_format_check() {
  verify_clang_version "${CLANG_FORMAT_BIN}"
  collect_files

  (( ${#FORMAT_FILES[@]} > 0 )) ||
    fail "No C, C++, or CUDA files were found for formatting"

  group_start "Clang-Format ${CLANG_VERSION}"

  "${CLANG_FORMAT_BIN}" \
    --dry-run \
    --Werror \
    --style=file \
    "${FORMAT_FILES[@]}"

  group_end
}

run_gcc_check() {
  verify_gcc_version
  load_python_configuration
  collect_files

  if (( ${#CPP_SOURCES[@]} == 0 )); then
    log "No C++ sources found; skipping GCC validation"
    return
  fi

  group_start "GCC ${GCC_VERSION} syntax validation"

  local source
  for source in "${CPP_SOURCES[@]}"; do
    log "Checking ${source}"

    "${GXX_BIN}" \
      -std=c++17 \
      -fsyntax-only \
      -I"${REPOSITORY_ROOT}/include" \
      -isystem "${PYBIND11_INCLUDE}" \
      -isystem "${PYTHON_INCLUDE}" \
      "${source}"
  done

  group_end
}

run_clang_tidy_cpp() {
  verify_clang_version "${CLANG_TIDY_BIN}"
  load_python_configuration
  collect_files

  if (( ${#CPP_SOURCES[@]} == 0 )); then
    log "No C++ sources found; skipping Clang-Tidy C++ analysis"
    return
  fi

  group_start "Clang-Tidy ${CLANG_VERSION}: C++"

  "${CLANG_TIDY_BIN}" \
    --format-style=file \
    --warnings-as-errors='*' \
    --header-filter='.*' \
    "${CPP_SOURCES[@]}" \
    -- \
    -std=c++17 \
    -fno-caret-diagnostics \
    --gcc-toolchain=/usr \
    -stdlib=libstdc++ \
    -I"${REPOSITORY_ROOT}/include" \
    -isystem "${PYBIND11_INCLUDE}" \
    -isystem "${PYTHON_INCLUDE}"

  group_end
}

run_clang_tidy_cuda() {
  verify_clang_version "${CLANG_TIDY_BIN}"
  load_python_configuration
  collect_files

  if (( ${#CUDA_SOURCES[@]} == 0 )); then
    log "No CUDA sources found; skipping Clang-Tidy CUDA analysis"
    return
  fi

  [[ -d "${CUDA_PATH}" ]] ||
    fail "CUDA toolkit directory does not exist: ${CUDA_PATH}"

  group_start "Clang-Tidy ${CLANG_VERSION}: CUDA"

  "${CLANG_TIDY_BIN}" \
    --format-style=file \
    --warnings-as-errors='*' \
    --header-filter='.*' \
    "${CUDA_SOURCES[@]}" \
    -- \
    -x cuda \
    -std=c++17 \
    -fno-caret-diagnostics \
    --gcc-toolchain=/usr \
    -stdlib=libstdc++ \
    "--cuda-path=${CUDA_PATH}" \
    "--cuda-gpu-arch=${CUDA_ARCH}" \
    -I"${REPOSITORY_ROOT}/include" \
    -isystem "${PYBIND11_INCLUDE}" \
    -isystem "${PYTHON_INCLUDE}"

  group_end
}

print_usage() {
  cat <<EOF
Usage:
  $(basename "$0") <command>

Commands:
  all         Run every lint and validation check
  verify      Verify the required local toolchain
  format      Check formatting with Clang-Format ${CLANG_VERSION}
  gcc         Validate C++ sources with GCC ${GCC_VERSION}
  tidy-cpp    Run Clang-Tidy ${CLANG_VERSION} on C++ sources
  tidy-cuda   Run Clang-Tidy ${CLANG_VERSION} on CUDA sources
  help        Show this help

Environment overrides:
  CLANG_VERSION       Default: ${CLANG_VERSION}
  GCC_VERSION         Default: ${GCC_VERSION}
  CUDA_ARCH           Default: ${CUDA_ARCH}
  CUDA_PATH           Default: ${CUDA_PATH}
  PYTHON_BIN          Default: ${PYTHON_BIN}
  CLANG_FORMAT_BIN    Default: ${CLANG_FORMAT_BIN}
  CLANG_TIDY_BIN      Default: ${CLANG_TIDY_BIN}
  CLANGXX_BIN         Default: ${CLANGXX_BIN}
  GXX_BIN             Default: ${GXX_BIN}
EOF
}

main() {
  local command="${1:-all}"

  case "${command}" in
    all)
      verify_environment
      run_format_check
      run_gcc_check
      run_clang_tidy_cpp
      run_clang_tidy_cuda
      ;;

    verify)
      verify_environment
      ;;

    format)
      run_format_check
      ;;

    gcc)
      run_gcc_check
      ;;

    tidy-cpp)
      run_clang_tidy_cpp
      ;;

    tidy-cuda)
      run_clang_tidy_cuda
      ;;

    help | --help | -h)
      print_usage
      ;;

    *)
      print_usage >&2
      fail "Unknown command: ${command}"
      ;;
  esac
}

main "$@"