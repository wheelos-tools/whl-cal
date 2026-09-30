#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --all --diff <git-ref>" >&2
}

run_all=false
diff_ref=""
while (($#)); do
  case "$1" in
    --all)
      run_all=true
      shift
      ;;
    --diff)
      if (($# < 2)); then
        usage
        exit 2
      fi
      diff_ref=$2
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ "$run_all" != true ]]; then
  usage
  exit 2
fi

if [[ -z "$diff_ref" ]]; then
  usage
  exit 2
fi

git rev-parse --verify "$diff_ref" >/dev/null
mapfile -t files < <(git diff --name-only --diff-filter=ACMRT "$diff_ref")
mapfile -t added_python_files < <(
  git diff --name-status --diff-filter=A "$diff_ref" |
    awk '$2 ~ /\.py$/ { print $2 }'
)

python_files=()
cpp_files=()
shell_files=()
bazel_files=()
for file in "${files[@]}"; do
  [[ -f "$file" ]] || continue
  case "$file" in
    *.py)
      python_files+=("$file")
      ;;
    *.c|*.cc|*.cpp|*.h|*.hpp)
      case "$file" in
        third_party/gril_native/include/Fusion/*|\
        third_party/gril_native/include/ikd-Tree/*|\
        third_party/gril_native/include/GroundSegmentation/PatchworkppNative.h)
          continue
          ;;
      esac
      cpp_files+=("$file")
      ;;
    *.sh)
      shell_files+=("$file")
      ;;
    BUILD|BUILD.bazel|*.bzl)
      bazel_files+=("$file")
      ;;
  esac
done

if ((${#python_files[@]})); then
  black --check --target-version py310 "${python_files[@]}"
  isort --check-only "${python_files[@]}"
fi

if ((${#added_python_files[@]})); then
  # Existing touched modules have legacy Flake8 debt; new Python files are
  # checked strictly while Black handles formatting across all changed files.
  flake8 \
    --extend-ignore=E203 \
    "${added_python_files[@]}"
fi

if ((${#cpp_files[@]})); then
  if command -v clang-format-18 >/dev/null 2>&1; then
    clang-format-18 --dry-run --Werror "${cpp_files[@]}"
  else
    clang-format --dry-run --Werror "${cpp_files[@]}"
  fi
fi

if ((${#shell_files[@]})); then
  shellcheck --exclude=SC1091 "${shell_files[@]}"
fi

if ((${#bazel_files[@]})); then
  buildifier -mode=check "${bazel_files[@]}"
fi
