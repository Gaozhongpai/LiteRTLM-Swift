#!/usr/bin/env bash
#
# Build CLiteRTLM.xcframework from Google's LiteRT-LM source.
#
# Prerequisites:
#   - Bazel 7.6.1 (install via Bazelisk: brew install bazelisk)
#   - Xcode 16+ with iOS SDK
#   - ~20 GB disk space for Bazel build cache
#
# Usage:
#   ./scripts/build-xcframework.sh [/path/to/LiteRT-LM]
#
# If no path is provided, clones the repo to a temp directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/Frameworks/LiteRTLM.xcframework"
USE_ADVANCED_ENGINE="${LITERTLM_USE_ADVANCED_ENGINE:-0}"
WORK_DIR="$(mktemp -d)"
LITERT_BUILD_FILE=""
LITERT_BUILD_BACKUP=""
LITERT_CORE_BUILD_FILE=""
LITERT_CORE_BUILD_BACKUP=""
LITERT_ENGINE_CC=""
LITERT_ENGINE_CC_BACKUP=""
LITERT_ENGINE_H=""
LITERT_ENGINE_H_BACKUP=""

cleanup() {
    local status=$?
    if [ -n "${LITERT_BUILD_FILE:-}" ] && [ -n "${LITERT_BUILD_BACKUP:-}" ] && [ -f "$LITERT_BUILD_BACKUP" ]; then
        cp "$LITERT_BUILD_BACKUP" "$LITERT_BUILD_FILE"
    fi
    if [ -n "${LITERT_CORE_BUILD_FILE:-}" ] && [ -n "${LITERT_CORE_BUILD_BACKUP:-}" ] && [ -f "$LITERT_CORE_BUILD_BACKUP" ]; then
        cp "$LITERT_CORE_BUILD_BACKUP" "$LITERT_CORE_BUILD_FILE"
    fi
    if [ -n "${LITERT_ENGINE_CC:-}" ] && [ -n "${LITERT_ENGINE_CC_BACKUP:-}" ] && [ -f "$LITERT_ENGINE_CC_BACKUP" ]; then
        cp "$LITERT_ENGINE_CC_BACKUP" "$LITERT_ENGINE_CC"
    fi
    if [ -n "${LITERT_ENGINE_H:-}" ] && [ -n "${LITERT_ENGINE_H_BACKUP:-}" ] && [ -f "$LITERT_ENGINE_H_BACKUP" ]; then
        cp "$LITERT_ENGINE_H_BACKUP" "$LITERT_ENGINE_H"
    fi
    rm -rf "$WORK_DIR"
    exit "$status"
}
trap cleanup EXIT

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Locate or clone LiteRT-LM source
# ---------------------------------------------------------------------------

LITERT_LM_DIR="${1:-}"

if [ -z "$LITERT_LM_DIR" ]; then
    LITERT_LM_DIR="$WORK_DIR/LiteRT-LM"
    info "Cloning LiteRT-LM source..."
    git clone --depth 1 https://github.com/google-ai-edge/LiteRT-LM.git "$LITERT_LM_DIR"
fi

if [ ! -f "$LITERT_LM_DIR/c/BUILD" ]; then
    error "Invalid LiteRT-LM source directory: $LITERT_LM_DIR (missing c/BUILD)"
fi

LITERT_LM_DIR="$(cd "$LITERT_LM_DIR" && pwd)"
LITERT_BUILD_FILE="$LITERT_LM_DIR/c/BUILD"
LITERT_CORE_BUILD_FILE="$LITERT_LM_DIR/runtime/core/BUILD"
LITERT_ENGINE_CC="$LITERT_LM_DIR/c/engine.cc"
LITERT_ENGINE_H="$LITERT_LM_DIR/c/engine.h"

info "Using LiteRT-LM source at: $LITERT_LM_DIR"
if [ "$USE_ADVANCED_ENGINE" = "1" ]; then
    info "Packaging with LiteRT-LM advanced engine (conversation clone capable when backend supports CreateNewContext)"
else
    info "Packaging with LiteRT-LM basic engine (stable default for LiteRT compiled-model backend)"
fi

# Upstream has the advanced engine/session implementation in source, but some
# checkouts do not expose it from Bazel. Add the missing targets temporarily so
# CLiteRTLM uses SessionAdvanced, which implements conversation cloning.
if [ "$USE_ADVANCED_ENGINE" = "1" ] && ! grep -q 'name = "engine_advanced_impl"' "$LITERT_CORE_BUILD_FILE"; then
    warn "Injecting temporary advanced engine Bazel targets"
    LITERT_CORE_BUILD_BACKUP="$WORK_DIR/runtime_core_BUILD.orig"
    cp "$LITERT_CORE_BUILD_FILE" "$LITERT_CORE_BUILD_BACKUP"
    cat >> "$LITERT_CORE_BUILD_FILE" << 'BAZEL_TARGET'

# Added temporarily by LiteRTLM-Swift/scripts/build-xcframework.sh.
cc_library(
    name = "session_advanced",
    srcs = ["session_advanced.cc"],
    hdrs = ["session_advanced.h"],
    deps = [
        ":session_utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//runtime/components:tokenizer",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/framework/resource_management:execution_manager",
        "//runtime/proto:sampler_params_cc_proto",
    ],
)

cc_library(
    name = "engine_advanced_impl",
    srcs = ["engine_advanced_impl.cc"],
    deps = ENGINE_IMPL_COMMON_DEPS + [
        ":session_advanced",
        "//runtime/components:default_static_gpu_samplers",
        "//runtime/executor:default_static_gpu_accelerator",
        "//runtime/framework/resource_management:execution_manager",
        "//runtime/framework/resource_management:serial_execution_manager",
        "//runtime/framework/resource_management:threaded_execution_manager",
        "//runtime/util:litert_util",
    ],
    alwayslink = 1,
)
BAZEL_TARGET
fi

# Official LiteRT-LM exposes the C library target but not an iOS shared-library
# target with the framework binary name this Swift package ships. Add that
# Bazel target temporarily for packaging only, then restore c/BUILD on exit.
if ! grep -q 'name = "libLiteRTLMEngine.dylib"' "$LITERT_BUILD_FILE"; then
    warn "Injecting temporary //c:libLiteRTLMEngine.dylib build target"
    LITERT_BUILD_BACKUP="$WORK_DIR/c_BUILD.orig"
    cp "$LITERT_BUILD_FILE" "$LITERT_BUILD_BACKUP"
    if [ "$USE_ADVANCED_ENGINE" = "1" ]; then
    cat >> "$LITERT_BUILD_FILE" << 'BAZEL_TARGET'

# Added temporarily by LiteRTLM-Swift/scripts/build-xcframework.sh.
cc_library(
    name = "engine_advanced",
    srcs = [
        "engine.cc",
    ],
    hdrs = [
        "engine.h",
    ],
    deps = ENGINE_COMMON_DEPS + [
        "//runtime/core:engine_advanced_impl",
    ],
    alwayslink = True,
)

cc_binary(
    name = "libLiteRTLMEngine.dylib",
    linkshared = True,
    deps = [
        ":engine_advanced",
    ],
)
BAZEL_TARGET
    else
    cat >> "$LITERT_BUILD_FILE" << 'BAZEL_TARGET'

# Added temporarily by LiteRTLM-Swift/scripts/build-xcframework.sh.
cc_binary(
    name = "libLiteRTLMEngine.dylib",
    linkshared = True,
    deps = [
        ":engine",
    ],
)
BAZEL_TARGET
    fi
fi

# Official upstream has the C++ ConversationConfig flag for eager preface
# prefill, but not the C setter needed by Swift. Inject a tiny temporary C API
# shim during packaging, then restore c/engine.{cc,h} on exit so the upstream
# checkout stays clean.
if ! grep -q 'litert_lm_conversation_config_set_prefill_preface_on_init' "$LITERT_ENGINE_H"; then
    warn "Injecting temporary prefill_preface_on_init C API shim"
    LITERT_ENGINE_CC_BACKUP="$WORK_DIR/c_engine.cc.orig"
    LITERT_ENGINE_H_BACKUP="$WORK_DIR/c_engine.h.orig"
    cp "$LITERT_ENGINE_CC" "$LITERT_ENGINE_CC_BACKUP"
    cp "$LITERT_ENGINE_H" "$LITERT_ENGINE_H_BACKUP"

    perl -0pi -e 's/(  bool filter_channel_content_from_kv_cache = false;\n)/$1  bool prefill_preface_on_init = false;\n/' "$LITERT_ENGINE_CC"
    perl -0pi -e 's/(void litert_lm_conversation_config_set_filter_channel_content_from_kv_cache\(\n    LiteRtLmConversationConfig\* config,\n    bool filter_channel_content_from_kv_cache\) \{\n  if \(config\) \{\n    config->filter_channel_content_from_kv_cache =\n        filter_channel_content_from_kv_cache;\n  \}\n\}\n)/$1\nvoid litert_lm_conversation_config_set_prefill_preface_on_init(\n    LiteRtLmConversationConfig* config, bool prefill_preface_on_init) {\n  if (config) {\n    config->prefill_preface_on_init = prefill_preface_on_init;\n  }\n}\n/' "$LITERT_ENGINE_CC"
    perl -0pi -e 's/(    builder\.SetFilterChannelContentFromKvCache\(\n        c_config->filter_channel_content_from_kv_cache\);\n)/$1    builder.SetPrefillPrefaceOnInit(c_config->prefill_preface_on_init);\n/' "$LITERT_ENGINE_CC"
    perl -0pi -e 's/(void litert_lm_conversation_config_set_filter_channel_content_from_kv_cache\(\n    LiteRtLmConversationConfig\* config,\n    bool filter_channel_content_from_kv_cache\);\n)/$1\n\/\/ Sets whether the conversation should prefill the preface \(system message,\n\/\/ tools, and history\) into the KV cache during conversation creation. When\n\/\/ true, the prefill cost is paid up front by litert_lm_conversation_create;\n\/\/ the first user turn then only prefills the new turn. When false \(default\),\n\/\/ the entire context is prefilled on the first turn.\n\/\/ @param config The config to modify.\n\/\/ @param prefill_preface_on_init Whether to prefill the preface at init.\nLITERT_LM_C_API_EXPORT\nvoid litert_lm_conversation_config_set_prefill_preface_on_init(\n    LiteRtLmConversationConfig* config, bool prefill_preface_on_init);\n/' "$LITERT_ENGINE_H"
fi

# ---------------------------------------------------------------------------
# 2. Check prerequisites
# ---------------------------------------------------------------------------

if ! command -v bazel &>/dev/null && ! command -v bazelisk &>/dev/null; then
    error "Bazel not found. Install via: brew install bazelisk"
fi

BAZEL_CMD="bazel"
if command -v bazelisk &>/dev/null; then
    BAZEL_CMD="bazelisk"
fi

if ! xcode-select -p &>/dev/null; then
    error "Xcode command line tools not found. Run: xcode-select --install"
fi

info "Using $($BAZEL_CMD --version | head -1)"
info "Using $(xcodebuild -version | head -1)"

DEVELOPER_DIR="$(xcode-select -p)"
export DEVELOPER_DIR

XCODE_SHORT_VERSION="$(xcodebuild -version | awk '/^Xcode / { print $2; exit }')"
XCODE_BUILD_VERSION="$(xcodebuild -version | awk '/^Build version / { print $3; exit }')"

if [ -z "$XCODE_SHORT_VERSION" ] || [ -z "$XCODE_BUILD_VERSION" ]; then
    error "Failed to determine active Xcode version/build from xcodebuild -version"
fi

BAZEL_XCODE_FLAGS=(
    "--xcode_version=$XCODE_SHORT_VERSION"
    "--repo_env=DEVELOPER_DIR=$DEVELOPER_DIR"
    "--action_env=DEVELOPER_DIR=$DEVELOPER_DIR"
)

info "Pinning Bazel to Xcode $XCODE_SHORT_VERSION ($XCODE_BUILD_VERSION)"

# ---------------------------------------------------------------------------
# 3. Build for iOS device (arm64)
# ---------------------------------------------------------------------------

info "Building for iOS device (arm64)..."
cd "$LITERT_LM_DIR"

$BAZEL_CMD build "${BAZEL_XCODE_FLAGS[@]}" --config=ios_arm64 //c:libLiteRTLMEngine.dylib

DEVICE_DYLIB="$LITERT_LM_DIR/bazel-bin/c/libLiteRTLMEngine.dylib"
if [ ! -f "$DEVICE_DYLIB" ]; then
    error "Device build failed: $DEVICE_DYLIB not found"
fi
info "Device build OK: $(du -h "$DEVICE_DYLIB" | cut -f1)"

# ---------------------------------------------------------------------------
# 4. Build for iOS simulator (arm64)
# ---------------------------------------------------------------------------

info "Building for iOS simulator (arm64)..."

$BAZEL_CMD build "${BAZEL_XCODE_FLAGS[@]}" --config=ios_sim_arm64 //c:libLiteRTLMEngine.dylib

SIM_DYLIB="$LITERT_LM_DIR/bazel-bin/c/libLiteRTLMEngine.dylib"
if [ ! -f "$SIM_DYLIB" ]; then
    error "Simulator build failed: $SIM_DYLIB not found"
fi
info "Simulator build OK: $(du -h "$SIM_DYLIB" | cut -f1)"

# Copy simulator dylib aside (Bazel overwrites bazel-bin between configs)
SIM_DYLIB_COPY="$WORK_DIR/libLiteRTLMEngine-sim.dylib"
cp "$SIM_DYLIB" "$SIM_DYLIB_COPY"

# Rebuild device to restore bazel-bin
info "Restoring device build..."
$BAZEL_CMD build "${BAZEL_XCODE_FLAGS[@]}" --config=ios_arm64 //c:libLiteRTLMEngine.dylib
DEVICE_DYLIB="$LITERT_LM_DIR/bazel-bin/c/libLiteRTLMEngine.dylib"

# Also grab companion dylibs if present. LiteRT-LM loads the Metal GPU
# accelerator and sampler dynamically at runtime, so they must be shipped next
# to CLiteRTLM and copied into the app bundle's Frameworks directory.
# Newer LiteRT-LM checkouts stage these in prebuilt/<platform>/ instead of
# bazel-bin/c/.
DEVICE_EXTRA_DYLIBS=()
SIM_EXTRA_DYLIBS=()

collect_extra_dylib() {
    local OUT_ARRAY_NAME="$1"
    local PLATFORM_DIR="$2"
    local NAME="$3"
    local CANDIDATE=""

    if [ -f "$LITERT_LM_DIR/prebuilt/$PLATFORM_DIR/$NAME" ]; then
        CANDIDATE="$LITERT_LM_DIR/prebuilt/$PLATFORM_DIR/$NAME"
        info "Found $PLATFORM_DIR $NAME in prebuilt/$PLATFORM_DIR"
    elif [ -f "$LITERT_LM_DIR/bazel-bin/c/$NAME" ]; then
        CANDIDATE="$LITERT_LM_DIR/bazel-bin/c/$NAME"
        info "Found $PLATFORM_DIR $NAME in bazel-bin/c"
    fi

    if [ -n "$CANDIDATE" ]; then
        eval "$OUT_ARRAY_NAME+=(\"\$CANDIDATE\")"
    fi
}

collect_extra_dylib DEVICE_EXTRA_DYLIBS ios_arm64 libGemmaModelConstraintProvider.dylib
collect_extra_dylib DEVICE_EXTRA_DYLIBS ios_arm64 libLiteRtMetalAccelerator.dylib

collect_extra_dylib SIM_EXTRA_DYLIBS ios_sim_arm64 libGemmaModelConstraintProvider.dylib
collect_extra_dylib SIM_EXTRA_DYLIBS ios_sim_arm64 libLiteRtMetalAccelerator.dylib

# ---------------------------------------------------------------------------
# 5. Package as .framework bundles
# ---------------------------------------------------------------------------

HEADERS_DIR="$LITERT_LM_DIR/c"
BUNDLE_ID="com.google.CLiteRTLM"
FRAMEWORK_NAME="CLiteRTLM"
MIN_IOS="13.0"

package_framework() {
    local ARCH_NAME="$1"  # e.g. "ios-arm64"
    local DYLIB_PATH="$2"
    shift 2
    local EXTRA_DYLIBS=("$@")
    local FW_DIR="$WORK_DIR/$ARCH_NAME/$FRAMEWORK_NAME.framework"

    mkdir -p "$FW_DIR/Headers" "$FW_DIR/Modules"

    # Copy binary (rename to framework name)
    cp "$DYLIB_PATH" "$FW_DIR/$FRAMEWORK_NAME"

    # Fix install name
    install_name_tool -id "@rpath/$FRAMEWORK_NAME.framework/$FRAMEWORK_NAME" "$FW_DIR/$FRAMEWORK_NAME"

    # Copy extra dylibs if present. Linked companions use @loader_path; GPU
    # plugins are also preloaded from Swift because LiteRT discovers them with
    # dlopen at engine-creation time.
    for EXTRA_DYLIB in "${EXTRA_DYLIBS[@]}"; do
        cp "$EXTRA_DYLIB" "$FW_DIR/"
        install_name_tool \
            -change "@rpath/$(basename "$EXTRA_DYLIB")" \
            "@loader_path/$(basename "$EXTRA_DYLIB")" \
            "$FW_DIR/$FRAMEWORK_NAME" || true
    done

    # Copy headers
    cp "$HEADERS_DIR/engine.h" "$FW_DIR/Headers/"
    if [ -f "$HEADERS_DIR/litert_lm_logging.h" ]; then
        cp "$HEADERS_DIR/litert_lm_logging.h" "$FW_DIR/Headers/"
    fi

    # Create module map
    cat > "$FW_DIR/Modules/module.modulemap" << 'MODULEMAP'
framework module CLiteRTLM {
    header "engine.h"
    export *
}
MODULEMAP

    # Create Info.plist
    cat > "$FW_DIR/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$FRAMEWORK_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$FRAMEWORK_NAME</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>MinimumOSVersion</key>
    <string>$MIN_IOS</string>
</dict>
</plist>
PLIST

    # Sign nested code before the framework bundle so the framework's sealed
    # resources capture the final dylib hash.
    for EXTRA_DYLIB in "${EXTRA_DYLIBS[@]}"; do
        codesign --force --sign - "$FW_DIR/$(basename "$EXTRA_DYLIB")"
    done
    codesign --force --sign - "$FW_DIR/$FRAMEWORK_NAME"

    info "Packaged $ARCH_NAME framework at $FW_DIR"
}

info "Packaging device framework..."
package_framework "ios-arm64" "$DEVICE_DYLIB" "${DEVICE_EXTRA_DYLIBS[@]}"

info "Packaging simulator framework..."
package_framework "ios-arm64-simulator" "$SIM_DYLIB_COPY" "${SIM_EXTRA_DYLIBS[@]}"

# ---------------------------------------------------------------------------
# 6. Create xcframework
# ---------------------------------------------------------------------------

info "Creating xcframework..."

# Remove existing
rm -rf "$OUTPUT_DIR"

xcodebuild -create-xcframework \
    -framework "$WORK_DIR/ios-arm64/$FRAMEWORK_NAME.framework" \
    -framework "$WORK_DIR/ios-arm64-simulator/$FRAMEWORK_NAME.framework" \
    -output "$OUTPUT_DIR"

info "XCFramework created at: $OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 7. Verify
# ---------------------------------------------------------------------------

info "Verifying xcframework..."

for ARCH_DIR in "$OUTPUT_DIR"/ios-*/; do
    BINARY="$ARCH_DIR$FRAMEWORK_NAME.framework/$FRAMEWORK_NAME"
    if [ -f "$BINARY" ]; then
        ARCH_INFO=$(file "$BINARY" | grep -oE 'arm64|x86_64' | head -1)
        SIZE=$(du -h "$BINARY" | cut -f1)
        info "  $(basename "$ARCH_DIR"): $ARCH_INFO ($SIZE)"
    fi
done

TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
info "Total xcframework size: $TOTAL_SIZE"

info "Done! xcframework is ready at Frameworks/LiteRTLM.xcframework"
