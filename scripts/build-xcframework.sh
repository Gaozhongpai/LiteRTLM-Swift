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
WORK_DIR="$(mktemp -d)"

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

info "Using LiteRT-LM source at: $LITERT_LM_DIR"

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

# Also grab the GemmaModelConstraintProvider dylib if present.
# Newer LiteRT-LM checkouts stage these in prebuilt/<platform>/ instead of
# bazel-bin/c/.
DEVICE_CONSTRAINT_DYLIB=""
SIM_CONSTRAINT_DYLIB=""

if [ -f "$LITERT_LM_DIR/prebuilt/ios_arm64/libGemmaModelConstraintProvider.dylib" ]; then
    DEVICE_CONSTRAINT_DYLIB="$LITERT_LM_DIR/prebuilt/ios_arm64/libGemmaModelConstraintProvider.dylib"
    info "Found device libGemmaModelConstraintProvider.dylib in prebuilt/ios_arm64"
elif [ -f "$LITERT_LM_DIR/bazel-bin/c/libGemmaModelConstraintProvider.dylib" ]; then
    DEVICE_CONSTRAINT_DYLIB="$LITERT_LM_DIR/bazel-bin/c/libGemmaModelConstraintProvider.dylib"
    info "Found device libGemmaModelConstraintProvider.dylib in bazel-bin/c"
fi

if [ -f "$LITERT_LM_DIR/prebuilt/ios_sim_arm64/libGemmaModelConstraintProvider.dylib" ]; then
    SIM_CONSTRAINT_DYLIB="$LITERT_LM_DIR/prebuilt/ios_sim_arm64/libGemmaModelConstraintProvider.dylib"
    info "Found simulator libGemmaModelConstraintProvider.dylib in prebuilt/ios_sim_arm64"
elif [ -f "$LITERT_LM_DIR/bazel-bin/c/libGemmaModelConstraintProvider.dylib" ]; then
    SIM_CONSTRAINT_DYLIB="$LITERT_LM_DIR/bazel-bin/c/libGemmaModelConstraintProvider.dylib"
    info "Found simulator libGemmaModelConstraintProvider.dylib in bazel-bin/c"
fi

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
    local EXTRA_DYLIB="${3:-}"
    local FW_DIR="$WORK_DIR/$ARCH_NAME/$FRAMEWORK_NAME.framework"

    mkdir -p "$FW_DIR/Headers" "$FW_DIR/Modules"

    # Copy binary (rename to framework name)
    cp "$DYLIB_PATH" "$FW_DIR/$FRAMEWORK_NAME"

    # Fix install name
    install_name_tool -id "@rpath/$FRAMEWORK_NAME.framework/$FRAMEWORK_NAME" "$FW_DIR/$FRAMEWORK_NAME"

    # Copy extra dylib if present
    if [ -n "$EXTRA_DYLIB" ] && [ -f "$EXTRA_DYLIB" ]; then
        cp "$EXTRA_DYLIB" "$FW_DIR/"
    fi

    # Copy headers
    cp "$HEADERS_DIR/engine.h" "$FW_DIR/Headers/"
    cp "$HEADERS_DIR/litert_lm_logging.h" "$FW_DIR/Headers/"

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
    if [ -n "$EXTRA_DYLIB" ] && [ -f "$FW_DIR/$(basename "$EXTRA_DYLIB")" ]; then
        codesign --force --sign - "$FW_DIR/$(basename "$EXTRA_DYLIB")"
    fi
    codesign --force --sign - "$FW_DIR/$FRAMEWORK_NAME"

    info "Packaged $ARCH_NAME framework at $FW_DIR"
}

info "Packaging device framework..."
package_framework "ios-arm64" "$DEVICE_DYLIB" "$DEVICE_CONSTRAINT_DYLIB"

info "Packaging simulator framework..."
package_framework "ios-arm64-simulator" "$SIM_DYLIB_COPY" "$SIM_CONSTRAINT_DYLIB"

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

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

rm -rf "$WORK_DIR"
info "Done! xcframework is ready at Frameworks/LiteRTLM.xcframework"
