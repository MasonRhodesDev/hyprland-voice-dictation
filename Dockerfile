# Multi-stage build for voice-dictation with all features
# Builds with vosk, parakeet, and whisper support

FROM fedora:42 AS builder

# Install build dependencies
RUN dnf install -y \
    gcc gcc-c++ \
    rust cargo \
    pkg-config \
    alsa-lib-devel \
    openssl-devel \
    clang-devel \
    cmake \
    git \
    wget \
    unzip \
    libxkbcommon-devel \
    wayland-devel \
    libX11-devel \
    vulkan-devel \
    mesa-libGL-devel \
    fontconfig-devel \
    google-carlito-fonts \
    pipewire-devel \
    && dnf clean all

# Download and install libvosk
WORKDIR /opt
RUN wget -q https://github.com/alphacep/vosk-api/releases/download/v0.3.45/vosk-linux-x86_64-0.3.45.zip \
    && unzip vosk-linux-x86_64-0.3.45.zip \
    && mv vosk-linux-x86_64-0.3.45 vosk \
    && rm vosk-linux-x86_64-0.3.45.zip

# Set up libvosk paths
ENV VOSK_PATH=/opt/vosk
ENV LD_LIBRARY_PATH=/opt/vosk:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/vosk:$LIBRARY_PATH
ENV PKG_CONFIG_PATH=/opt/vosk:$PKG_CONFIG_PATH

# Create pkg-config file for vosk
RUN echo 'prefix=/opt/vosk' > /opt/vosk/vosk.pc && \
    echo 'libdir=${prefix}' >> /opt/vosk/vosk.pc && \
    echo 'includedir=${prefix}' >> /opt/vosk/vosk.pc && \
    echo '' >> /opt/vosk/vosk.pc && \
    echo 'Name: vosk' >> /opt/vosk/vosk.pc && \
    echo 'Description: Vosk speech recognition library' >> /opt/vosk/vosk.pc && \
    echo 'Version: 0.3.45' >> /opt/vosk/vosk.pc && \
    echo 'Libs: -L${libdir} -lvosk' >> /opt/vosk/vosk.pc && \
    echo 'Cflags: -I${includedir}' >> /opt/vosk/vosk.pc

# Copy source code
WORKDIR /build
COPY . .

# Build with all features (vosk + parakeet + slint-gui + pipewire)
# Note: GPU feature requires CUDA which isn't in this image
RUN cargo build --release --features "vosk,parakeet,slint-gui,pipewire"

# Create output directory with binary and required libraries
RUN mkdir -p /output/lib \
    && cp target/release/voice-dictation /output/ \
    && cp /opt/vosk/libvosk.so /output/lib/

# Runtime stage (minimal image)
FROM fedora:42 AS runtime

# Install runtime dependencies only
RUN dnf install -y \
    alsa-lib \
    pipewire \
    && dnf clean all

# Copy binary and libraries
COPY --from=builder /output/voice-dictation /usr/local/bin/
COPY --from=builder /output/lib/libvosk.so /usr/local/lib/

# Update library cache
RUN ldconfig

ENTRYPOINT ["voice-dictation"]

# Export stage - just the artifacts
FROM scratch AS export

COPY --from=builder /output/ /
