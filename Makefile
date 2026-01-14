.PHONY: help check deps build release install docker-build docker-install test test-manual clean dev fmt lint rpm deb

DOCKER_IMAGE := voice-dictation-builder
DOCKER_OUTPUT := docker-output

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

check:  ## Check system dependencies
	@./scripts/check-deps.sh

deps:  ## Download Vosk models (2GB download!)
	@cd models && ./download-models.sh

build: docker-build  ## Build release with all features via Docker

release: docker-build  ## Build release with all features via Docker

docker-build:  ## Build release binary with all features using Docker
	@echo "Building with Docker (includes vosk + parakeet)..."
	@mkdir -p $(DOCKER_OUTPUT)
	docker build --target export -o $(DOCKER_OUTPUT) .
	@echo "Build complete. Output in $(DOCKER_OUTPUT)/"
	@ls -la $(DOCKER_OUTPUT)/

docker-install: docker-build  ## Build via Docker and install to ~/.local
	@echo "Installing to ~/.local/bin and ~/.local/lib..."
	@mkdir -p ~/.local/bin ~/.local/lib
	@pkill -x voice-dictation 2>/dev/null || true
	@sleep 1
	cp $(DOCKER_OUTPUT)/voice-dictation ~/.local/bin/
	cp $(DOCKER_OUTPUT)/lib/libvosk.so ~/.local/lib/
	@echo "Installing UI files to ~/.config/voice-dictation/ui/..."
	@mkdir -p ~/.config/voice-dictation/ui/examples
	cp slint-gui/ui/*.slint ~/.config/voice-dictation/ui/
	cp slint-gui/ui/examples/* ~/.config/voice-dictation/ui/examples/
	@echo "Updating library path..."
	@grep -q 'LD_LIBRARY_PATH.*\.local/lib' ~/.bashrc 2>/dev/null || \
		echo 'export LD_LIBRARY_PATH="$$HOME/.local/lib:$$LD_LIBRARY_PATH"' >> ~/.bashrc
	@echo "Installed! Run: source ~/.bashrc (or restart shell)"

install: docker-install  ## Build and install to ~/.local/bin (uses Docker)

test:  ## Run automated tests
	cargo test

test-manual:  ## Run manual test with mic
	./test_manual.sh

dev:  ## Quick rebuild and install for development
	cargo build && \
	cargo install --path . --root ~/.local --force

fmt:  ## Format code
	cargo fmt --all

lint:  ## Run clippy
	cargo clippy --all-targets -- -D warnings

clean:  ## Clean build artifacts and temp files
	cargo clean
	rm -f /tmp/voice-dictation*.sock /tmp/voice-dictation-state /tmp/dictation-*.log

uninstall:  ## Uninstall system
	./uninstall.sh

deb:  ## Build DEB package (requires dpkg-deb, takes 5-10 minutes)
	@command -v dpkg-deb >/dev/null 2>&1 || { echo "Error: dpkg-deb not found. Install with: sudo apt install dpkg-dev"; exit 1; }
	@echo "Note: This will take 5-10 minutes as it compiles the entire project..."
	./packaging/deb/build-deb.sh

rpm:  ## Build RPM package (requires rpm-build, takes 5-10 minutes)
	@command -v rpmbuild >/dev/null 2>&1 || { echo "Error: rpmbuild not found. Install with: sudo dnf install rpm-build"; exit 1; }
	@echo "Note: This will take 5-10 minutes as it compiles the entire project..."
	./packaging/rpm/build-rpm.sh

status:  ## Check if dictation is running
	@pgrep -f dictation-engine >/dev/null && echo "Running" || echo "Not running"

logs:  ## Tail live logs
	tail -f /tmp/dictation-engine.log
