.PHONY: help check deps build install test test-manual clean dev fmt lint rpm deb

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

check:  ## Check system dependencies
	@./scripts/check-deps.sh

deps:  ## Download Vosk models (2GB download!)
	@cd models && ./download-models.sh

build:  ## Build debug binaries
	cargo build

release:  ## Build release binaries
	cargo build --release

install: release  ## Build and install to ~/.local/bin
	./install.sh

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
	tail -f /tmp/dictation-engine.log /tmp/dictation-gui.log
