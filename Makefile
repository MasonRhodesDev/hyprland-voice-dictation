# hyprland-voice-dictation — install automation
#
# Usage:
#   make build        # cargo build --release
#   make install      # build + install binary & service, reload, restart daemon
#   make uninstall    # remove binary & service (keeps your model & config)
#   make doctor       # report PATH/binary consistency problems
#   make model        # download the Parakeet model
#
# Override locations if needed:
#   make install BINDIR=/usr/local/bin UNITDIR=/etc/systemd/user

BINDIR  ?= $(HOME)/.local/bin
UNITDIR ?= $(HOME)/.config/systemd/user

BIN     := $(BINDIR)/voice-dictation
UNIT    := $(UNITDIR)/voice-dictation.service
RELEASE := target/release/voice-dictation

# Known-bad shadow location: `cargo install --path .` lands here, and ~/.cargo/bin
# typically precedes ~/.local/bin on PATH, so a stale copy here silently wins.
SHADOW  := $(HOME)/.cargo/bin/voice-dictation

.PHONY: all build install uninstall doctor model clean

all: build

build:
	cargo build --release

install: build
	install -Dm755 $(RELEASE) $(BIN)
	install -Dm644 packaging/systemd/voice-dictation.service $(UNIT)
	systemctl --user daemon-reload
	@# Remove any stale shadow binary that PATH might prefer over $(BIN).
	@if [ -e "$(SHADOW)" ] && [ "$(SHADOW)" != "$(BIN)" ]; then \
		echo ">> Removing stale shadow binary: $(SHADOW)"; \
		rm -f "$(SHADOW)"; \
	fi
	@# Restart the daemon so it runs the binary we just installed (else client/daemon skew).
	@if systemctl --user is-active --quiet voice-dictation; then \
		echo ">> Restarting daemon to match the installed binary"; \
		systemctl --user restart voice-dictation; \
	else \
		echo ">> Daemon not running. Start it with: systemctl --user enable --now voice-dictation"; \
	fi
	@echo ">> Installed $(BIN)"
	@$(MAKE) --no-print-directory doctor

uninstall:
	-systemctl --user disable --now voice-dictation
	rm -f $(BIN)
	rm -f $(UNIT)
	systemctl --user daemon-reload
	@echo ">> Uninstalled. Model & config under ~/.config/voice-dictation were kept."

# Pure-shell check — no daemon or model required. Flags the exact failure mode
# that makes the keybind silently do nothing: more than one binary on PATH.
doctor:
	@echo "Binary consistency check:"
	@found=$$(printf '%s\n' "$$PATH" | tr ':' '\n' | while read -r d; do \
		[ -n "$$d" ] && [ -x "$$d/voice-dictation" ] && readlink -f "$$d/voice-dictation"; \
	done | awk '!seen[$$0]++'); \
	count=$$(printf '%s\n' "$$found" | grep -c . || true); \
	if [ "$$count" -gt 1 ]; then \
		echo "  WARNING: $$count 'voice-dictation' binaries on PATH (the FIRST wins):"; \
		printf '%s\n' "$$found" | sed 's/^/    /'; \
		echo "  Keep only $(BIN); remove the others."; \
	elif [ "$$count" -eq 1 ]; then \
		echo "  OK: single binary -> $$found"; \
	else \
		echo "  No voice-dictation found on PATH. Is $(BINDIR) on your PATH?"; \
	fi

model:
	$(BIN) download-model

clean:
	cargo clean
