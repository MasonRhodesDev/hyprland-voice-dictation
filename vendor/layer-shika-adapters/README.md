# layer-shika

<div align="center">
  <h1 style="font-size: 5em;">🦌</h1>
  <p><i>"A cute layer of abstraction where Slint UIs grow antlers and become cute Wayland shells."</i></p>
  <p><b><a href="https://codeberg.org/waydeer/layer-shika">Main repo</a> | <a href="https://git.dren.dog/waydeer/layer-shika">Mirror</a> | <a href="https://github.com/waydeerwm/layer-shika">Temp mirror (github)</a></b></p>
</div>

Oh deer! 🦌 You've stumbled upon `layer-shika`, a Rust library providing Wayland layer shell integration with Slint UI. Create overlay windows, status bars, panels, popups, and more, that integrate seamlessly with Wayland compositors.

> [!CAUTION]
> This library is currently in early development and is not yet ready for production use. API may change before 1.0.

> [!NOTE]
> **Coming soon:** A complete Wayland shell built on top of layer-shika! Stay tuned for updates.

## Features

- **Slint Integration**: Runtime `.slint` file compilation via slint-interpreter or compile-time code generation. Support via pre-compiled is planned.
- **Multi-Surface Support**: Create multiple independent layer shell windows, each with its own configuration and lifecycle
- **Flexible Configuration**: Both fluent builder API and declarative configuration support
- **Comprehensive Popup System**: Full xdg-popup protocol implementation with multiple positioning modes, grab support, and content-based sizing (rework in progress)
- **Multi-Output Support**: Per-monitor component instances with flexible output policies (primary only, all outputs, specific outputs)
- **Event Loop Integration**: Custom event sources (timers, channels, file descriptors) via calloop integration
- **Clean-like Architecture**: Organized as a Cargo workspace with clear separation of concerns (domain, adapters, composition)
- **HiDPI Support**: Configurable scale factors for high-resolution displays

## Architecture

layer-shika is organized as a **Cargo workspace** with three crates:

- **domain** ([crates/domain/](crates/domain/)): Core domain models, value objects, and port trait definitions. No framework dependencies.
- **adapters** ([crates/adapters/](crates/adapters/)): Concrete implementations for Wayland (smithay-client-toolkit), rendering (femtovg + EGL), and platform integration.
- **composition** ([crates/composition/](crates/composition/)): Public API layer providing Shell-based API, builder patterns, and system integration.

This clean-like architecture enables flexibility, and clear dependency boundaries (composition → adapters → domain).

## Current Status

**What's Working:**

- Multi-surface shell with builder and declarative configuration APIs
- Wayland layer shell protocol (wlr-layer-shell) via smithay-client-toolkit
- Full xdg-popup protocol support with flexible positioning and sizing
- EGL context management with femtovg renderer integration
- Multi-output detection, tracking, and per-output component instances
- Event handling framework with ShellEventContext and ShellControl
- Custom event loop integration (timers, channels, file descriptors)
- HiDPI scale factor support

> [!WARNING]
> **Known Limitations:**
>
> - Documentation is work in progress
> - Some edge cases may not be fully tested
> - It's recommended to wait for a stable release before using this library in production projects
>
> However, if you want to experiment with the current version, feel free to do so! Development is trying to be as fast as a running skippy deer!

## Quick Start

Check out the [examples/](examples/) directory for comprehensive demonstrations.
Each example includes detailed documentation and can be run with:

```bash
cargo run -p simple-bar
cargo run -p multi-surface
cargo run -p declarative-config
...
...
```

See the [examples README](examples/README.md) for detailed usage instructions and patterns.

## Usage

> [!IMPORTANT]
> If you want to use it now, use this repo as a dependency instead of crates.io versions.

## First Stable Release

The aim is to have the first stable release by the end of 2025, with a focus on core functionality and API stability.

Stay tuned!

## Contributing

> [!TIP]
> As the library is in a very early stage, it's recommended to open an issue to discuss ideas or proposed changes before submitting contributions. The project doesn't bite, it's not that kind of deer!
