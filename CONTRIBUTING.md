# Contributing to OffGridAI

Thank you for your interest in contributing. This project is a reference implementation for a privacy-first, fully local AI cluster — contributions that improve reproducibility, hardware compatibility, or documentation are especially welcome.

## What we want

- **Hardware compatibility reports** — Does this work on other SBCs (Rock 5B, Orange Pi 5, etc.)? Open an issue with your hardware and any changes needed.
- **Alternative model support** — Instructions for running other GGUF models (Llama 3, Qwen, Gemma, etc.) with the same setup.
- **Bug fixes** — Especially cross-distro issues (Ubuntu vs Raspberry Pi OS vs Arch).
- **Documentation improvements** — Clarity, typos, better explanations.
- **Roadmap items** — PRs implementing vision pipeline (Hailo TAPPAS), voice (Piper TTS), or home automation integration.

## What we don't want

- Cloud dependencies — this project is explicitly local-first, no cloud APIs.
- Telemetry or data collection of any kind.
- Features that require internet access at runtime.

## How to contribute

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Test on real hardware if possible
5. Open a PR with a clear description of what changed and why

## Reporting issues

Use the issue templates:
- **Bug report** — for setup failures, service crashes, unexpected behavior
- **Hardware compatibility** — for new hardware combinations

## Code style

- Shell scripts: POSIX-compatible where possible, `set -e` always on
- Python: standard library preferred, minimal new dependencies
- No secrets, tokens, IPs, or usernames in committed code — use `.env.example` patterns

## License

By contributing, you agree your contributions will be licensed under the project's MIT License.
