# Changelog

<!-- ## [0.3.0] - Unreleased -->


## [0.2.0] - 2026-09-22

### Added

- Added stateful high-level Prolog query API.
- Added optional ProgramMetadata sidecar support for synthesis classifications, usage, and weights.
- Added bounded sparse predicate-symbol lookup to the WAM compiler.


### Changed

- Replaced dense raw-symbol predicate lookup with bounded sparse storage.
- Replace the currents configuration sections with typed frontend, compiler, and runtime sections.
- Now uses native PlUnit fixtures for pytest.


### Removed

- Removed the legacy prefix-encoded answer API in favor of solution values.


### Fixed

- Fix invalid X-register allocation when compiling nested terms and negated goals.


## [0.1.0] - 2026-08-03

### Added

- Initial public release.

[0.2.0]: https://github.com/cognitive-modeling/WAMpy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cognitive-modeling/WAMpy/releases/tag/v0.1.0
