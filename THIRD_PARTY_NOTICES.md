# Third-Party Notices

Deepmate is distributed under the MIT License. It depends on third-party
open-source packages whose licenses remain governed by their upstream projects.

## Python Dependencies

Runtime Python dependencies are declared in `pyproject.toml`. Dependency license
metadata is provided by each package distribution. Deepmate currently depends on
Textual for the TUI.

## Desktop Pet Frontend

The optional desktop pet frontend under `pet_ui/` uses Electron and npm
dependencies declared in `pet_ui/package.json` and locked in
`pet_ui/package-lock.json`. These packages include MIT, ISC, BSD-style, and
other permissive licenses.

If distributing a packaged Electron application or other binary bundle, include
the full dependency notices generated from the bundled dependency tree.

## Built-In Skills and Templates

Built-in `SKILL.md` files and templates shipped under `src/deepmate/` are part
of this repository unless otherwise noted.
