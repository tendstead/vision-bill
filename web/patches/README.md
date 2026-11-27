# Shared shell localization

`@tendstead+ui+0.1.1.patch` adds optional reactive label overrides to AppShell,
including visible text and accessible names. English remains the default;
Vision Bill supplies labels from its existing English/German dictionaries.

`npm ci` applies this patch through `postinstall` and fails if it cannot apply.
The Dockerfile copies this directory before installing dependencies for the same
behavior in container builds. Keep the patch when updating the lockfile until a
published `@tendstead/ui` version provides equivalent localization support, then
remove the patch and patch-package together when adopting that version.
