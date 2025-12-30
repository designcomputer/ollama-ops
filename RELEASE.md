# Release instructions

Steps to publish an initial alpha release (local):

1. Commit your changes:

```bash
git add .
git commit -m "chore: prepare alpha release 0.1.0a1"
```

2. Create and push a tag:

```bash
git tag -a v0.1.0a1 -m "Initial alpha release"
git push origin --tags
```

3. The GitHub Actions workflow `.github/workflows/release.yml` will run tests and create a prerelease automatically when the tag is pushed.

4. Optionally, create the release manually with GitHub CLI:

```bash
gh release create v0.1.0a1 --title "v0.1.0a1" --notes-file CHANGELOG.md --prerelease
```

Notes:
- Ensure `gh` is authenticated (`gh auth login`) if using the GitHub CLI.
- CI uses `python -m unittest -q` so no extra test dependency is required.
