# GitHub Actions workflows (pending)

These two workflows could not be pushed into `.github/workflows/` because the GitHub
connection lacks the `workflows` permission. To activate them, run from the repository root:

```bash
mkdir -p .github/workflows
git mv ci/github-workflows/build-apk.yml ci/github-workflows/tests.yml .github/workflows/
git commit -m "Enable CI workflows" && git push
```

- `tests.yml` runs the pytest suite on every push.
- `build-apk.yml` builds the Android APK (artifact `crypto-monitor-apk`).
