# Release procedures

For schema-v2 prebuilt installation variants, input staging, physical
qualification and assets-last publication, use the
[distribution guide](distribution/README.md). The variants are `app`, `ocr-cuda`,
`ocr-rocm`, `app-ocr-cuda` and `app-ocr-rocm`. App installations use an external
LLM endpoint and an explicit installed model. This update verifies the software
gates locally. Public distribution and physical accelerator qualification require
a separate release run. The app-image workflow below publishes the app image.

## Unreleased package

Durable 202 uploads/retries, source-preserving receipt review, exact money, explicit
Gold/v4 diagnostics, strict optional OCR, independent native OCR candidates and external-only LLM distribution are integrated in this package. The update also
includes account and view isolation and completion-only receipt recovery. Dedicated
OCR downloads and verifies its four pinned models when enabled. Its first prediction
uses the first submitted receipt. The source launcher supports upgrading the admitted
installation in place. Pending uploads bind their formatter endpoint, and migrated
unverified receipts can be reviewed using their current findings. Historical split/CPU results retain
their original meaning. See [upgrade prerequisites and accepted evidence](UPGRADE_AND_VERIFY.md).
No version/tag/release or deployment has been authorized by this documentation
update. In particular, publishing the app image does not qualify CUDA/ROCm OCR.

After maintainers deliberately adopt this workflow on the default branch, open **Actions → Release → Run workflow**, select the
repository's default branch, and choose `patch`, `minor`, or `major`. Merging a PR
or adding a label never publishes a release. Only users with repository write
access can dispatch workflows; protect the default branch and workflow changes
with the repository's usual review rules.

The workflow verifies the dispatch commit by calling this repository’s CI workflow in read-only
jobs (Python lint/unit tests, frontend verification, release-driver tests,
and backend/frontend E2E suites). A separate job reserves an annotated `vX.Y.Z` Git tag, builds the Dockerfile
for Linux AMD64, pushes `ghcr.io/<owner>/<repository>:X.Y.Z`, and creates a GitHub
release containing the image digest and commit SHA. Repository names are
lowercased for GHCR. The baseline is the greatest stable SemVer tag, or `0.0.0`
when none exists. A first patch is `0.0.1`; a first minor is `0.1.0`.

No Docker Hub account, personal access token, custom publishing secret, local
configuration, database, or deployment access is used. The only publishing
credential is GitHub's short-lived `GITHUB_TOKEN`, scoped to `contents: write`
and `packages: write` in the publishing job. Checkouts do not persist credentials;
checkout and Docker publishing actions are pinned to commit SHAs. The Docker build receives no GitHub API
token or secret build arguments. OCI metadata contains only the repository URL,
version and commit. The Dockerfile copies application sources and build outputs;
`.dockerignore` excludes local environment files, credentials, backups and Git data.
This reduces exposure; it cannot protect secrets deliberately committed into
application sources. Review workflow, Dockerfile and dependency changes as code.

## Repository setup

- Allow GitHub Actions and organization package creation for this repository.
- Permit the workflow token to create release tags, releases and GHCR packages;
  repository/organization rulesets may restrict these operations.
- For an existing package, grant this repository Actions write access.
- Check the package visibility after the first publication. Repository visibility
  does not guarantee the desired package visibility. Set the package public if
  unauthenticated pulls are intended. The workflow does not change visibility.

See GitHub's [container registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
and [manual workflow documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).

## Recovery and version identity

Each reservation records the workflow run ID, bump and exact dispatch SHA. A rerun
reuses that reservation, even after the default branch moves. **Rerun the original
failed workflow**, rather than starting a new dispatch, to recover a failed
publication. Do not delete reservation tags. Starting a new run intentionally
allocates a new version. Unpublished reservations remain part of version history.

Release runs are serialized without canceling a running release. GitHub's default
concurrency behavior retains only one pending run; a newer dispatch can replace a
pending one. Avoid queuing releases and rerun canceled requests when needed.
Atomic tag creation also prevents version collisions. If an image already exists,
its OCI source/revision/version annotations must match before it is reused; lookup
errors other than an explicit registry 404 abort publication. An existing matching
image is never rebuilt. Existing mismatched GitHub releases cause failure.

Git tags, registry pushes and GitHub releases are separate operations; a failed run
can leave a tag or image without a release. A GitHub release is created only after
successful image publication and digest validation. No mutable `latest`, major or
minor image aliases are published, and rerunning an older release does not mark
it as GitHub's latest release. Deployments should use the exact version or digest.
Publishing never restarts the application or changes its database.

## Reuse and validation

The workflow and `scripts/release.mjs` are repository-local and derive identity
from GitHub's context. To adopt them in another Tendstead repository, copy both,
provide an equivalent project verification command, and adjust the Dockerfile or
platform deliberately. There is no dependency on a separate automation repository.
The current platform is AMD64, matching this repository's existing build target.

`make release-test` exercises version increments, dispatch guards, concurrent
allocation, failed-publication ordering, reruns, API handling and registry reuse.
The CI checks job includes these release-driver tests. Release verification
reuses the complete CI workflow before publishing. Tests do not publish images or create tags remotely.
Live GHCR permissions and the first publication require a real manual release
from the default branch after merge.
