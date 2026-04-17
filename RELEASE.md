# Release and Archival Workflow

This repository supports citable software releases, but the permanent identifier is minted outside the repository at release time.

## Recommended archival path

1. Update `CHANGELOG.md` and repository metadata files.
2. Create or update a GitHub release tag for the version being published.
3. Ensure the GitHub repository is connected to Zenodo.
4. Publish the GitHub release.
5. Let Zenodo archive the tagged release and mint a DOI.
6. Add the minted DOI back into:
   - `CITATION.cff`
   - the GitHub release notes
   - any manuscript `Code availability` statement

## Repository files that support release citation

- `CITATION.cff`
- `codemeta.json`
- `CHANGELOG.md`
- `.zenodo.json`

## Notes

- A permanent identifier is not embedded here in advance because it must correspond to a specific archived release.
- For unreleased development snapshots, cite the repository URL together with a commit SHA.
- ToolShed publication and GitHub release tagging should be kept synchronized when possible so Galaxy administrators can map a released wrapper set to an archived source snapshot.
