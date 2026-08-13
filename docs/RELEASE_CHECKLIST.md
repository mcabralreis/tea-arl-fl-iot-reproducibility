# GitHub and Zenodo release checklist

- [x] Review the curated files and remove any material that cannot be shared.
- [x] Confirm the licence decision: no reuse licence is granted and copyright
      applies by default.
- [x] Confirm author order, affiliations, and the article title in
      `CITATION.cff` and `.zenodo.json`.
- [ ] Add the accepted article DOI when available.
- [x] Commit and push the curated package while the repository remains private.
- [x] Make the GitHub repository public when the authors and journal permit it.
- [x] Enable the repository in the Zenodo-GitHub integration before creating
      the first GitHub release.
- [x] Create the Git tag and GitHub release `v1.0.0`.
- [x] Wait for Zenodo to archive the release and mint the version DOI.
- [x] Reserve/use the Zenodo concept DOI for references that should resolve to
      the latest version; use the version DOI for the submitted immutable
      package when appropriate.
- [x] Insert the DOI badge and DOI link into `README.md` and `CITATION.cff`.
- [ ] Replace the repository/DOI placeholder in the manuscript Data
      Availability Statement.
- [ ] Download the Zenodo archive once and verify its manifest and contents.
