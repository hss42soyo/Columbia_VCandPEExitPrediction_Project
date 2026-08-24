# Data Handling

This code folder is the reproducibility surface. It should not contain
row-level licensed vendor data.

The paired data package is private and licensed. It contains copied Crunchbase,
Preqin-derived linkage, Common Crawl proxy, and frozen output artifacts required
for third-party review.

The Common Crawl artifacts are public-derived, but they are joined to licensed
company/domain surfaces. Treat the combined package as private unless a separate
public synthetic or pseudo-data pack is built.

No cleanup step in this package deletes source files. If a package needs to be
retired, move it explicitly to an obsolete location and record the transaction.
