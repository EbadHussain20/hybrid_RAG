# Engineering Coding Standards

**Department:** Engineering
**Version:** 2.0
**Last Updated:** 2025-09-05

## Language and Style

Backend services are written in Python (type-hinted, formatted with `black`) or Go. Frontend code uses TypeScript with strict mode enabled.

## Testing Requirements

All new features require unit test coverage of at least 80% for the changed lines. Integration tests are required for any change touching a public API contract.

## Code Review

Pull requests must include a description of the change and a link to the relevant ticket. Reviewers should respond within one business day. See the Deployment Guide for the full release pipeline once a PR is merged.

## Security in Code

Secrets must never be committed to version control; use the internal secrets manager. This mirrors the broader company Security Policy's prohibition on storing restricted data insecurely.
