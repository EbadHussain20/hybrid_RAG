# Engineering Deployment Guide

**Department:** Engineering
**Version:** 4.3
**Last Updated:** 2026-01-20

## Environments

NimbusForge maintains three environments: `dev`, `staging`, and `production`. All changes must pass through staging before reaching production, except for documented hotfixes.

## Deployment Process

1. Open a pull request against `main`; at least one approving review is required.
2. CI runs unit tests, integration tests, and a security scan (Snyk).
3. Merges to `main` auto-deploy to `staging`.
4. Production deploys require a manual approval gate from the on-call lead and are limited to weekdays between 9 AM and 4 PM local time, except for Critical security patches.
5. Rollback is performed via the `deploy rollback <service>` CLI command, which reverts to the last known-good image within 5 minutes.

## On-Call

Engineers are on-call for one week at a time on a rotating schedule. On-call engineers must acknowledge PagerDuty alerts within 10 minutes for Sev1 incidents.

## Infrastructure

Production runs on Kubernetes across two regions for redundancy. Database backups run every 6 hours with a 30-day retention window.
