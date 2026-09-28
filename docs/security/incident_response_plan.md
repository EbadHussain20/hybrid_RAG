# Incident Response Plan

**Department:** Security
**Version:** 1.4
**Last Updated:** 2025-11-02

## Purpose

This document defines how NimbusForge responds to confirmed or suspected security incidents affecting company systems, data, or customers.

## Severity Levels

- **Critical**: Active data breach, ransomware, or customer data exposure. CISO and Legal must be notified within 15 minutes of confirmation.
- **High**: Attempted intrusion or malware contained to a single host. SOC lead notified within 1 hour.
- **Medium**: Policy violations or suspicious activity without confirmed compromise.
- **Low**: Informational reports, false positives.

## Response Workflow

1. **Detection** — Alert raised via SIEM, employee report, or automated scanner.
2. **Triage** — SOC analyst assigns severity within 30 minutes.
3. **Containment** — Affected systems isolated from the network.
4. **Eradication** — Root cause removed (malware deleted, credentials rotated).
5. **Recovery** — Systems restored from clean backups and monitored for 72 hours.
6. **Post-Incident Review** — A retrospective is held within 5 business days for High and Critical incidents.

## Communication

External communication regarding a Critical incident must be approved by Legal and the Head of Communications before release. Employees must not discuss incidents on public channels.

## Related Documents

See the Employee Security Policy for the initial reporting steps employees must take.
