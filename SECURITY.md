# Security Policy

Paralingo stores voice-derived personal data and, through the HTTP API, API keys. Please report
security problems privately.

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting**: the "Report a vulnerability" button under the
repository's Security tab. Do not open a public issue.

Please include the affected version, steps to reproduce, and the impact. You will get an
acknowledgement within 7 days. Fixes are released as soon as practical, and reporters are credited
unless they prefer not to be.

## Supported versions

| Version | Supported |
|---|---|
| 1.x | Yes |

## Deployment guidance

* Never commit API key files (`keys.json`) or profile data to Git; both are in `.gitignore`.
* Serve the HTTP API behind TLS (a reverse proxy such as Caddy or nginx).
* Keep the profiles directory on local disk with restricted permissions.
* Give each key the smallest set of scopes it needs, and one tenant per customer.
