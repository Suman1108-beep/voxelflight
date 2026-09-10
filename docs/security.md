# Security and deployment boundaries

- Public scene assets contain the deliberately shared demo. Private user videos, job directories and tokens are excluded from this repository.
- `public_security.py` verifies Firebase token signature, audience, issuer, authentication time and supported provider. Ownership uses the verified subject, never a client-supplied email or name.
- Private job listing, status, cancellation and downloads enforce ownership. Another user's job is not disclosed.
- Public uploads enforce declared and received length, queue admission and account limits. File names and outputs use server-controlled paths.
- The local development API is loopback-only. It must not be exposed as a substitute for the authenticated public API.
- Frontend Firebase project IDs, authorized public domains and API hostnames are configuration, not credentials. No OAuth client secret, service-account key or Firebase bearer token belongs in frontend code or GitHub.
- Google sign-in is enabled. GitHub repository publication is separate from GitHub sign-in; the latter remains disabled until its OAuth application is configured.
- CI has read-only repository permissions and no deployment secrets. Firebase publication remains an authenticated operator action.
- A temporary tunnel is not an uptime or high-availability guarantee. Use a stable, managed deployment for production.

Please report suspected security problems privately to the repository owner rather than opening a public issue containing tokens or private files. Do not upload credentials into an issue or a pull request.
