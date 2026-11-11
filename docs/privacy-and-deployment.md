# Privacy, consent and deployment

Paralingo stores voice-derived information about real people. These controls let you collect only what
people agreed to, and delete it on time.

## Consent flags

Set them with `eng.set_consent(speaker_id, ...)`, the `paralingo_set_consent` tool, or the CLI
(`paralingo consent caller-42 --health-indicators yes`).

| Flag | Default | Effect |
|---|---|---|
| `voice_analysis` | `True` | If `False`, `analyze_utterance` refuses with `ConsentError` and nothing is stored |
| `health_indicators` | `False` | Non-diagnostic speech-health signals are computed only when `True`. Turning it off deletes existing indicators and health observations. |
| `store_transcripts` | `True` | If `False`, only derived numbers are kept, never the words |
| `recording` | `False` | Records whether the person agreed to audio being kept (Paralingo itself never stores audio) |
| `retention_days` | `365` | Sessions and observations older than this are deleted automatically on the next write |
| `granted_via` | none | How consent was given, e.g. `"verbal-consent-recorded"` or `"web-form"` |

Emotion inference and speech-health signals may be regulated where you operate, for example the EU AI
Act's limits on emotion recognition in workplaces and schools, or biometric-data laws such as GDPR and
BIPA. Get explicit consent and check local law.

## Deleting a person's data

```python
from paralingo import SpeechProfileStore

SpeechProfileStore("profiles").delete("caller-42")   # removes the file; returns True if it existed
```

Over HTTP: `DELETE /v1/speakers/caller-42` with an `admin` key.

## Limiting file size

* Closed sessions beyond the most recent `max_detailed_sessions` (default 20) keep only their summary;
  their utterance details are dropped.
* Each session keeps at most `max_utterances_per_session` (default 300) turns.
* `audit_log` keeps 500 entries, `observations` 1,000, and `trends` 200 points per metric.

Both settings live in the profile's `settings` section. A stored turn is about 4.5 KB.

## Concurrency and storage

Any number of agents and processes **on one machine** can share a `profiles` folder:

1. A writer creates `<file>.lock` exclusively. Others wait, up to 10 s by default (`LockTimeout`
   after that). A lock older than 30 s is treated as stale and removed.
2. It reads the profile, applies the change, increases `revision`, and appends to `audit_log`.
3. It validates against the JSON Schema (when `jsonschema` is installed), writes a temporary file,
   and atomically replaces the real file.
4. If anything fails before the write, the file is unchanged.

To refuse writing over someone else's newer change, pass `expected_revision` to `store.edit()`; it raises
`ConcurrencyConflict` if the file moved on.

**Keep the folder on local disk.** Lock files aren't reliable on synced folders (OneDrive, Dropbox) or many
network drives. For several machines, run the [HTTP API](http-api.md) on one host as a shared service.

## Deployment options

| Option | When |
|---|---|
| **Embedded library** | Your voice agent runs in Python on one machine: import `paralingo` directly |
| **MCP server** (`paralingo-mcp`) | LLM agents and tools that speak MCP |
| **HTTP API** (`paralingo-api`) | Agents in other languages, several machines, or several customers (tenants) |

## Security checklist

* Never commit `keys.json`, profiles or recordings; the project's `.gitignore` already excludes them.
* Serve the HTTP API over HTTPS, and give each key the fewest scopes it needs.
* With the tools, set `audio_root` so a model can only point at WAV files inside one folder; the HTTP API
  refuses file paths entirely.
* Report vulnerabilities privately, as described in [SECURITY.md](../SECURITY.md).
