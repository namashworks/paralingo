# HTTP API

Use Paralingo from any language (JavaScript, Go, Java, C#...) over HTTP, protected by API keys.

## Set up

```bash
pip install "paralingo[api]"
```

Create a key. It is shown **once**; only a SHA-256 hash is stored in `keys.json`:

```bash
paralingo-keys --keys keys.json create --name support-bot --tenant acme --scopes profiles:read profiles:write audio:analyze
```

Start the server:

```bash
paralingo-api --root ./profiles --keys keys.json --port 8000
```

It listens on `127.0.0.1` (this machine only) by default. To serve other machines, add `--host 0.0.0.0`
and put it behind HTTPS (a reverse proxy such as Caddy or nginx). Interactive documentation for every
endpoint is at `http://127.0.0.1:8000/docs`.

## Authentication

Send the key in either header:

```
Authorization: Bearer spk_...
X-API-Key: spk_...
```

| Concept | Meaning |
|---|---|
| **Scopes** | `profiles:read` (summaries, guidance, sections, trends), `profiles:write` (sessions, observations, accent, consent), `audio:analyze` (submit turns), `admin` (all of these plus deleting a speaker) |
| **Tenants** | Each key belongs to one tenant, and each tenant's profiles are stored in their own folder. A key can never see another tenant's data, even with `admin`. Use one tenant per customer. |
| **Rate limit** | Per key, 120 requests per minute by default (`--rate` when creating the key) |

Manage keys:

```bash
paralingo-keys --keys keys.json list
```

```bash
paralingo-keys --keys keys.json revoke <key_id>
```

Revocation takes effect immediately, without a restart.

## Endpoints

| Method and path | Scope | Purpose |
|---|---|---|
| `GET /v1/health` | none | Liveness and version |
| `GET /v1/tools?format=anthropic\|openai\|mcp` | any | Tool definitions this key may call |
| `POST /v1/tools/{name}` | depends on tool | Run any [tool](tools-reference.md) with a JSON body |
| `GET /v1/speakers` | `profiles:read` | List speakers in your tenant |
| `GET /v1/speakers/{id}/summary` | `profiles:read` | Snapshot plus `prompt_context` |
| `GET /v1/speakers/{id}/guidance` | `profiles:read` | Current guidance and emotion |
| `POST /v1/speakers/{id}/sessions` | `profiles:write` | Start a session; body `{"channel", "context"}` (optional) |
| `POST /v1/speakers/{id}/sessions/{sid}/end` | `profiles:write` | Close a session and update the baseline |
| `POST /v1/speakers/{id}/utterances` | `audio:analyze` | Analyse one turn (multipart form, below) |
| `DELETE /v1/speakers/{id}` | `admin` | Erase a speaker's data |

## A call, end to end

```bash
KEY=spk_your_key_here
```

```bash
curl -s -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"channel": "phone"}' http://127.0.0.1:8000/v1/speakers/caller-42/sessions
```

Returns `{"session_id": "sess_..."}`. Then, for each user turn:

```bash
curl -s -H "Authorization: Bearer $KEY" -F session_id=sess_... -F transcript="um, I was charged twice" -F response_latency_s=0.8 -F audio=@turn.wav http://127.0.0.1:8000/v1/speakers/caller-42/utterances
```

Form fields for `/utterances` (at least one of `audio`, `transcript`, `word_timestamps`):

| Field | Type |
|---|---|
| `audio` | WAV file (PCM), up to 25 MB. It is analysed and deleted immediately, never stored. |
| `transcript` | Text |
| `word_timestamps` | JSON string: `[{"word": "hi", "start": 0.1, "end": 0.3}, ...]` |
| `external` | JSON string of other models' outputs, e.g. `{"asr_confidence": 0.93}` |
| `session_id`, `response_latency_s`, `t_start`, `t_end` | Optional |

At the end of the call:

```bash
curl -s -X POST -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/v1/speakers/caller-42/sessions/sess_.../end
```

Before the next call, fetch what the agent should know:

```bash
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/v1/speakers/caller-42/summary
```

## From JavaScript

```javascript
// sketch: Node 18+ or a browser backend; never expose API keys in front-end code
const base = "http://127.0.0.1:8000/v1";
const headers = { Authorization: `Bearer ${process.env.PARALINGO_API_KEY}` };

const form = new FormData();
form.append("transcript", "um, I was charged twice");
form.append("audio", new Blob([wavBytes], { type: "audio/wav" }), "turn.wav");
const r = await fetch(`${base}/speakers/caller-42/utterances`, { method: "POST", headers, body: form });
const { emotion, flags, guidance } = await r.json();
```

## Errors

| Status | Meaning |
|---|---|
| 400 | Bad input, e.g. unknown session, invalid JSON field, or `audio_path` sent to a tool (upload audio instead) |
| 401 | Missing, invalid or revoked API key |
| 403 | The key lacks the scope, or the speaker hasn't consented to voice analysis |
| 404 | Unknown speaker or tool |
| 409 | Conflicting concurrent edit |
| 413 | Audio larger than 25 MB |
| 415 | Audio isn't a WAV file |
| 429 | Rate limit exceeded; wait for the `Retry-After` seconds |
| 503 | The profile was locked by another writer for too long; retry |

Every response carries an `X-Powered-By: paralingo/<version>` header, and every write is recorded in the
profile's `audit_log` as `api:<key name>`.
