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
| `language` | Language your ASR detected, e.g. `ja-JP` (optional hint; see [languages](languages.md)) |
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

## Real-time (WebSocket)

Stream live audio and get [turn-taking events](realtime.md) back: end of turn, barge-in, backchannel,
plus a full analysis of each finished turn. Needs a key with scope `audio:analyze`, plus
`profiles:write` to save the call's latency statistics.

```
ws://127.0.0.1:8000/v1/realtime?speaker_id=caller-42&sample_rate=16000&encoding=pcm16&session_id=sess_...
```

| Query parameter | Default | Meaning |
|---|---|---|
| `speaker_id` | required | Whose profile to use and update |
| `sample_rate` | 16000 | 8000–48000 |
| `encoding` | `pcm16` | `pcm16` (16-bit little-endian), `mulaw` (G.711) or `float32` |
| `session_id` | none | Session to add analyses to; also where latency statistics are saved |
| `language` | none | e.g. `ja-JP`; otherwise detected from the text |
| `analyze` | `true` | Run the full Paralingo analysis on each finished turn |

**Authentication.** Send an `Authorization: Bearer` header, or, from a browser, make the first message
`{"type": "auth", "api_key": "spk_..."}`.

**You send:**

| Message | Meaning |
|---|---|
| binary frame | Audio, any size (e.g. 20 ms) |
| `{"type": "transcript", "text": "...", "final": false, "language": "ja-JP"}` | Partial or final ASR text, as it arrives |
| `{"type": "agent_speaking", "value": true}` | Your TTS started (`false` when it stops) |
| `{"type": "stop"}` | End of the call |

**You receive:**

| Message | Meaning |
|---|---|
| `{"type": "ready", ...}` | Connected; includes the speaker's calibrated `personal_pause_ms` |
| `{"type": "speech_start", "t": 1.23}` | The person started speaking |
| `{"type": "end_of_turn", "t": ..., "data": {...}}` | Finished: `transcript`, `language`, `probability`, `reasons`, `silence_ms`, `prosody`, `lexical` |
| `{"type": "barge_in", ...}` / `{"type": "backchannel", ...}` | Interruption (stop TTS) / "mm-hmm" (keep talking) |
| `{"type": "analysis", "data": {...}}` | Emotion, metrics, flags and guidance for the finished turn |
| `{"type": "summary", "data": {...}}` | On close: end-of-turn delay, response time, processing cost |
| `{"type": "error", "error": "..."}` | A problem; close codes 4401 (bad key), 4403 (scope), 4429 (rate limit), 4400 (bad parameters) |

See the [Azure guide](azure.md#browser-or-javascript-apps) for a browser example.

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
