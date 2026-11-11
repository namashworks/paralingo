# Troubleshooting

**Fillers and repetitions are always 0.**
Your ASR is probably cleaning the transcript. Many services drop "um", "uh" and repeated words unless you
enable a verbatim or disfluency option. Paralingo can only count what's in the transcript.

**`ConsentError: speaker has not consented to voice analysis`.**
That speaker's `voice_analysis` flag is `False`. Set it with `set_consent` only after the person agrees.
Adding a `health_indicator` observation also needs `health_indicators=True`.

**Pitch, voice quality and audio quality are `null`.**
No audio was passed (transcript-only turns), the turn was silent, or the audio was too short. Pass
`audio_path` or `samples` + `sample_rate`.

**`audio_quality` is `poor` and voice-quality numbers look odd.**
Background noise, clipping or a phone codec. Paralingo excludes poor-audio turns from the baseline and from
voice-strain checks. Phone audio (8 kHz) is supported; see
[phone audio](integration-guide.md#phone-audio-8-khz-µ-law).

**Emotion is always `neutral` or has low confidence.**
Normal for a new speaker: the personal baseline needs 5+ turns (`provisional`) and 3+ sessions with 15+
turns (`established`). Close sessions with `end_session` so the baseline updates. For stronger emotion
estimates, pass a dedicated speech-emotion model's output in `external={"emotion": {...}}`.

**`end_of_turn_silence_ms` or `recommended_speaking_rate_wpm` is `None`.**
Not enough data yet. These need a few turns with pauses or a transcript; keep your own defaults until then.

**`LockTimeout`.**
Another writer held the profile for more than 10 seconds. Skip that turn's analysis and continue the call.
If it keeps happening, check whether the `profiles` folder is on a synced or network drive; keep it on
local disk.

**`ProfileError: unknown session` or `session ... is closed`.**
The `session_id` doesn't exist or was already ended. Start a new session, or omit `session_id` to use the
speaker's active one.

**`wave.Error` or `unsupported sample width` when loading audio.**
`audio_path` must be a PCM WAV file (8/16/24/32-bit). For MP3, Opus or µ-law streams, decode first and pass
`samples` + `sample_rate` instead.

**HTTP API returns 401, 403 or 429.**
401: wrong or revoked key. 403: the key lacks the scope, or the speaker hasn't consented. 429: too many
requests; wait for `Retry-After`. See [HTTP API errors](http-api.md#errors).

**Health indicators never appear.**
They're opt-in: set `health_indicators=True`. They're evaluated at `end_session`, need 3+ turns and about
15 s of speech in the session, and most are judged against the person's baseline, so they appear only
after a few sessions.

Still stuck? [Open an issue](https://github.com/namashworks/paralingo/issues) with the Paralingo version,
Python version, and the error message.
