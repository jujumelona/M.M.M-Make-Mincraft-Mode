---
name: generate-audio
description: Generate or import Minecraft OGG audio and bind sounds.json, subtitles and SoundEvents.
schema_version: mmm/skill-v2
---

activate_when:
  - The approved proposal contains sound effects, ambience, music or UI sounds.

inputs:
  - approved audio requests
  - target Fabric project
  - optional reviewed existing OGG files

required_rag:
  - Minecraft 1.20.1 sound resource format
  - Fabric SoundEvent registration APIs
  - audio asset license and provenance

allowed_tools:
  - execute_complete_project
  - apply_source_patch
  - run_gradle_build
  - runtime_register_screenshot

output_schema:
  - OGG paths and hashes
  - sounds.json and subtitle keys
  - GeneratedSounds Java binding
  - playback gate receipts

validators:
  - bounded duration, frequency, volume and file size
  - OGG file existence and deterministic registration
  - no overwrite outside the approved project
  - client playback and loop review

retry_policy:
  max_attempts: null
  strategy: regenerate only the failed audio asset or binding
  stop_on_repeated_error_signature: true

approval_required:
  writes: true
  runtime: true
  read_only_research: false

forbidden_actions:
  - unlicensed audio ingestion
  - claiming playback success from file existence alone
  - hidden network uploads

exit_conditions:
  success:
    - Build and client playback gates pass for every requested sound.
  blocked:
    - OGG encoder or client runtime is unavailable.
  failed:
    - Audio or registration remains invalid after retries.
