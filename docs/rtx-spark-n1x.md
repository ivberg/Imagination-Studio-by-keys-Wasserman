# Imagination Studio on an NVIDIA RTX Spark N1X laptop

The RTX Spark N1X (Windows 11 on Arm) has the same GPU architecture as the DGX Spark's GB10: Blackwell, sm_121,
6144 CUDA cores. The Studio runs on it **natively on Windows**: the DGX Spark's render path (ComfyUI, the Studio's
graphs and TensorFold's engines) on Windows' own NVIDIA driver. There is no WSL version: the WSL2 install was removed
on 2026-11-06. The dated measurements below that compare with WSL are kept as they were taken.
What differs from a DGX Spark is the laptop's memory split and its power budget.

## Ref2VA VSA on a smaller memory partition

This is a **separately selected backend**, not a lower memory limit for supported H3. The normal H3
`MIN_POOL_MB` policy remains 90,000 MiB. Installed RAM, Windows commit headroom and dedicated GPU memory are
different budgets; they are never added together to make a render fit.

With the native CUDA/ComfyUI core and the recipe's weights already installed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-windows-vsa.ps1
```

This installs only Kablex's pinned Apache-2.0 custom node, retaining its licence. It does not download weights,
install packages or build H3. The required filenames and folders are in [the VSA recipe](../app/windows_video_vsa.py):
INT8 ConvRot Ref2VA, NVFP4/AWQ Qwen3VL-32B, FP16 video VAE, FP32 audio VAE, the Ref2VA four-step Turbo adapter
and the FastH3 VSA gate. MiniMax H3's Community License and territory restrictions still apply.

Restart the **real Studio**, then open **Video -> Ref2VA VSA**, or choose that backend in Create.
Both backends use the same five Create steps: shot, picture/reference, words and sound, prompt review, then render.
The shared take sheet shows VSA's own fixed recipe, length, seed and reference status, without supported H3's
recipe shortcuts or estimates; its Roll button uses the same guarded VSA action as the Render step.
Scene, action, camera, dialogue, soundscape, music, seed and the reviewed prompt are shared when switching backends;
review or explicitly rebuild the prompt after switching to change its frame/reference wording.
VSA keeps its own reference picture, clip length and sequence settings; supported H3's frames, references,
LoRAs, canvas and passes remain separate and are not sent to this recipe. In step 2, upload a picture or pick a
still/scout from the Library. **Make scouts in Image** opens the normal image form without queuing a render;
**animate this**, Home's **animate** and the Library's **use** hand the chosen picture to your selected video backend.
Making pictures keeps that backend selection. Supported H3 still treats its chosen picture as a first frame;
VSA treats it only as a reference. Explicit first/last-frame and audio-reference actions stay on supported H3.

An oversized VSA picture is not silently resized or sent to the queue. **Create resized reference copy** explicitly
makes a PNG in uploads, preserving the aspect ratio within 512x288, without cropping, upscaling or changing the
original. The CPU-only preparation endpoint, `POST /api/video-vsa/reference`, inspects `{"path":"..."}` without
writing; adding `"resize":true` makes the copy with a 30-second tool timeout. Neither action runs a model.
The Render step stays disabled until the selected picture has been checked and fits the recipe.
Finished VSA clips/sequences use the normal Editor/download actions, can open their Library processing actions,
and can be chosen for a soundtrack in Music. Those existing features retain their own prerequisites and guards.
In Production/Slate, a shot can explicitly choose **VSA reference** under **Roll camera**.
Its separate Library path is saved on the shot; prepare an oversized picture in Create first.
The film must use 480p and 16:9. Duration (including music-bar slots) keeps the model-valid frame count;
the quality preset becomes the fixed four-pass recipe. Extra LoRAs, Drama 3 voice references and previs clips
are refused, not dropped. VSA takes stay attached to the film with their backend, seed and actual recipe,
and use the same take review and Editor delivery. Existing text, first-frame, previous-shot and cast-still
modes keep supported H3; none are silently converted to VSA.
Importing a finished clip as a Slate take is also available with Windows qualification enabled: it copies media,
not a model render. The existing Studio-folder, hidden-file and remote-media restrictions still apply.
The desktop supervisor uses the UI environment's `pythonw.exe` to avoid a venv console redirector window;
the app's PowerShell entry point retains console Python and the supervisor's hidden process ownership.
The simple Production board likewise offers **a VSA reference** in each segment's **Opens on**.
It saves a separate picture path, requires the 864x480 output preset and refuses project LoRAs. The normal
segment queue and Editor handoff retain the resulting clip; text, image and previous-segment modes are unchanged.
Only finished segments can be sent to the Editor; queued/running segments cannot be submitted twice, and
cancelled segments are eligible for the board's batch retry. Board and Slate handoffs measure the actual
saved clip duration, rather than the requested shot length; the board overview shows the model-valid
frame count and seconds. An unreadable clip or failed save leaves the
existing timeline unchanged and reports the error.
VSA uses **Machine -> Feature checks -> Video -> Ref2VA VSA**, alongside the other restored diagnostics.
Create's **VSA feature checks** link opens that card without running anything. The card offers the basic
22-frame check, any selected model-valid length, the automatic length sweep, saved length history and
normal queue cancellation. **Open VSA** selects the VSA backend in Create, not ordinary H3.
Diagnostics and individual checks remain available when qualification gating/badges are off.
Qualification behavior and memory guards are unchanged; technical passes do not certify perceptual
audio quality or absence of unwanted speech. The API equivalent of the basic check is:

```powershell
Invoke-RestMethod http://127.0.0.1:7870/api/feature-checks/check -Method Post `
  -ContentType application/json -Body '{"feature":"video-vsa"}'
```

The first check uses the bundled baker picture, resized in that job to 512x288. It generates **864x480,
22 frames at 24 fps (0.917 seconds), four Euler passes, sparsity 0.75**, with video/audio sigma shifts 12/3.
Text, reference and audio tokens remain dense. Only generated-video attention is sparse.
Every render is cold: gate transplantation mutates model modules, so a patched backend is never reused.
Cancellation closes the queue-owned Windows process tree, including the native backend and observer.
For sequences, the queue finalizes the interrupted report as cancelled/failed and publishes already completed
parts alongside it. An interrupted part is not called successful, and completed parts are not overwritten.

The upstream VSA patch currently matches the installed pin. Its packed-layout code retains dense text, audio
and reference prefixes, and reads both keyframes and references; that makes more references plausible, not
qualified on this laptop. Its supplied API example has one picture at 1344x768, 124 frames, and the same four-pass
Turbo adapter. It does not demonstrate extra style LoRAs or multiple-picture/audio/video reference workflows.
Another canvas, extra references and extra LoRAs each need separate correctness and host/commit/VRAM measurements
before enabling them here. The upstream reference-video attention lifetime fix is already in this pin;
there is no newer upstream patch to install as a substitute for those measurements.

No successful local full-model render means **unverified**, even if synthetic adapter or CUDA primitive tests
passed. A pass saves actual host, commitment and WDDM observations plus model checksums and runtime provenance.
With qualification enabled, only the basic 22-frame check is required for the queue's `video-vsa` kind
(`image`, `prompt`, `seed`, optional `frames`). All model-valid lengths are selectable afterward, including
unmeasured ones. Use one Studio picture at most 512x288, named `<Picture 1>` in your prompt. Ordinary H3,
other canvases, additional LoRAs, reference video/audio and worker placement remain outside this recipe.
With qualification disabled there is no evidence gate or qualification UI. At 128 GiB installed memory
and above it is disabled by default; opting in restores the basic gate and diagnostic panel. Installation,
recipe limits and all live memory guards apply in either mode.

The native dynamic-VRAM loader and handle-owned process tree remain in use. The independent observer and server
guard fail closed on missing/stale Windows/WDDM readings. Host and commit floors are retained, with a 2 GiB
dedicated GPU reserve. The VSA dynamic loader additionally requests 3 GiB of unused VRAM headroom
(`--vram-headroom 3`); this constrains loading rather than weakening the guard. Future admission uses matching
measured **commit** growth plus a 1 GiB error margin, not a combined RAM/VRAM allowance.
Historical host and GPU resident growth are saved as **advisory predictions**, including their error margins:
Windows can reclaim file-backed pages and page other working sets, while DynamicVRAM adapts weight residency.
A prior drop in available physical RAM or resident GPU peak is not a fixed nonpageable allocation requirement.
Short resident predictions do not block admission if measured commit capacity and live floors pass.
The same live 2 GiB host/GPU reserves (or higher configured host floor), 4 GiB commit floor, 3 GiB loader headroom
and independent physical/commit/WDDM supervision remain enforced. Paging is not guaranteed to prevent pressure;
an OS low-memory signal or breached live floor still stops the owned render.
An unmeasured length has an unknown peak: it is admitted on live floors and independently supervised,
not assigned the shorter clip's memory profile. A memory refusal is a blocker, never a reason to relax the floors.
The saved observations establish resource safety for this short recipe, **not visual fidelity or a laptop
throughput claim**. Neither the author's RTX 4090 benchmark nor synthetic kernel results qualify larger clips.

Fresh Production UI renders passed after commit capacity was freed: a 73-frame Board clip (3.042 seconds)
and a 124-frame Slate dialogue clip (5.167 seconds), both 864x480 with sound. Reload preserved the Board's
seed zero and the Slate take's VSA backend/recipe; both Editor handoffs saved the probed media duration.
Observed host/GPU/commit minima were 5749/3497/13180 MiB for the Board clip and 6421/3504/12797 MiB for
Slate, with no refused observer samples. Both decoded audio tracks had finite samples. Listening review
found the Slate English dialogue good, but heard indistinct, foreign-language-like speech in the Board
clip despite its ambient-only soundscape and closed-mouth instruction. That clip failed the no-dialogue
audio-quality check. These are one workflow/resource check each, not new qualification receipts or
verification of precise lip-sync.

A seed-only repeat batch also completed through the Studio queue: the same 73-frame ambient-only prompt
at seeds 1 and 2, and the same 124-frame English-dialogue prompt at seed 2. Within each case the compiled
prompt was reused byte-for-byte; only the seed changed. All three produced valid 864x480 clips with sound
and passed the observer guards. Listening review heard unwanted, partly English-like/indistinct speech
and no ambience in both ambient-only repeats. Including the original seed 0, the ambient-only speech
constraint failed at all three tested seeds. Both explicit-dialogue clips (seeds 1 and 2) had clear English.
These small counts describe this one baker scene and recipe, not general success rates; exact wording
and precise lip-sync were not verified. This establishes repeatable prompt-adherence trouble here,
not a quality regression against dense H3, which has not had a matched comparison.

A prompt-path audit reconstructed the original quiet prompt through `/api/compose` and matched its recorded
SHA-256. It contains closed-mouth and ambient-only instructions, without dialogue or a voice request. The
checked VSA path forwards that text unchanged; the installed MiniMax tokenizer adds picture labels and vision
blocks, not a chat or speech template. There is no audio reference in this recipe, and audio finishing uses the
same job's decoded soundtrack, not the Voice service. This is hash matching and source tracing, not a retained
historical conditioning capture. The action does say "a small greeting": a possible implicit speaking cue,
not an established cause.

A subsequent fixed-seed A/B completed through the Studio queue at seed 1 and 73 frames. A reused the exact
original prompt; B changed only `overall_soundscape` to concrete oven-fan, baking-tray and cloth sounds.
The picture, greeting action, closed-mouth instruction and rendering recipe were unchanged. Both retained
their guarded measurements and media. Listening review heard unwanted dialogue in both, and B lacked the
requested oven-fan and tray sounds. The soundscape-only change did not resolve this case. The greeting action
remains a possible confound; neither technical completion nor this A/B establishes why the speech occurs.

A further controlled clip removed only "in a small greeting" from A's action, retaining its original soundscape,
picture, seed 1 and 73-frame recipe. It completed with 658 observer samples and no refused sample; minima were
2480 MiB host available, 3448 MiB dedicated GPU free and 10974 MiB commit headroom. Listening review still heard
unwanted, unintelligible speech. Removing the greeting wording did not resolve this audio-adherence failure;
that wording alone does not explain it. The model/recipe/conditioning cause remains unestablished.

Native VSA renders retain `video.workflow.json` and `video.conditioning.json` beside the clip for diagnosis.
The Studio's pass-through audit node records the prompt hash, actual tokenizer IDs/reference types,
conditioning and latent shapes, model sampling shifts, and the video sigma entering each forward. Audio
sigmas/times are derived from those observed inputs with ComfyUI's own MiniMax shift helper, not read back
from an internal per-token timestep tensor. It does not change the tokens, conditioning, model weights,
sampler or returned tensors. These records are diagnostics, not audio-quality or qualification passes.

Two 124-frame (5.167-second), seed-1 controls completed with that runtime audit: the unchanged neutral-action
baker prompt, then an empty bakery with a place-only visual description and the same ambient soundscape.
Both had matching prompt hashes, one image reference, no `<d>`/`</d>` tokenizer IDs and four forwards.
Observed video sigmas were approximately `[1, 0.972973, 0.923077, 0.8]`; the derived audio sigmas were
`[1, 0.9, 0.75, 0.5]`, matching the published four-pass 12/3-shift recipe. The model's audio scale was 4.
No conditioning-input or schedule mismatch was identified; this does not verify the quantized encoder's
semantic fidelity or establish audio quality. Listening review again heard foreign-language-like speech from
the five-second baker (language not verified). The user clarified that the empty bakery had energetic,
advertisement-like ambience with no voices. That case met the no-voices goal, though not the requested
soft room ambience. Longer duration did not resolve the baker's unwanted speech. The empty bakery uses a new
reference with different composition/aspect ratio, so it is a scene diagnostic, not a person-only causal test.
The first submission was refused while the completed scout's owned backend was warm; the backend was
explicitly unloaded through the Studio before retrying. Memory floors were not changed.

A subsequent simplified-prompt control kept the baker reference, seed 1 and 124-frame recipe,
replacing the subject/summary/retention sections with a short visual paragraph and retaining the
same action, closed-mouth instruction and soundscape. Windows signalled low physical memory during
execution; the watchdog stopped the owned job and no clip was produced. This is an inconclusive
prompt comparison, not an audio failure or pass. No automatic retry or guard relaxation was made.
A user-authorized identical retry subsequently completed at 124 frames (5.167 seconds).
Its saved audit matched the submitted prompt hash, image-only reference and four-pass 12/3-shift
recipe. Listening review still heard the baker speaking, now French-like to the listener
(language not verified). Simplifying the prompt did not resolve unwanted speech in this case.

A local 175-frame single clip (7.292 seconds) completed under these guards; its observed minima were
3063 MiB host available, 3448 MiB dedicated GPU free and 5392 MiB commit headroom. These are one run,
not new admission floors or a qualification receipt. A subsequent three-part, 14-second attempt stopped
before part 2 loaded: commit headroom was 37294 MiB against 37312 MiB required by the measured prediction
and reserve. Part 1 remained available. Cancellation between parts also retained the completed clip
and finalized the report while the owned backend stopped.
The renderer's final preflight now uses the same bounded recovery wait as the sequence supervisor:
Python startup can consume a narrow margin after the earlier admission. It requires two fresh fitting
samples within 60 seconds before starting ComfyUI; persistent contention or any breached live floor
still refuses the part. The measured prediction, error margin and reserves are unchanged.
A subsequent three-part retry completed in 596 seconds: 336 decoded frames, exactly 14 seconds, 864x480
and sound. Its first renderer initially had 37199 MiB commit headroom, 113 MiB below the same requirement;
the bounded wait recovered before rendering. Across its three model parts, observed minima were
4185 MiB host available, 3440 MiB dedicated GPU free and 8228 MiB commit headroom. The backend drained
afterward. This is one successful local sequence, not a guarantee of fit or seamless continuity.

Visual sampling of the earlier eight-second chain showed a close-up-to-wide jump at its join. Both that
clip and the 175-frame clip moved the camera despite a locked-camera prompt. Treat these sequences as
potentially visible cuts, not seamless extensions. Decoded audio had finite samples and stayed below
full scale in the sampled chain, but this is not a listening test or proof of voice/sound continuity.

### Incremental frame calibration

The length chooser exposes **all 21 model-valid counts from 22 to 362 frames** (0.917 to 15.083 seconds),
at the same 864x480 canvas, four passes and sparsity. Unmeasured lengths are selectable for your own explicit
checks, not labelled measured. When qualification is enabled, first pass the 22-frame baseline, then select
any valid length and render. **Check the N-frame recipe** is optional: each length can have independent
evidence and measured admission, but missing longer-length evidence never blocks selection or rendering.

For incremental testing, start with **39, 56, then 73 frames**. Wait for each job to finish successfully
before queuing the next, and stop on any failure or memory refusal.
All existing host, commit and dedicated GPU floors remain unchanged.

```powershell
Invoke-RestMethod http://127.0.0.1:7870/api/feature-checks/check -Method Post `
  -ContentType application/json -Body '{"feature":"video-vsa","frames":39}'
```

Repeat with 56, then 73 only after the previous check passes. These durations are exactly frames / 24:
1.625, 2.333 and 3.042 seconds. Reports and memory observations are saved separately by frame count,
using verified model checksums from the baseline rather than hashing unchanged weights again.
Neither a successful longer calibration nor a failed attempt replaces another length's production receipt.
Normal renders can specify any valid count, for example `frames: 39`; omission keeps the 22-frame default.
The selectable maximum is the model's limit, **not a claim that this laptop can render it**.

### Automatic clip-length qualification on this machine

Use **Qualify clip lengths automatically** in the VSA panel, or:

```powershell
Invoke-RestMethod http://127.0.0.1:7870/api/feature-checks/check -Method Post `
  -ContentType application/json -Body '{"feature":"video-vsa","sweep":true}'
```

The Studio checks the 21 model-valid counts in increasing order, through **362 frames / 24 = 15.083 seconds**.
Matching local passes are reused with their original dates; unverified counts get cold, independently supervised
renders with generated clip audio. A missing baseline is checked first. It stops at the first memory refusal,
telemetry problem, render error, timeout or cancellation, never intentionally at OOM and never by lowering a floor.
The sweep has a two-hour total budget plus cleanup; each render keeps its existing 30-minute upper bound.

Qualification is tied to this machine's actual hardware partition, recipe, weights and execution contract, not
just installed RAM. A 128 GB install can use this VSA model without qualifying ordinary H3 or
obtaining qualification evidence when qualification is disabled. Host, commitment and dedicated GPU
budgets remain separate. This does **not** qualify separate sound-effects/Foley models or concurrent model residency.

When qualification is enabled, the panel shows the **highest successful matching clip and its original date**,
including supervised production renders as well as feature checks. Production successes are diagnostics,
not fabricated qualification receipts. It also shows how many counts have separate feature checks and the
first stopped candidate with its reason and sweep date. A higher previous pass remains valid if a new sweep stops
at a lower unverified count; neither the highest success nor a gap is a selectable-length cap.
A stop under current contention, missing telemetry
or a model error is **not the machine's absolute memory limit**. If every count passes, the model's 15.083-second
single-clip bound still applies. Current progress and partial results persist in the job's qualification report;
individual successful receipts survive failure or cancellation. No automatic retry follows a stop.

### VSA sequences up to one minute

Select **Sequence up to 60 seconds** in the VSA backend. With qualification enabled, the chooser initially selects
the longest separately checked per-shot length; all other valid lengths remain selectable. With qualification
disabled it keeps your selected length. Enter a whole-number duration from **1 to 60 seconds**.
This is a sequence of short shots, **not one continuous model render**. Picture and sound joins may be visible.
Longer sequences and seamless continuity have not been qualified on this laptop; individual short-clip passes
do not establish their visual quality.

The normal Studio queue runs every part serially, with the mode-dependent basic gate, fresh memory admission,
unchanged host/commit/dedicated floors, and a cold owned backend. The first shot uses your reference picture;
later ones use a resized copy of the preceding shot's last frame. That picture is a reference, **not a guaranteed
first-frame anchor**. The prompt stays unchanged; seeds increment per shot, wrapping at the 32-bit limit.
No duplicate initial frame is discarded: Ref2VA does not guarantee one.

The supervisor inherits the app's private session identity so a between-part release really drains its owned
backend. Before each part it waits up to **60 seconds** for memory recovery, requiring **two consecutive fresh
samples** to satisfy measured commit admission and all live floors. Missing/stale telemetry or a live floor violation fails closed;
persistent contention stops the chain with the last reason rather than waiting indefinitely. Recovery is included
in the two-hour sequence budget and its initial/final readings are saved with the part's observations.
The backend is also drained before CPU assembly. No foreign application is stopped or memory floor lowered.

The API uses the existing `video-vsa` job with `image`, `prompt`, `seed`, `frames` (per shot) and
`sequence_seconds` (1-60). For example, 60 seconds at 73 frames per shot needs 20 full model clips.
CPU-only ffmpeg joins them and trims the result to exactly 1440 video frames at 24 fps with sound.
The assembly is independently memory-supervised and all external tools have deadlines. No render bypasses the queue.

There is a **two-hour sequence budget**, plus owned-process cleanup time, and the existing per-part deadline.
Cold loads can dominate the runtime. A memory refusal, failed part, timeout or cancellation stops the sequence;
finished clips remain under the job's `parts` folder. There is no automatic retry or resume. `sequence.json`
records completed parts, seeds, observations and completion/failure; the final `video.mp4` is published only after
assembly and exact-frame validation. Ordinary single clips and the normal H3 memory policy are unchanged.

<a id="install-one-shot-natively"></a>

## Install: one shot

In an administrator PowerShell on the laptop:

```powershell
irm https://raw.githubusercontent.com/drowzeys/Imagination-Studio-by-keys-Wasserman/main/oneshot-setup-rtx-spark-n1x.ps1 -OutFile $env:TEMP\n1x.ps1
powershell -ExecutionPolicy Bypass -File $env:TEMP\n1x.ps1
```

`oneshot-setup-rtx-spark-n1x.ps1` checks the machine (ARM64, an N1X), clones the Studio on Windows (`-StudioDir`,
default `%USERPROFILE%\Imagination-Studio`), asks once for a **Hugging Face read token**
(https://huggingface.co/settings/tokens; accept the FLUX3-K and Stable Audio Open licences with that account; saved in
`%USERPROFILE%\.cache\huggingface\token`, `-HfToken hf_...` passes it, `-NoHfToken` skips the gated models) and runs
`scripts\setup-windows-cuda.ps1`. It then makes **Imagination Studio** (start) and **Stop Imagination Studio** with the
Studio's icon on the Desktop and in Start, **pins Imagination Studio to the taskbar** (Windows 11's taskbar layout,
appended to your own pins; it shows after Explorer restarts, or at the next sign-in), and keeps the laptop awake for
renders: no sleep or hibernate plugged in, the lid does nothing, Windows on Best performance, the Wi-Fi adapter never
powered down (`-NeverSleepOnBattery` for battery too, `-NoPower` to skip). Optional: `-StartAtLogon` (starts the Studio
at sign-in), `-EnableSsh -AuthorizedKey "ssh-ed25519 ..."` (another machine reaches the laptop over SSH on the home
network). `-Native` is accepted for older command lines and changes nothing.

Before any component install/download, native setup prints a runtime-fit plan. The default broad selection
skips missing models whose existing runtime admission requirement exceeds this machine's GPU pool (and host
pool where applicable), not merely its current free memory. For example, Drama 3 needs 50 GiB free under its
current policy, so a fresh install with a 30.5 GiB N1X GPU pool skips Voice automatically. A machine with 96 GiB
installed RAM still needs a suitable GPU carve-out and live headroom; it is not automatically qualified.
Passing `-NativeParts` selects an explicit set instead. An explicit capacity mismatch requires
`-AllowInsufficientModel` to download anyway. Cached files remain untouched; temporary contention and unknown
requirements are not labelled permanently incompatible. `-PlanOnly` is still probe-free. See
[Windows setup and qualification](windows.md) for policy sources, limitations, and the separate unknown-fit
image acknowledgment.

`scripts\setup-windows-cuda.ps1` installs:

1. the build tools where missing: Python 3.13 (ARM64), Git, Zig 0.17 and the Visual Studio 2022 Build Tools with the
   ARM64 compiler (winget), and NVIDIA's CUDA 13.4 toolkit for Windows on Arm from its redistributable archives (nvcc,
   the runtime, cuBLAS, NVRTC, nvJitLink, CCCL, NVVM; each archive checked against NVIDIA's sha256) into `C:\cuda-13.4`;
   Zig is detected before invoking it. Missing or older versions trigger the pinned winget install, a terminal
   PATH refresh and a version recheck. If Zig is still unavailable, setup stops with instructions to open a new
   PowerShell window or correct PATH order, rather than continuing to an engine build with missing tools.
2. the app's venv, FFmpeg, the GPU venv (NVIDIA's PyTorch 2.14 for CUDA 13.4) and the pinned ComfyUI (PR #200's scripts);
3. TensorFold's engines **built from source** (no prebuilt release yet): drowzeys/TensorFold `studio-1.2.2` at
   `1c8bdbfd`: `zig build tf-h3-int8 -Dsm=121` (`tf_h3.dll`), `zig build native bolt -Dsm=121 -Doptimize=ReleaseFast`
   (`tensorfold-native.exe` + `bolt.dll`), `zig build fish-drama3-generate fish-drama3-lib -Dsm=121`
   (`drama3-generate.exe` + `tensorfold_drama3.dll`) and, with the music part, `zig build music3-generate -Dsm=121
   -Doptimize=ReleaseFast` (`music3-generate.exe`), in an ARM64 developer prompt, into
   `%LOCALAPPDATA%\ImaginationStudio\tf-h3`, `...\bolt\native\bin`, `...\drama3\bin` and `...\music3\bin`, each with
   the commit it was built from (a run at the same commit keeps them);
4. the models (`scripts\windows-cuda-setup.py models`: the DGX Spark's files from the same repos, with the Spark
   scripts' own SHA-256s where they pin one; `-NativeParts`, default video, image, music, FLUX3-K, sound effects, swap
   (Ref2VA int8, the Character Swap and Turbo LoRAs, SAM 3.1: 25 GB), workflows (the Library's actions: three node packs,
   SAM 3.1 and the latent upscaler), foley (kijai's ComfyUI-MMAudio and MMAudio's 5.6 GB) and finish (Tesseract), plus
   NVIDIA's Bolt checkpoints and Fish Audio's Drama 3; `-NoBolt` and `-NoVoice` leave those out; `train` adds
   musubi-tuner's venv and its 46 GB of files: MiniMax H3's FL2VA transformer in bf16, its fp16 video encoder and ostris's
   adapter);
5. ComfyUI's model folders and `cuda-tier.json`.

Run again, it resumes and updates. `-ToolchainOnly` installs step 1 alone; `-SkipToolchain -Parts <part>` adds one part
to an install.

## Start and stop

- **Start:** click Imagination Studio on the taskbar or Desktop. Its notification-area tray controller starts the
  Studio and opens `http://127.0.0.1:7870`. If an app is already running, it only opens the page.
- **Stop:** choose Stop Studio in the tray, or Stop Imagination Studio in Start. Confirm the action; queued or
  running jobs prevent stopping. A tray-owned app gracefully unloads its models before exiting.
- **Exit tray** leaves Studio running; a separate supervisor owns the app, so no visible command window is needed.
  It never kills by name.
- An app started manually, over SSH, or by an older launcher is labelled **External Studio**. The tray can open
  it but will not adopt or stop it. Stop it through its original launcher before starting a tray-owned session.

An existing checkout can register without running model setup. Use an updated checkout and a Windows Python 3.13
runtime (`py` or the native UI environment); the tray controller itself uses only the standard library:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-windows-desktop.ps1
```

Registration creates the Start/Stop shortcuts and keeps a local copy of the tray controller. A WSL2 registration left
by an older install is ignored and dropped at the next registration. Settings and private launch identity live outside
the repository in `%LOCALAPPDATA%\ImaginationStudio`; credentials never appear in URLs, public app state or logs.

## Memory

The firmware gives the GPU a **fixed carve-out**: on an HP OmniBook X 14 with 128 GB, the driver reports 98,304 MiB for
the GPU and Windows sees about 30 GB. Nothing on Windows changes that split: the laptop's BIOS (F.05) has no graphics or
memory-size setting among its 128, and the driver's carve-out is fixed. The CUDA pool Windows offers is about 113 GB
(the dedicated 98 GB plus shared system memory). The models live in the GPU's part, so the split suits the Studio;
what has to fit in Windows' 30 GB is loading them.

ComfyUI loads a model with its dynamic VRAM loader: the file is mapped, and the weights are streamed into GPU memory as
they are first used. The pages it reads stay in ComfyUI's working set after they are copied, so a big load (FastH3's
42 GB, after the 26 GB text encoder) runs Windows' available memory down to under 1 GB, with commit and the GPU fine;
Windows then raises its low-memory signal. `spark/comfy_win.py`'s memory guard answers that signal by trimming
ComfyUI's working set first (`EmptyWorkingSet`: clean file pages go back to Windows' standby list, as Linux's page cache
gives them back on a Spark), and stops the render only if host memory is still low 2 seconds after the trim. Commit and
GPU floors are never relaxed this way.

Measured 2026-11-06 on that laptop: FastH3, 864x480, 90 frames, 4 passes, from a stopped Studio: **133 s** in all,
4.7 s a pass, of which 71.6 s was TensorFold's one-time int8 conversion of the checkpoint (saved for the next start).
The trim fired once during the load; Windows' available memory stayed between 1 and 6 GB until the model was
in GPU memory, then went back to about 20 GB. Without the trim the
same render was stopped by the guard at the load, each time. ComfyUI's classic loader (`--highvram`) is no way out:
it builds the 42 GB model in Windows' RAM before moving it and crashed reading the file.

<a id="native-windows-the-native-cuda-tier"></a>

## How the native Studio works

Background queue and observer processes use hidden launches. A hidden venv command uses its sibling
`pythonw.exe` because the `python.exe` redirector can open a console even with `CREATE_NO_WINDOW`.
Logs remain captured and the process tree stays job-owned. A missing windowless launcher reports that
the Python environment needs repair rather than silently reverting to a visible console.

On Windows the Studio is the same app as on a DGX Spark (`app/windows_runtime.py`, `app/windows_cuda.py`), with
TensorFold's Windows engines and a CUDA pool of about 113 GB:

- each feature uses its installed backend. The shared GPU Python/ComfyUI core does not depend on H3;
  H3 alone retains its DLL and installer-measured pool requirement of at least 90,000 MiB.
  Bounded recipes measured on [48/64 GB](windows-rtx-spark-plan.md) remain available alongside native alternatives.
  The legacy file `cuda-tier.json` records paths/capacity rather than selecting a different app; `-Preview` is a launch alias;
- qualification gating and badges are required below 128 GiB installed RAM and optional/off by default above it.
  Diagnostics remain visible and individual checks can run either way. Prerequisites and live memory guards always
  apply; the threshold does not guarantee model eligibility or qualify unmeasured Windows image recipes;
- the Spark's jobs build the same commands; the bash scripts they run are replaced by their Python ports
  (`spark/windows_scripts.py`: scripts/video.sh, studio.sh, scout.sh, image.sh, edit.sh, with the same checks and words)
  and spark/comfy.sh by `spark/comfy_win.py`: TensorFold's H3 node copied into ComfyUI with `TF_H3_LIB` at
  `tf_h3.dll`, a lower priority, the server owned by a job object and its recorded identity (never by name or port), the
  Windows memory guard beside it (`app/windows_memory.py`: Windows' commit and low-memory signals and the CUDA pool,
  four times a second; it stops the server and the render says why), and the server left warm after a render;
- the Bolt brain is TensorFold 1.2.2's `tensorfold-native.exe` with `bolt.dll` (Mia's recipe on the Windows build), run by
  the Studio as on a DGX Spark (loaded on the first question, unloaded when idle or when a render needs the memory),
  in a job object that ends with the Studio;
- every job kind of the DGX Spark runs natively once its part is installed; a part that is not installed says which
  command installs it (`scripts\setup-windows-cuda.ps1 -SkipToolchain -Parts <part>`). Only
  DGX Spark workers (handing clips to other Sparks) are not offered: this machine renders alone;
- the Library's actions (turnaround sheet, upscale, find a person, remove a person) run their authors' own ComfyUI node
  packs at the Spark's pinned commits (MATLOWAI's H3 Contact Sheet, dntpi's H3 Video Upsampler, Akatz Labs' H3 Relay,
  SAM 3.1 on ComfyUI's nodes) through `spark/spark_workflows.py`, and `scripts/turnaround.sh` has its port
  (`windows_scripts.py turnaround`); Character Swap runs `spark/spark_swap.py` as on a Spark. Their frames and LoRAs are
  hard links where a Spark makes symlinks (a symlink needs a privilege a person may not have), and which ComfyUI serves
  is read from `comfy_win.py`'s record instead of `pgrep`;
- MMAudio foley runs kijai's ComfyUI-MMAudio with the Studio's sampler node, as on a Spark; the installer adds the pack's
  requirements only where they add packages (librosa, numba and the rest have Windows ARM64 wheels);
- Looks -> Train a LoRA runs kohya-ss's musubi-tuner in a venv of its own (`%LOCALAPPDATA%\ImaginationStudio\train`,
  NVIDIA's PyTorch for CUDA 13.4, musubi-tuner's own Pillow fallback where opencv has no ARM64 wheel; bitsandbytes'
  8-bit AdamW runs a step on the N1X), each step in a job object with the Windows memory guard beside it;
- Finish runs Tesseract and Tesseract Converter from their windows-x86_64 releases (the only Windows builds Mirage
  publishes) through the x64 emulation of Windows 11 on Arm (`scripts\setup-windows-finish.ps1`: pinned SHA-256s, the
  bundles' own checksum lists; Tesseract's own install.ps1 refuses an ARM64 processor, so it is not used);
- Fish Audio Drama 3 runs on its native Windows engine (`drama3-generate.exe` with `tensorfold_drama3.dll`) and MiniMax
  Music 3 on its own (`music3-generate.exe`, in place of ComfyUI's nodes), both TensorFold `studio-1.2.2` `1c8bdbfd`, the
  same server protocol as on a DGX Spark; where their files are missing the Voice and Music tabs say so (songs then run on
  ComfyUI's MiniMax Music 3 nodes).

## Measurements from the WSL2 install (2026-11-03 and 04)

Taken while the Studio also had a WSL2 version on the laptop (removed 2026-11-06); kept as measured.

### Speed: performance mode against the DGX Spark

Measured 2026-11-03, HP OmniBook X 14 (RTX Spark N1X, 128 GB, BIOS F.05), plugged in, Windows on Best performance,
against a DGX Spark (GB10). Same Studio (`main`), same TensorFold int8 engine (`1f7ecc75`), same ComfyUI, same seed:
the clips are identical byte for byte on both machines. GPU clock and power are nvidia-smi's, sampled every 2 s while
the passes ran.

**864x480, 124 frames, 8 passes (models loaded):**

| Machine and mode | A pass | The clip | GPU while rendering |
|---|---|---|---|
| DGX Spark (GB10) | **6.6 s** | — | — |
| N1X, HP **Performance** | **7.0 s** | 72.7 s | ~1957 MHz, ~58 W, up to 75 °C |
| N1X, HP default (Smart Sense / Balanced) | 9.0 s | 93.8 s | ~1555-1600 MHz, ~35-38 W, ~70 °C |
| N1X, HP default, Fan Always On (BIOS) | 9.0 s | 94.2 s | ~1555 MHz, ~35 W |
| N1X, Windows **Best power efficiency** (power saving) | 13.5 s | 138.5 s | ~1507 MHz, ~30 W, up to 67 °C |

**1376x768, 192 frames (8 s), a Studio video job from a first frame, from a cold start (models loaded first):**

| Machine and mode | The whole job | Of it, the render | GPU while rendering |
|---|---|---|---|
| DGX Spark (GB10) | 625.6 s | 410.1 s | — |
| N1X, HP **Performance** | **619.0 s** | 585.1 s | ~1896 MHz, ~51 W, up to 74 °C |
| N1X, out of Performance (Smart Sense had switched it) | 736.1 s | 658.8 s | ~1400 MHz, ~31 W |

- In HP's Performance mode the N1X is within about 6% of the DGX Spark a pass at 864x480, and a whole 1376x768 job
  finished as fast as the DGX Spark's (619 s against 626 s; its render took 585 s against 410 s, its QA gate less);
  in its default mode a pass is about 36% slower, in Windows' power-saving mode about twice as slow (13.5 s).
- The limit is the laptop's power budget, not heat or WSL: the GPU is 96% busy, under 75 °C, and nvidia-smi reports
  software power capping; the fan made no difference. The GPU's own power limit is not settable (N/A).
- HP's Smart Sense can leave Performance by itself: check myHP before a long render, keep it plugged in and the lid
  open (many laptops cut power with the lid shut).
- The whole-job times include loading the models and the QA gate's look at the take, which differ between runs.

### An LLM on the N1X: Performance against power saving

Not the Studio's own workload, measured the same way for anyone serving a language model on the laptop beside it.
Measured 2026-11-04 on the same HP OmniBook X 14, plugged in, in WSL2 Ubuntu: vcruz305's
[Qwen3.8-Flash-Next EXL3 recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe) (exllamav3
`bccada04` and TabbyAPI `880e6b8`, built for sm_121 with the CUDA 13.4 toolkit) serving
[CYBER-FROST-3.8-EXL3-SAGE-3.87bpw](https://huggingface.co/vcruz305/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw) with MTP drafts and
the 39 GB n-gram table streamed from disk (WSL has ~29 GB of RAM to cache it), about 69 GB of the GPU's 96. The
recipe's own clients: `bench_v1.py` (one warmup, three 400-token replies a class) and `concurrency.py` (3,000-token
prompts, 400 tokens, three rounds, totals include prefill). "Performance" is HP Performance with Windows on Best
performance; "power saving" is Windows' Best power efficiency. GPU clock and power: nvidia-smi every 2 s while busy.

| | Performance | Power saving | DGX Spark (the recipe's own) |
|---|---:|---:|---:|
| Decode, code (tok/s) | **41.0** | 25.6 | 45.8 |
| Decode, DevOps (tok/s) | **36.3** | 21.6 | 44.3 |
| Decode, prose (tok/s) | **29.0** | 16.4 | 34.4 |
| First token, short prompt (s) | 0.74-0.80 | 1.33-1.44 | — |
| 17,841-token prompt: prefill (tok/s) | **950** | 460 | — |
| 17,841-token prompt: first token (s) | 19.1 | 39.6 | — |
| Decode after that prompt (tok/s) | 29.3 | 18.3 | — |
| 1 / 2 / 4 requests at once, total (tok/s) | 30.6 / 33.6 / 36.2 | 17.6 / 18.3 / 19.6 | — |
| GPU while busy | 2166 MHz, 48 W, up to 72 °C | 1375 MHz, 15 W, up to 62 °C | — |

- Power saving roughly halves everything (decode 0.56-0.62x, prefill 0.48x) at under a third of the GPU's power.
- In Performance the laptop decodes at 82-90% of the DGX Spark's figures (the recipe's, on a GB10).
- More requests at once add little (+18% total at four): every token reads the n-gram table, which WSL's small RAM
  cannot hold, so the disk is shared.
- vcruz305's DSV4.1-Flash-1017 EXL3 recipe holds about 107 GiB of device memory on a GB10 and does not fit the N1X's
  96 GB carve-out.

### TensorFold on the N1X: in WSL and native on Windows

The same model family on [TensorFold](https://github.com/ashhart/TensorFold) 1.2.2's Flash-Next CUDA engine, serving
[Blackfrost-AI/CYBER-FROST-3.8-FP4-Local](https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-FP4-Local) (ModelOpt
mixed FP4, 72.7 GiB on the GPU, n-gram tables read from the files, MTP drafts), `--context 32768 --parallel 4
--no-thinking`, measured 2026-11-04 with the same clients and settings as the EXL3 runs above, HP Performance mode:

- **In WSL**: TensorFold's Linux build, rebuilt with one fix. Its integrated-GPU admission reads `/proc/meminfo` on
  Linux, which under WSL is the ~30 GB of RAM, not the GPU's pool, so it refused the checkpoint; under WSL it now reads
  the CUDA pool as on native Windows (sent upstream).
- **Native on Windows**: `tensorfold-native.exe` built from source for `aarch64-windows`, `sm_121` (Zig 0.17, the CUDA
  13.4 Windows-on-Arm toolkit from NVIDIA's redistributable archives, MSVC ARM64 build tools); its GPU suites pass
  (`glue`: 1473 of 1473 byte-equal). Natively the WDDM pool is 113 GB (WSL sees 96).

Client end-to-end tokens/s (prefill included; TensorFold's server reports no separate decode rate):

| | DGX Spark, EXL3¹ | N1X EXL3, WSL | N1X EXL3, WSL, power saving | N1X TensorFold, WSL | N1X TensorFold, native Windows |
|---|---:|---:|---:|---:|---:|
| Code | 45.8 | 38.6 | 25.6² | 43.8 | **50.4** |
| DevOps | 44.3 | 34.5 | 21.6² | **42.2** | 40.2 |
| Prose | 34.4 | 27.8 | 16.4² | 35.7 | **37.3** |
| First token, short prompt (s) | — | 0.74-0.80 | 1.33-1.44 | **0.21-0.24** | 0.22-0.28³ |
| First token, ~17,850-token prompt (s) | — | 19.1 | 39.6 | **10.2** | 10.9³ |
| 1 / 2 / 4 requests at once, total | — | 30.6 / 33.6 / 36.2 | 17.6 / 18.3 / 19.6 | 36.4 / 50.2 / 65.6 | **36.6 / 54.4 / 70.0** |
| GPU while busy | — | 2166 MHz, 48 W | 1375 MHz, 15 W | 2120 MHz, 49 W | 2139 MHz, 42 W |
| GPU memory left for streams | — | — | — | 15.4 GiB | **19.0 GiB** |

¹ [vcruz305's recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe), its own GB10 figures for
the Cyber Frost 3.87 bpw pack (server decode, n-gram table on disk, corrected template); not measured here.
² Server decode rate (the power-saving run's client figure was not kept). ³ Includes ~6 ms of network round trip: the
clients ran on another machine through an SSH tunnel.

- TensorFold against EXL3 on the same laptop: +13-28% a stream, first token 3.4x sooner, a long prompt in about half
  the time, and 1.8-1.9x the total at four requests.
- Native Windows against WSL: within about ±15% (three samples a class), and more memory left for streams.
- On TensorFold the N1X matches or passes the DGX Spark's EXL3 figures for this model.

## Native measurements (2026-11-04)

### Measured natively

2026-11-04, the HP OmniBook X 14 (RTX Spark N1X, 128 GB), plugged in, Windows' Balanced plan, the Studio on the native tier, every job queued through the Studio's own API (`/api/jobs`,
`/api/voice/takes`, `/api/music/takes`, `/api/brains/test`) as the page queues it. Wall time is the job's, queued to done.

| Job | Native Windows | WSL on the same laptop | DGX Spark |
|---|---:|---:|---:|
| FastH3, 8 passes, 864x480, 124 frames (5 s), text to video, models loaded | **73.8 s** (a pass 7.03 s) | 72.7 s in HP Performance (7.0 s), 93.8 s in HP's default (9.0 s) | 73 s (6.6-7.1 s) |
| The same, first job after a start (ComfyUI started, models loaded) | 114.0 s | — | — |
| The same from a first frame (the still below), models loaded | 92.7 s (a pass 8.54 s) | — | — |
| Text to picture to video (`studio`, Qwen-Image then FastH3), from a cold server | 131.8 s | — | — |
| A still, Qwen-Image-2.1-Turbo 8 steps, 864x480, server started for it | 31.0 s (the graph 14.2 s) | — | — |
| Fish Audio Drama 3, one 7.5-7.8 s sentence, engine loaded for it | 39.1-39.2 s (load 34.0-34.4 s; 4.3-4.7 s to speak it, 1.65-1.74x real time, first sound 223-229 ms) | — | — |
| MiniMax Music 3 on ComfyUI, a 30 s song, Drama 3 unloaded first | 133.8 s | — | — |
| Bolt (Studio-managed), load and first reply | 27.0-30.4 s; a 70-token reply 1.4 s | — | — |
| Bolt, 400 tokens, one stream (prefill included) | prose 58.6-60.8 tok/s, code 115.9-119.2 tok/s | — | — |

- With the GPU to itself a pass is 7.03 s natively and a warm clip 73.8 s: WSL's HP Performance figures (7.0 s, 72.7 s)
  and within 7% of a GB10's pass. Earlier runs the same day, beside another session's Drama 3 engine in WSL, took 7.56-7.65
  s a pass (80.7 s warm, 122.7 s cold); the first-frame row is from those. Keep HP's Performance mode (myHP), plugged in.
- These rows are from engines the installer built from `a545f786` (H3, Bolt, Drama 3 rebuilt in 261 s with the
  toolchain already there); the first runs used builds of the same branch made by hand.
- The memory guard watched every job. On WDDM `cudaMemGetInfo` gives a process its own budget (it read 105.7 of
  113.1 GB free throughout while ComfyUI held 58 GB), so the guard holds NVML's dedicated counter to a floor
  (`DEDICATED_FLOOR_MB`) beside Windows' commit and low-memory signals. With another engine holding about 40 GB of
  WSL's RAM, Windows' low-memory signal stopped a video's load (commit 116.6 of 117.6 GB): run one stack at a time.
  The guard holds commit to a 4 GB floor rather than to `windows_memory.pressure`'s 0.5-second projection, which read
  a model load's single 21 GB commit as commit falling to nothing and stopped a load with 19 GB to spare.

### Every feature natively: what was run (2026-11-04, later the same day)

The same laptop and Studio (`windows-native-features`), the engines rebuilt by the installer from TensorFold
`studio-1.2.2` `1c8bdbfd`, every job queued through the Studio's own API (`/api/jobs`, `/api/voice/takes`,
`/api/music/takes`, `/api/finish/{id}/convert`, `/api/loras/{id}/install`). Wall time is the job's, queued to done. The
clip used by the actions is a FastH3 render of a generated scene (a man in a red jacket crossing a town square, 864x480,
124 frames: 113.9 s from a cold server at 4 passes).

| Job | Native Windows | DGX Spark |
|---|---:|---:|
| Find a person (SAM 3.1 on ComfyUI's nodes, Akatz Labs' green mask), the 5 s clip, ComfyUI restarted for it | **65.7 s** (finding and following him 41.9 s, the clean plate 0.1 s) | 36-40 s for the mask (README) |
| Foley (MMAudio large_44k_v2, 25 passes, kijai's nodes), the 5.17 s clip, ComfyUI restarted for it | **48.7 s** (the graph 31.4 s) | — |
| Finish (Tesseract 0.4.0, x64 emulation), two clips, a title card and a song under them, 11.7 s at 1080p 24 fps | **12.4 s** | — (a Spark finishes on a Mac) |
| Finish, two cuts, 6 s at 720p | 7.6 s | — |
| Finish -> After Effects project (Tesseract Converter 0.3.0) | done, a few seconds | — |
| MiniMax Music 3 on its TensorFold engine, a 30 s song, Drama 3 unloaded first | **80.9 s** (load 20.9 s; the song 57.9 s: words 32.6 s, sound 23.4 s, decode 1.9 s; 1.93x real time; 16.5 GB at the peak) | 60.3 s for the song, load ~16 s |
| The same song on ComfyUI's nodes (before this change) | 133.8 s | — |
| Fish Audio Drama 3 on the rebuilt engine, a 5.3 s sentence, engine loaded for it from a cold disk | 94.4 s (load 90.6 s; 3.45 s to speak it, 1.55x real time, first sound 204 ms) | 29-31 frames/s |
| LoRA hub: Turnaround (matlod, 63 MB) installed | done | — |

- Find a person and foley restart ComfyUI before they load (a server that rendered a clip starts again empty, as on a
  Spark). The node packs of all four actions and foley load in the Windows ComfyUI (its import report lists no failure).
- The Drama 3 load above is from a cold disk (the engines and their page cache had just been rebuilt); a warm load is
  34 s, as in the table before. The Studio no longer rewrites Windows paths for Drama 3: `1c8bdbfd` JSON-escapes the
  path it echoes (`88415e41`), and an engine from `a545f786` would now time out on its first take: rebuild it (the
  installer does, by its commit).
- MiniMax Music 3's own engine is 1.65x faster than ComfyUI's nodes on this laptop, and its song as fast as on a GB10;
  its load is slower (20.9 s against about 16 s).
- Not run here, for want of disk (the laptop's C: had 19.7 GB free, other models fill the rest): the turnaround sheet,
  upscale, remove a person and Character Swap need MiniMax H3 Ref2VA (21 GB in int8) and LoRA training needs 46 GB of
  files (FL2VA in bf16, the fp16 video encoder, ostris's adapter). Their code paths ran as far as the weights: the node
  packs load, SAM 3.1 finds and follows the person, the hub installs their LoRAs, the training venv installs
  (`windows-cuda-setup.py train`, 180 s) and bitsandbytes' 8-bit AdamW runs a step on the GPU; the Studio then refuses
  each with the command that installs the missing part. Measured on a DGX Spark they hold 28-53 GB (remove, swap) and
  about 55 GB (a LoRA run), inside the native pool's 113 GB.
- Tesseract Converter's Windows release (0.3.0, and 0.4.0's source) makes the After Effects project but refuses a
  Premiere Pro one that links an After Effects composition ("AEP artifact is not a bounded portable local path"): it
  checks each linked path written with `/` against the same path rebuilt with `\`
  (`apps/tesseract-conv/src/hybrid/package.rs`; [mirage-hq/Tesseract-Converter#6](https://github.com/mirage-hq/Tesseract-Converter/issues/6)).
  Titles, captions and any clip scaled to the canvas need that link. `setup-windows-finish.ps1` therefore builds the
  converter natively for Windows on Arm from the release's commit with that comparison made on `/` both sides, against
  BtbN's FFmpeg 8.1 LGPL shared build (Rust, LLVM's libclang and the ARM64 compiler, installed when missing); on this
  laptop the build took 166 s. Its Premiere package for a finish with a title and a Foley take has the same ten files,
  sizes and relative references as the Mac's. If the build fails, the release is installed and the Studio says how to
  build the fix. Resolve's FCPXML and EDL, written by the Studio itself, come with every finish.
- Tesseract's own MP4 export on Windows (0.4.0, x64 emulation) hands the encoder frames out of step: in a finish of a
  title, a Foley take and a render (1080p24, 11 s), 170 of 177 clip frames differed from the same project rendered on
  a Mac, some repeated, the title's fade-out was held, and the MP4 was H.264 Constrained Baseline at 3.6 Mbit/s. Its
  ProRes export matched the Mac's on every clip frame (the fade-out is still held), so on Windows the Studio renders
  ProRes and encodes the MP4 itself with its FFmpeg (H.264 High, BT.709, 7.6 Mbit/s), drawing the title's fade to black
  there. That finish takes 62 s instead of 9 s; without a title, 46 s for 7.4 s.
- The installer, end to end: `-ToolchainOnly` into an empty folder fetched and checked NVIDIA's eight CUDA 13.4.1
  archives (0.79 GB) in 62.9 s, and that toolchain built `tf_h3.dll` and `music3-generate.exe` from an empty cache in
  78.5 s; a full run with the toolchain present (every part above, the models already there) took 284 s, the four
  engines rebuilt at the new commit; `setup-windows-finish.ps1` took 15.3 s.
