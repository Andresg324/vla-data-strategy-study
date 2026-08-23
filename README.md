# SO-101 × SmolVLA: How Demonstration-Collection Strategy Shapes Generalization

A controlled empirical study on a self-built low-cost robot arm (Seeed SO-ARM101,
LeRobot platform) using the SmolVLA vision-language-action model.

<p align="center">
  <img src="media/overhead_demo.gif" width="520" alt="Autonomous cube pick-and-place">
</p>

## The question

A VLA learns to map camera images and a language instruction directly to robot motion,
end-to-end from demonstrations, with no hand-coded perception or inverse kinematics. This
project asks: how does the way you collect demonstrations affect how well the learned policy
generalizes to conditions it never saw?

## Demo

The clip above is the **overhead camera view** during autonomous inference. The leader is
disconnected and the policy is driving the follower arm on its own. It receives only the two
camera feeds (overhead and wrist) and the instruction *"Pick up the cube and place it in the
cup."* and outputs motion directly, with no hand-coded perception, planning, or teleoperation.

This is one in-distribution rollout from the Color-varied policy at seed 1000, played at 2x.
The cube is at the trained position T6 under baseline lighting, and the policy picks it up and
releases it into the cup inside the 45-second window. The four policies already separate here,
before anything is held out: across both seeds Clean succeeds 30 of 30 in distribution, Color 25
of 30, Recovery 17 of 30 and Randomized 12 of 30. The question the study asks is what happens
when any one thing changes, and the answer differs sharply by how the demonstrations were
collected.

## Design

See [PROTOCOL.md](PROTOCOL.md) for the full pre-registered protocol and
[RUN_SHEET.md](RUN_SHEET.md) for the as-run record.

Fix the model (SmolVLA), the task (Pick up the cube and place it in the cup), the demo budget
(50 episodes per condition), and the training hyperparameters. Vary only the data-collection
strategy, one factor at a time:

| Condition | What varies |
|---|---|
| **Clean** | nothing: red cube, one fixed position, every demo a first-try success |
| **Randomized** | cube start position, cycled over 10 marked positions |
| **Recovery** | 20 of 50 demos include a deliberate drop during the carry |
| **Color-varied** | cube color, cycled over five colors; green held out for evaluation |

A fifth condition, **Density**, was added afterwards as an exploratory probe
([PROTOCOL.md §8.30](PROTOCOL.md#8-amendments)): the same 50-demonstration budget split between
two positions rather than one or ten. It is single-seed, was declared before it was run, and is
never pooled into the four-condition grid.

<p align="center">
 <img src="figures/fig_positions.png" width="440" alt="Training and held-out cube positions">
</p>

*The ten training positions (T) and the five held-out evaluation positions (E), with the convex
hull of the training set drawn. E2, E3 and E4 fall inside it, E1 and E5 outside.*

All four policies are then evaluated on the same five cells: one in-distribution reference plus
four held-out (new positions, reduced lighting, unseen object color, distractors), at 15 scored
episodes each (16 recorded, index 0 discarded as warmup), for 300 rollouts per seed and 600
registered rollouts across two training seeds. A further 107 rollouts were recorded for three
exploratory probes (displacement, demonstration pace and sampling density) and are reported
separately, never pooled into the grid. Held-out positions are split into interpolation and
extrapolation relative to the convex hull of the training positions and reported separately.

The protocol was pre-registered before any study data was collected. Amendments made on the
rebuilt workstation are listed, dated and justified in
[§8](PROTOCOL.md#8-amendments).

## Results

Every quantity below regenerates from the raw scores with the commands in
[SETUP.md](SETUP.md#analysis); the analysis map is in
[analysis/README.md](analysis/README.md).

Across 600 registered rollouts **both pre-registered comparisons are null, for opposite
reasons**. New Positions is 0/15 for every policy at both seeds, 0 of 120, so that
comparison sits on a floor; on the held-out color both compared policies sat at or near
ceiling. Success rate reported no effect anywhere, but the telemetry did.

Ranking the four conditions depends entirely on which measure you use, which is the
study's central point in one picture.

<p align="center">
 <img src="figures/fig_by_condition.png" width="640" alt="Four measures of the four conditions">
</p>

*Four measures of the same four policies, circles seed 1000 and diamonds seed 2000. Grid
success rate; the fraction of episodes in which the arm never left the home pose (registered
cells only, 75 per seed); the spread of median commanded bearing across the four cells that
hold the cube at T6; and final training loss. Randomized is last on the first, worst on the
second and third, and best on the fourth.*

**Six of the eight policies aim at the trained cube position even in the cell where the
cube is somewhere else.** Their median commanded bearing moves by 0.2 to 2.3 degrees
across all five evaluation cells. Only the position-randomized policy's aim moves with
the cube, by 9.1 and 22.0 degrees, and inside the training hull it localizes to within a
fifth of a cube width.

**The clean-data policy's reach is a fixed sweep.** It turns to about 27 degrees whatever is in
front of it. Reaching the cube requires 24.2 degrees at the trained position, 26.6 at a one-inch
displacement and 29.4 at two inches, so the sweep clears the first comfortably, grazes the second
and cannot reach the third. Episodes that get far enough to touch the cube: 15 of 15 in
distribution, then 6 of 8 and 4 of 8 at one inch across the two seeds, then 0 of 8 at two inches
at both. The policy is not failing to localize the cube, it is not looking for it.

<p align="center">
 <img src="figures/fig_aim_invariance.png" width="420" alt="Commanded bearing by evaluation cell">
 <img src="figures/fig_envelope.png" width="420" alt="Furthest bearing reached, clean policy">
</p>

*Left: median commanded bearing per policy across the five evaluation cells. Six of the eight
lines are flat; only the position-randomized policy's aim moves with the cube. Right: the clean
policy's reach against the bearing each cube position requires.*

**Clutter destabilizes the position-randomized policy's aim.** Its commanded bearing under
distractors scatters to an interquartile range of 8.4 degrees at both seeds, against at most 1.6
for every other policy in every cell. The median moves too, though only decisively at seed 2000,
by 8.4 degrees against at most 1.7 elsewhere. The aim does not land on the distractors, so this
is destabilization rather than capture: the one policy trained to look around is the one clutter
unsettles.

**Position is in the representation either way.** A linear probe reads cube bearing from
every policy's final hidden state at 8.6 to 11.4 degrees against a 42-degree chance
floor, including the policies whose aim never moves. Holding out whole positions rather
than episodes collapses decoding to 26 to 42 degrees with intervals spanning chance, so
the code is interpolative. What the collection strategy changed is not whether position
enters the representation, but whether the readout to action uses it. Only the action
expert is fine-tuned, so the backbone supplying that representation is identical across
all eight policies.

<p align="center">
 <img src="figures/fig_probe.png" width="420" alt="Bearing decoded from hidden states">
 <img src="figures/fig_aiming_error.png" width="420" alt="Aiming error, interpolation vs extrapolation">
</p>

*Left: bearing decoded from each policy's final hidden state, by episode and with whole positions
held out. Right: the randomized policy's aiming error at held-out positions, split by whether the
position falls inside the training hull.*

**The outcome is unreadable until the gripper closes.** A probe swept across the episode does
not clear its own within-cell permutation null before the grasp in any policy that replicates
across seeds, while a full-episode control reaches 0.991 to 1.000. Randomized clears it at seed
1000 (p = 0.046) but not at seed 2000 (p = 0.176), so the one apparent early signal does not
replicate. Two policies are not testable: Clean at seed 1000 and Color at seed 2000 have only 5
and 4 episodes in the minority outcome.

<p align="center">
 <img src="figures/fig_success_sweep.png" width="480" alt="Outcome decoding swept across the episode">
</p>

**Position diversity cost execution outright.** The randomized policy never left the home pose in
11 and 18 of 75 episodes across the two seeds, against 4 and 7 of 91 for Clean, and posted the
lowest grid totals in the study at 12/75 and 13/75. Fifty demonstrations spread across ten
positions left too few at each to specify an action confidently.

**Twenty-five demonstrations at a position buys that position and nothing else.** The
exploratory Density policy ([§8.30](PROTOCOL.md#8-amendments)) split the same 50-demonstration
budget between two positions on opposite sides of the arm, T6 at +24.2 degrees and T2 at −31.0.
It scores 15/15 at both, the only policy in the study to reach ceiling anywhere other than T6,
and 0/15 at the five held-out positions. What it does there is not poor aim. At four of the five
it settles within 1.4 to 3.2 degrees of one of the two bearings it was trained on while sitting
22 to 64 degrees from the cube, including three episodes that require +34.7 degrees and are
answered at −28.9. Sampling density fixes execution exactly where it is spent and produces a
selector between learned answers everywhere else. This experiment was a single seed, a separate 
recording session, and never pooled into the grid.

<p align="center">
 <img src="figures/fig_bearing_selection.png" width="640" alt="Commanded bearing against required bearing at held-out positions">
</p>

*Commanded bearing against the bearing each held-out cube position requires. Left: the
position-randomized policy follows the identity line where it interpolates and loses the target
where it extrapolates. Right: the density policy collapses onto the two bearings it was trained
on and ignores the target.*

**Training loss does not order the policies the way evaluation does.** All ten runs
converge by roughly step 6,000. Randomized reaches the lowest final loss of any run and
scores worst in the study, while the two highest final losses belong to Density, which is
perfect at both positions it was trained on, and Recovery, which scores three times better
than Randomized. The collapse is not a training failure.

<p align="center">
 <img src="figures/fig_loss.png" width="480" alt="Training loss for all ten fine-tuning runs">
</p>

**Recovery inherits the release point exactly.** Its dropped cubes land at a median
bearing of -11.7 degrees against the demonstrated -10.8, one degree apart against a
calibration accurate to 0.86 degrees, both about 65% of the way from cube to cup. It
inherits the first half of the demonstrated behavior and not the second: 6 of 53
releases were followed by the re-grasp that completes every demonstration.

<p align="center">
 <img src="figures/fig_release.png" width="480" alt="Release bearing, demonstrations against rollouts">
</p>

**The demonstrator's tempo transfers.** Completion time on successes is perfectly rank-ordered
with demonstration velocity across all five datasets (Spearman rho = -1.0, n = 5, p = 0.017,
the floor at this n), and the four datasets with policies at both seeds reproduce the ordering.
The slow-pace control policy finishes in 22.4 s against the retained Color policy's 13.4 and
13.5 s.

Tables in `figures/table1.md`, `figures/table2.md` and `figures/table_loss.md`; every figure
above is regenerated by `analysis/make_figures.py` in both PNG and PDF. Raw scores in
`documents/results_raw_two_seeds.xlsx` (707 scored episodes, one row each).

## Failure modes on video

Overhead camera, real time unless noted. Each clip is one scored episode from the
datasets listed above.

### The reference

<img src="media/clip1_clean_success.gif" width="420" alt="Clean policy completing the task">

Clean, in distribution. Approach, grasp, carry, release. Every clip below is a departure
from this.

### Two inches is enough to break it

<img src="media/clip2_offset_2in.gif" width="420" alt="Arm falling short of a displaced cube">

Clean, cube moved two inches toward the arm. The arm stops short and the gripper closes on
nothing. Reaching the displaced cube needs a bearing of 29.4 degrees; across the eight
episodes the arm reaches a median maximum of 27.7, and none of the eight gets there. In
distribution it reaches 27.0 against a requirement of 24.2, so the trajectory barely
responds to the displacement at all: the target moved 5.1 degrees and the arm moved 0.7
(p = 0.20).

### The policy that does not move

<img src="media/clip3_no_motion.gif" width="420" alt="Arm vibrating in the home pose">

Randomized seed 2000, green cube. Thirty-eight seconds with a maximum joint deviation of
5.19 degrees. Trimmed to 15 s; the rest looks the same. Eighteen of 75 episodes at this
seed never left the home pose.

### The drop that transferred, and the recovery that did not

<img src="media/clip4_drop_no_recovery.gif" width="420" alt="Cube dropped twice mid carry with no recovery">

Recovery seed 2000. The deliberate release from the demonstrations transferred cleanly: across
53 episodes the policy releases at a median bearing of −11.7 degrees against a demonstrated
−10.8. What did not transfer is what comes after it. Here it grasps and releases twice and
recovers neither time, and the window expires with the cube on the bench.

<img src="media/clip5_rare_regrasp.gif" width="420" alt="Cube dropped then recovered">

The exception: 6 of 53 drop episodes ended in a successful re-grasp, all at seed 2000.

### Correct aim, no grasp

<img src="media/clip6_correct_aim_no_grasp.gif" width="420" alt="Arm aiming correctly but failing to grasp">

Randomized seed 2000 at a held-out position. Commanded bearing 5.8 degrees against a true
5.7. The arm points at the cube and still fails to close on it, which is why aiming and
success are reported separately.

### A policy choosing between two learned answers

<img src="media/clip7_density_selector.gif" width="420" alt="Arm turning toward one trained position then crossing to the other">

Density (PROTOCOL.md §8.30), trained on 25 demonstrations each at T6 and T2 only, evaluated
at held-out E4. The cube needs +34.7 degrees. The arm turns toward T6, hesitates near the
middle, then crosses to T2 and stays. It is not aiming badly; it is picking one of the two
bearings it knows.

## Hardware

- Seeed SO-ARM101 Pro (leader 5 V / follower 12 V), Feetech STS3215 servos
- Two USB cameras: Logitech C270 overhead on a fixed gantry, Seeed webcam at the wrist, both
  operated at **640 × 480 @ 30 fps**
- Work surface in flat gray primer; two 1000 lm / 4000 K clamp LEDs bounced off the ceiling
- MacBook Air for collection and inference; Colab A100 for training

<p align="center">
 <img src="media/bench_wide.jpeg" width="600" alt="The study bench">
</p>

*The bench: leader and follower arms, the overhead gantry, both clamp LEDs, the cup and the
marked work surface. Camera position is locked for the duration of the study.*

<p align="center">
 <img src="media/overhead_baseline_lighting.jpg" width="380" alt="Overhead camera view">
 <img src="media/wrist_view.jpg" width="380" alt="Wrist camera view">
</p>

*The only two visual inputs the policy receives, at the resolution it receives them: overhead (left) and
wrist (right), both 640 x 480.*

## Pipeline

1. `scripts/record_dataset.sh`: teleoperate and record synchronized camera and joint data
2. Train SmolVLA on a cloud GPU (fine-tune from `lerobot/smolvla_base`), log to Weights & Biases
3. `scripts/run_inference.sh`: trained policy drives the arm autonomously
4. `tools/export_results.py` then `analysis/`: regenerate every reported number from the scores

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | collection, inference and camera bring-up shell scripts |
| `tools/` | camera checks, episode playback, and the motion and geometry analyses |
| `analysis/` | pre-committed statistics, exploratory analyses, figure generation |
| `probing/` | activation extraction and linear probes (extracted activations are gitignored) |
| `configs/` | the resolved training configuration for each of the ten runs, as written by LeRobot |
| `documents/` | raw scores and the derived CSVs |
| `figures/` | generated tables and figures (`make_figures.py`) |
| `media/` | photographs, camera frames and failure-mode clips referenced by the docs |

`analysis/README.md` maps each script to the numbers it produces. Two tools worth knowing about
outside the pipeline: `tools/check_cameras.py`, a headless-safe camera probe that saves a frame
from each camera so framing can be verified before recording, written because LeRobot's OpenCV
build cannot open a live preview window; and `tools/show_episode.py`, which plays a single
episode out of LeRobot v3's chunked video files.

## Status

- [x] Hardware assembled and calibrated
- [x] Teleoperation verified (leader → follower mirroring)
- [x] Pilot dataset recorded, SmolVLA fine-tuned, first autonomous pick
- [x] Workstation rebuilt; protocol pre-registered and amended before collection
- [x] Four-condition data collection (200 demonstrations retained; Color re-collected, see
      [PROTOCOL.md §8.18](PROTOCOL.md#8-amendments))
- [x] Eight training runs (four conditions × two seeds), plus two exploratory policies
      (demonstration pace and sampling density)
- [x] Evaluation grid complete (600 registered rollouts across two seeds) and analyzed
- [x] Three exploratory probes complete (107 rollouts): displacement, demonstration pace,
      sampling density
- [x] Label audit across all 707 scored episodes, three telemetry screens

## Pilot results (superseded by the study)

> **These results come from the previous workstation and are not part of the study.** The bench
> was rebuilt in August 2026 with different camera geometry and a gray work surface in place of
> blue tape, so the pilot data is no longer distribution-matched and is **not** mixed with study
> data. It is kept here because it established that the pipeline works end to end.

Closed the full pipeline end-to-end: teleoperated data collection, SmolVLA fine-tuning
(10k steps, single A100), autonomous inference on the real arm. The policy reliably picks and
places when the cube is at the trained position (4 consecutive successes).

When the cube was moved off the trained position, the policy reached and missed repeatedly. This
was a direct, observed instance of the generalization gap that the four-condition study is
designed to measure. The clean-data policy nails the in-distribution pose and degrades off it.

<p align="center">
 <img src="media/loss_curve.png" width="500" alt="Pilot training loss">
</p>

Training loss (`train/losses_after_rm_padding`) fell from ~0.19 to ~0.045 over 10k steps,
plateauing around step 6k; the policy converged well within the run. **This curve is the pilot
run only.** The ten study runs are exported to `documents/training_loss.csv` and plotted
separately by `analysis/make_figures.py`.

**Pilot artifacts:** dataset and trained model links withheld for review.
account; links withheld for review.

## Limitations & observed failure modes

1. **No position generalization:** trained only on the clean (fixed-position) condition, the
   policy reliably fails to grasp when the cube starts outside its trained pose. This is the
   motivating observation for the four-condition study.
2. **Setup-tied:** the policy is bound to the exact camera framing, lighting, background and work
   surface it trained on. This is not incidental, it is why the pilot data was discarded rather
   than reused when the workstation was rebuilt, and why all four conditions are collected on the
   same bench without moving the camera.
3. **Scope:** one arm, one task, 50 demonstrations per condition; results are suggestive, not
   conclusive.
4. **Two training seeds per condition** (1000 and 2000), reported separately and never pooled.
   Two seeds bound training-run variance loosely; per-cell confidence intervals capture
   episode-level uncertainty, not training-run variance.
5. **Only the action expert is fine-tuned.** The vision encoder and language backbone are frozen
   by the base model's defaults, so all eight policies share an identical perceptual front end and
   the study measures what collection strategy does to the action readout, not to perception. See
   [PROTOCOL.md §9](PROTOCOL.md#9-known-limitations) for the full list.

## Reproducing

Requires the LeRobot environment (Python 3.12). See [SETUP.md](SETUP.md).
