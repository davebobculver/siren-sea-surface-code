<!--
EDITING NOTES (invisible on GitHub — delete this block when you're happy with the file)

Voice rules I tried to hold to, so your edits match:
  - First person, past tense for what happened, present tense for what the code does.
  - Short declarative sentences. No hedging, no "we sought to".
  - Every design choice gets a "because". If a paragraph doesn't say why, cut it.
  - Admit rough edges plainly and once. Don't apologize twice for the same thing.

Search this file for "CHECK:" — those are claims I inferred from the code and you
should confirm or reword. Search for "YOURS:" — those are where your own words
will beat mine (links, dates, what you actually believe). Both are inside HTML
comments, so they are invisible to a reader on GitHub and visible to you here.
-->

# Reconstructing a breaking sea surface with neural networks — working repo

This is the research code behind *Differentiable Sea Surface Representations from
Neural Networks Modeling Noisy lidar Point Clouds* (Culver, Feddersen, Morzfeld).
A drone-mounted multibeam lidar flew over extreme swell breaking off China Rock,
Monterey. It returns an irregular, noisy point cloud — not a grid, not a time
series, just ~17 500 scattered `(x, y, t) → z` returns per frame at 10 Hz, tens of
millions over the record, off a surface that is folding over on itself. The job is
to turn that cloud into a function.

I fit a coordinate network $\hat\eta(x,y,t\mid\theta)$ directly to the raw returns.
The trained network *is* the sea surface model: continuous everywhere, defined off
the beams, and differentiable in space and time by construction, so
$\nabla\hat\eta$, $\nabla^2\hat\eta$ and $\partial\hat\eta/\partial t$ come out of
autograd rather than out of finite differences on an interpolated grid.

## Read this first

**This is the lab notebook, not the paper.** It is the repo where the ideas were
found, so it contains the dead ends, the four abandoned regularizers, the
notebook that exists only because I needed one plot on one afternoon, and two
preprocessing modules that do the same thing because I rewrote one and never
deleted the other.

The clean version lives separately: a Zenodo repo with one script per figure, a
single set of modules, and a checkpoint for every number in the paper. That one
shows what I know. This one shows how I got there.

<!-- YOURS: put the Zenodo DOI / link here once it's minted. -->

I am keeping this repo public deliberately. Research code that has been tidied
after the fact hides the part of the work that actually matters — which ideas
were tried, which were discarded, and on what evidence. Everything below is an
honest account of both.

## The modeling problem, and why a network is the right tool

Lidar over breaking water gives you the worst-case version of a scattered-data
problem:

- **The sampling is structured, not random.** 32 beams sweep the surface. Points
  are dense along a beam and sparse between beams, and the gaps move as the drone
  moves. Any error metric that averages over points is really reporting on the
  beams, not on the surface.
<!-- CHECK: confirm the 6 cm noise figure and where it came from. -->
- **The noise is not small relative to the signal.** Range noise is around 6 cm
  against wave amplitudes of a few meters, and aerated whitewater returns are
  worse than that.
- **The interesting quantity is a derivative.** Slope, curvature and surface
  velocity are what the physics wants — nonlinear dispersion, breaking onset. A
  method that nails elevation and produces garbage derivatives has failed.

That last point kills the obvious approach. Bin to a grid, interpolate, then take
finite differences, and the derivative field inherits every artifact of the
binning: you differentiate your interpolation scheme, not the ocean. What I wanted
was a representation whose derivatives are objects in their own right.

A coordinate network gives exactly that. It is a function of position and time
with learnable parameters, its derivatives are available analytically through the
same computational graph that produced its values, and it never commits to a grid.

### Why sinusoidal activations

The workhorse here is a sine-activated MLP, in the SIREN family of Sitzmann et al.
(2020). The reasoning is short and it is the whole reason the project works:

**The derivative of a sine network is a sine network.** With ReLU, the second
derivative is identically zero almost everywhere, so curvature is not merely
inaccurate — it does not exist in the model. With tanh you get derivatives, but
they are smooth in a way that suppresses exactly the sharp crests that carry the
wave physics. With sine, differentiating the network gives you another network of
the same class, so $\nabla^2\hat\eta$ is as well-posed an object as $\hat\eta$
itself. That is not a performance argument. It is a well-posedness argument, and
it is why the activation was picked before anything else was.

`nets/cat_vid/` holds the same architecture trained under `relu`, `tanh` and
`siren` so the comparison is a fact in the repo and not a claim in a paragraph.

### Why input scaling is a modeling decision, not a preprocessing chore

The domain is roughly 120 m × 30 m of ocean over several hundred seconds, at
coordinates near $x \approx -200$. Feed those raw into $\sin(\omega_0 x)$ and the
first layer is sampling the activation at effectively random phase — the network
starts life as noise and the spectral bias that makes SIRENs work is gone.

So the scaler is part of the model, and it ships inside the checkpoint. This has a
consequence that took me a while to appreciate: **derivatives come out in scaled
units**, and converting them back to $\mathrm{m}\,\mathrm{m}^{-1}$ or
$\mathrm{m}\,\mathrm{s}^{-1}$ means carrying a chain-rule factor of
`out_scale / in_scale`. `Modules/evaluate.py` exists to make that round trip
impossible to get wrong, and `taking_gradients.ipynb` is where I got it wrong
enough times to write `Modules/evaluate.py`.

### Why the checkpoints carry so much

Every `.pt` in `nets/` is a dict:

```python
{
  'model_state_dict', 'optimizer_state_dict', 'epoch',
  'model config',   # e.g. (3, 300, 500, 500, 300) -> input dims + hidden widths
  'scalers',        # the fitted SS objects — the model is wrong without them
  'masker',         # the exact boolean train/validation split
  'act', 'w_0',     # activation object and frequency
}
```

The `masker` is the piece I would argue for hardest. A random 80/20 split of this
point cloud is not a real held-out set. Returns along a beam are centimeters
apart, so a randomly withheld point almost always has a training point sitting
next to it: the network can pass validation by memorizing locally and still know
nothing about the surface between beams. The split leaks, and validation error
reports a number that is far too good. Storing the mask means that six months later I can
re-open a checkpoint and evaluate on *precisely* the points that network never
saw, with no possibility of a fresh `torch.rand` quietly reshuffling what
"held out" means.

Storing `scalers` and `act` in the same dict means a checkpoint is self-describing:
you never have to remember which experiment it came from to load it correctly.
That habit is the single thing in this repo that saved me the most time.

### How I tested whether any of it was real

Three tests, in increasing order of how much they told me.

**1 — Error against the instrument, not against zero.** Everything is normalized
by the lidar noise floor: `rmse(z1, z2, normalization=0.06)`. A reported error near
1 means the network agrees with the data as closely as the data agrees with
itself. Reporting a raw RMSE would have invited me to keep optimizing past the
point where I was fitting noise.

**2 — Held-out beams, not held-out points.** The strong generalization test is to
drop entire beams from training (`beams = [b for b in beams if b not in [4, 30]]`)
and ask the network to reconstruct a strip of ocean it has literally never seen.
That is a spatial-extrapolation test, and it is much harder — and much more
honest — than interpolating between neighbors.

**3 — Two networks, same data, different seeds.** `nets/consis/` holds paired
`first.pt` / `second.pt` fits over four different time windows. Their pointwise
difference is a map of where the reconstruction is determined by the observations
and where it is determined by initialization. This is the result I care about
most, because it separates *fitting the data* from *knowing the surface*: a region
where two independent fits disagree is a region where the answer was invented, and
no amount of low training error tells you that.

### The regularizers, and why most of them are commented out

`Modules/Losses.py` is a graveyard with a purpose. The recurring idea is that in
the gaps between beams the data constrains nothing, so the loss should say
something about what the surface does *where there are no observations* — which is
a function-space prior, evaluated at collocation points, in the spirit of a PINN
but without a governing equation.

Concretely: `d_loss` and `No_dx` penalize $|\partial\hat\eta/\partial x|^2$ and
$|\partial\hat\eta/\partial y|^2$ at points off the data; `spt_reg` penalizes the
cross-beam gradient $\alpha\,\overline{(\partial\hat\eta/\partial y)^2}$ on a fixed
grid at the batch's timestamp; `continuity_loss` tries to tie consecutive frames
together; the commented `c_loss` was an attempt to penalize the network's
projection onto the low-eigenvalue directions of a data covariance — i.e. push
error into the subspace the observations cannot see.

The honest outcome is that none of them beat plain MSE by enough to justify the
extra hyperparameter, once enough frames were being fit at once. Fitting a wide
time window turns out to regularize the surface better than any explicit penalty,
because the wave field has to be consistent with itself across time. The loss
curves for the $\alpha$ sweep are in `data/e_*_a_*.pt` (`e` = epochs, `a` = $\alpha$)
and the code paths still run. I left them in because "we tried this and it did not
help" is a result.

<!-- CHECK: soften or sharpen this paragraph — you know better than I do whether
     the regularizers were abandoned on evidence or on time. -->

### Windowed fits instead of one global network

`training_all.py` fits an independent network per ~200-frame window rather than
one network over the whole record. The trade-off is capacity against frequency
content: a fixed-width MLP asked to represent 350 s of a moving wave field has to
spend its spectral budget on the slow drift and has none left for individual
crests. One network per window solves a much easier problem, at the cost of not
being continuous across the seams.

Window width is the free parameter, and the two sweeps in the repo chose it
differently: `training_all.py` uses ~200-frame windows (output in
`nets/full_fit/`), while `nets/complete/` is 35 windows of 100 frames covering the
whole 350 s record. Seam continuity is a known open end, not a solved problem.

## Layout

```
Modules/            the library, such as it is
  Neural_Net.py       Net (configurable MLP), Sine, NetWrapper (freeze t, plot in 2D)
  preprocess.py       the rewrite: SS scaler, data_gen, FrameDataset, split
  pre_pro.py          the original: same job, plus plotting and an old trainer. Legacy.
  trainers.py         training loop, checkpoint-on-best, early stopping
  Losses.py           MSE, gradient penalties, the collocation-point regularizers
  gradients.py        autograd derivatives of a trained net; disk/ball samplers
  evaluate.py         Evaluator — physical coords in, physical values out
  beams.py            beam_trim: select lidar beams, drop the beam column
  stats.py            noise-normalized RMS. Scratch quality.
  old_modules/        pre-rewrite. Kept for provenance, imported by nothing.

training_all.py     the real training job: one net per time window, in a loop
backwash/           standalone one-off runs (retrain-with-regularizer, loss curves)
test.py             unrelated scratch. See "rough edges".

*.ipynb             where the thinking happened — see below
data/               saved loss curves, named by epochs and regularizer weight
nets/               checkpoints, organized by experiment
figures/            rendered PNG sequences, mostly frames for movies
plots/              one-off stills
mesh.pt             cached plotting grid
```

### The notebooks

Each one is a question I was trying to answer, not a tutorial.

| notebook | the question |
| --- | --- |
| `train_nets.ipynb` | the current training recipe, seeded and reproducible |
| `debugging.ipynb` | *is the training loop doing what I think it is?* — one frame, then two, then ten; regularizer contribution isolated from MSE |
| `errors.ipynb` | training vs validation vs held-out-beam error distributions, against a fitted normal |
| `net_plots.ipynb` | the main figure factory — surfaces, Hovmöller diagrams, derivative fields, weight distributions, movie frames |
| `taking_gradients.ipynb` | derivative estimation and the scaler chain rule; averaging $\|\nabla\hat\eta\|$ over disks of growing radius |
| `plotting.ipynb` | data beside reconstruction, and the difference field between two independent fits |
| `data.ipynb`, `data_set.ipynb` | what the raw point cloud actually looks like |

### Reading the artifact names

- `nets/complete/1900_2000.pt` — frames 1900–2000, i.e. 190–200 s at 10 Hz.
- `nets/consis/2800_2900s/{first,second}.pt` — the two-seed reliability pairs.
  <!-- CHECK: the r/s/l suffixes (250_350r, 2800_2900s, 950_1050l) — I could not
       recover what they mean. Spell it out here. -->
- `nets/cat_vid/{relu,tanh,siren}.pt` — the activation comparison.
- `nets/init_siren/mod1..mod5.pt` — the initialization sweep.
- `nets/xavier*.pt` — sine activation with Xavier initialization, which is the
  branch `training_all.py` takes (`siren=False`).
- `nets/siren_init.pt` — the SIREN initialization scheme instead.
- `*_losses_dict.pt` — the loss curve for the checkpoint of the same name.
- `data/e_10000_a_5.pt` — 10 000 epochs, regularizer weight $\alpha = 5$.

## Running it

Nothing here runs standalone. The lidar file is 1.8 GB and lives outside the
repo at `../../data/data.pt` — a list of per-frame tensors with columns
`(x, y, z, beam)`. Every script hard-codes that relative path.

Requirements: `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `cmcrameri`,
`seaborn`, `tqdm`. Python 3.12. Training assumes CUDA; evaluation is fine on CPU.

The whole pipeline, end to end:

```python
import numpy as np, torch
import Modules.preprocess as pp, Modules.trainers as tr
from Modules.Neural_Net import Net, Sine

data   = torch.load('../../data/data.pt', weights_only=True)
frames = np.arange(300, 322, 2)          # 30.0 s to 32.0 s, every 0.2 s
beams  = list(range(32))

ds = pp.data_gen(data, beams=beams, frames=frames, normalize='SS', splits=.8)
train_dl, ver_dl = ds.get_dataloaders(num_workers=4, batch_size=1)

net = Net(Sine(w_0=1), 3, 300, 500, 500, 300).to('cuda')
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

tr.training('cuda', train_dl, (3,300,500,500,300), opt, net,
            scalers=ds.get_scalers(), masker=ds.get_mask()) \
  .MSE_training(200, prints=True, saving=True, save_path='nets/run.pt')
```

One detail worth flagging, because it is not obvious: with `FrameDataset`, a
*batch* is a lidar frame. Points are binned by timestamp and the dataloader hands
back all points sharing a bin, so `batch_size=1` means one instant in time with
tens of thousands of points in it. That is deliberate — the gradient regularizers
need a spatially coherent set of points at a single time to mean anything.

## Known rough edges

I would rather list these than have someone find them.

1. **Two preprocessing modules.** `pre_pro.py` is the original, `preprocess.py` is
   the rewrite, and `evaluate.py` contains a third copy of `in_out`/`temp_data`/
   `split`. Notebooks import from whichever existed the week they were written.
   Consolidating them is the single highest-value cleanup in the repo.
2. **The patience counter in `MSE_training` only resets inside the `if saving:`
   branch.** With `saving=False` it never resets, so training always stops after
   `count` epochs regardless of progress. Every run that matters used
   `saving=True`, so no published result is affected — but it is a live bug.
3. **`trainers.stop()` is wrong.** It fits a quadratic to a window of recent
   losses, then evaluates that polynomial at the *loss value* instead of the epoch
   index (`np.polyval(coeffs, y)`), and compares a raw loss against a residual
   threshold. The spike detector was never what stopped a run; the tolerance check
   and the patience counter were.
4. **`Neural_Net.py` opens with my own warning about the initialization, and it is
   justified.** The SIREN branch scales hidden layers by $1/\omega_0^2$ where
   Sitzmann et al. use $1/\omega_0$, and uses $\sqrt{6/\mathrm{fan\_in}}$ on the
   first layer where the paper uses $1/\mathrm{fan\_in}$. Two things follow. Every
   checkpoint I have opened was trained at $\omega_0 = 1$, so the $\omega_0^2$
   discrepancy never actually bit. And the main training script passes
   `siren=False`, so the workhorse models are **sine activations with Xavier
   initialization**, not SIRENs in the strict sense — which is what `xavier*.pt`
   means. The runs that do exercise the SIREN branch are isolated in
   `nets/init_siren/` and `nets/siren_init.pt`. Fixing the init and re-running the
   $\omega_0$ sweep is the most interesting unfinished experiment in here.
5. **`.gitignore` lists `figures/*` and `nets/*`, but both were committed before I
   added it.** Ignore rules do not untrack, so the repo carries hundreds of PNGs
   and a few hundred MB of checkpoints. Left as-is: they are the record.
6. **`stats.py` imports `from beams import ...`** rather than `from Modules.beams`,
   so it only runs from inside `Modules/`, and it loads a dataset at import time.
7. **`test.py` has nothing to do with this project.** It is scratch for a
   disk-backed top-k selection pattern I needed elsewhere.
8. **Checkpoints in `nets/consis/` predate the `act` key**, so you have to know
   they were sine to load them.

## What I did about it

The Zenodo repo is the answer to this list: one set of modules, one canonical
point ordering, one script per figure with a `compute` / `plot` / `make_figure`
split so a result can be pulled apart without editing the plotting code, and a
`nets/README.md` documenting every checkpoint. The rules I would carry into the
next project, learned the expensive way here:

- **Put everything needed to reproduce an evaluation inside the checkpoint.** The
  scalers and the split mask are part of the model. This one already paid for
  itself several times over.
- **Delete the old module the day you finish the rewrite.** Keeping it "just in
  case" cost me more than rewriting it ever would have.
- **Name the experiment, not the file.** `xavier1.pt` told me nothing three weeks
  later; `nets/complete/1900_2000.pt` still tells me everything.
- **Fix the metric before optimizing it.** Normalizing by instrument noise and
  holding out whole beams changed which models looked good — a lot of tuning
  before that was tuning against a number that could not distinguish a better
  surface from a better fit to the noise.

<!-- YOURS: this last list is the part a hiring team will read most carefully.
     Cut anything you don't actually believe, and add anything you do. -->
