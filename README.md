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

# This is a research journal that contains the ideas behind the paper *Continuous and Differentiable Neural Network Representations of the Surfzone Sea-Surface from Lidar Point Clouds* 

A drone-mounted multibeam lidar flew over extreme swell breaking off China Rock,
Monterey. It returns an irregular, noisy point clouds in a sequence of frames gathered at 10 Hz. The goal is
to create a coordinate based neural network that returns a continous a differentiable sea surface representaion.

I fit a coordinate based neural network $\hat\eta(x,y,t ; \theta)$ to  the lidar measured sea surface.
The trained network is the sea surface model: continuous everywhere, defined off
the beams, and differentiable in space, so
$\nabla\hat\eta$, $\nabla^2\hat\eta$ and $\partial\hat\eta/\partial t$ come out of
autograd rather than out of finite differences on an interpolated grid.

## Read this first

This containes messy ideas, deadends, undeleted comments, and
overall slop. That being said it also contains the full training pipeline, the initialization, and how I preprocessed 
and plotted some of the data. It shows how I think about writing code, and getting projects done

A zenodo repository will also be associated with this work that will contain a cleaner and more consice demonstration of the 
data and software of this paper. Also, if you want to pick my brain about how this all work please shoot me an 
email and I will try to quickly respond. I would actually suggest you do that instead of trying to cipher out some on my 
messy ideas. 


## Why continous and differential NN is a meaningfull approach

The lidar sensor returns a distribution of data that is highly irregular. Traditional gridding attempts
work, the gird points can often be distant from the data. While this remains true for the NN, it offers a surface
that can be freely evalutated and is just deteremined by learned parameters. 



### Why sinusoidal activations

The underlying network architture is a sine-activated MLP, in the sinousoudal representation network (SIREN) family
which Sitzmann et al. (2020) explores. He shows accuracy in the derivative of a SIREN 
created representation, which is the a strong inspiration for modeling the sea surface.

Sitzmann states a beautiful result, **"The derivative of a SIREN is a SIREN."** 
With ReLU, the second
derivative is zero everywhere. With tanh you get derivatives, but
they are unsteady and difficult to manage. With sine, differentiating the network gives you another network of
the same class, so $\nabla^2\hat\eta$ is a well-posed object. 

### Scaling is a required step for the training SIRENs

I create a custom  StandardScaler inside pytorch, which is a major part of the preprocessing.
It's made to work with on CUDA and the cpu. checl Module/preprocess.py

Each scaler is associated to each model, this is because of the random sampling. This could have been avoided
but I choose to do it because I felt like I wanted the testing points to not affect the scaling. 
Notice that when computing derivatives, the scaled units actually affect the values of the derivatives. 
This is why I created the StandardScalar in PyTorch, so I can just `torch.autograd` over the preprocessing, which
really saves a headache.


### Each Saved network needs a bit of info

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

Each of the 8 entries in the above dictionary describes a network and how to continue training it. 
Model_state_dict, is the raw network parameters, optimizer_state_dict is the vector describing the state/momentum of the Adam 
optimizer, epoch states how long the model had been trained fro, scalers is the custom PyTorch StandardScalers that is used for evaluation of the network,
masker defines the specific 80/20 datasplit for the particular model, act is a custom PyTorch sine modules used in the network architechture, w_0 for the architecture. 



### Extra regularization included in `Modules/Losses.py`

`Modules/Losses.py` contians 4 seperate loss functions, but only one ended up being usefull for the publication.
The first is the regular MSE loss. The second takes inspiration of PINNs, where I include the spatial derivative of the network as a 
regularization.
The third considers only the alongshore derivative of the network as a regularization. The fourth uses a covaraince matrix to 
penalize varitiaon in the alongshore in the loss fucntion. Turns out good old MSE was actually just the best. Shocker...

Concretely: `d_loss` and `No_dx` penalize $|\partial\hat\eta/\partial x|^2$ and
$|\partial\hat\eta/\partial y|^2$ at points off the data; `spt_reg` penalizes the
cross-beam gradient $\alpha\,\overline{(\partial\hat\eta/\partial y)^2}$ on a fixed
grid at the batch's timestamp; `continuity_loss` tries to tie consecutive frames
together; the commented `c_loss` was an attempt to penalize the network's
projection onto the low-eigenvalue directions of a data covariance — i.e. push
error into the subspace the observations cannot see.

The honest outcome is that none of them beat plain MSE by enough to justify the
extra hyperparameter, once enough frames were being fit at once. 
<!-- CHECK: soften or sharpen this paragraph — you know better than I do whether
     the regularizers were abandoned on evidence or on time. -->


## Layout

```
Modules/            the library that I use for many purpose
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
backwash/           I originally used this for a vm but now it just artifact.
test.py             unrelated scratch.

*.ipynb             where the thinking happened — see below
data/               saved loss curves, named by epochs and regularizer weight
nets/               checkpoints, organized by experiment
figures/            rendered PNG sequences, mostly frames for movies
plots/              one-off stills
mesh.pt             cached plotting grid
```

### The notebooks

Each one is a question I was trying to answer. I just used these as tools to 
use the underlyin`.py` files in Modules.

| notebook | the question |
| --- | --- |
| `train_nets.ipynb` | the current training |
| `debugging.ipynb` | *is the training loop doing what I think it is?* — one frame, then two, then ten; regularizer contribution isolated from MSE |
| `errors.ipynb` | training vs validation vs held-out-beam error distributions, against a fitted normal |
| `net_plots.ipynb` | the main figure factory — surfaces, Hovmöller diagrams, derivative fields, weight distributions, movie frames |
| `taking_gradients.ipynb` | anything to do with derivatives or curvature |
| `plotting.ipynb` | data beside reconstructions |
| `data.ipynb`, `data_set.ipynb` | what the raw point cloud actually looks like |


## Running it

Nothing here runs standalone. The lidar file is 6.5 GB and lives outside the
repo at `../../data/data.pt` — a list of per-frame tensors with columns
`(x, y, z, beam)`. Every script hard-codes that relative path. The data is just to big to host online, so it sits on my local.
If you want to access a chunk I will try to get you a piece. 

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


## Known rough edges

I need to list everything that is either a headache, poor practice, or can lead 
to errors.

1. **Two preprocessing modules.** `pre_pro.py` is the original, `preprocess.py` is
   the rewrite, and `evaluate.py` contains a third copy of `in_out`/`temp_data`/
   `split`. This is just ugly and unclean. A better file system would have these cleaned out and
   in better harmony.
3. **The patience counter in `MSE_training` only resets inside the `if saving:`
   branch.** With `saving=False` it never resets, so training always stops after
   `count` epochs regardless of progress. Every run that matters used
   `saving=True`, so no published result is affected — but it is a live bug.
4. **`trainers.stop()` is wrong.** It remains unused, but it needs to edited before it slots it.
5. **`Neural_Net.py` opens with a stale warning about the initialization.** The
   init is correct; the comment is left over from a version where it was not, and
   I never took it back out.
6. **`.gitignore` lists `figures/*` and `nets/*`, but both were committed before I
   added it.** Ignore rules do not untrack, so the repo carries hundreds of PNGs
   and a few hundred MB of checkpoints.
7. **`stats.py` imports `from beams import ...`** rather than `from Modules.beams`,
   so it only runs from inside `Modules/`, and it loads a dataset at import time.
8. **`test.py` has nothing to do with this project.** It is scratch for a
   disk-backed top-k selection pattern I needed elsewhere.
9. **Checkpoints in `nets/consis/` predate the `act` key**, so you have to know
   they were sine to load them.


Anyway, these all these issues could be fixed, but they did not affect the necesarry keys for the publication there was 
no need to fix them during them during publication. For the publication I used a cleaner and less informative set 
notebooks which can be accessed in the papers associated Zenodo repo. I wanted to share this repo as more of an 
insight to how I thought about the project, which I felt was fitting for my github.

## Zenodo repo note

The paper has a zenode repo that has a chunck of data, and easily runnable scripts. Again, this 
repo acts as a little gateway into my mind, but is mostly difficult to use. Look for zenodo if
you want to test this method, look at my github to see how I think.
