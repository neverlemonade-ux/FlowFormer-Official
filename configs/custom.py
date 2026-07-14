"""
Custom FlowFormer training config
----------------------------------
Copied from: FlowFormer-Official/configs/sintel.py (drinkingcoder/FlowFormer-Official)

Purpose: do ADDITIONAL training on top of the official Sintel-finetuned
checkpoint (checkpoints/sintel.pth), rather than starting from FlyingThings
like the original sintel.py does.

Drop this in your local clone as: configs/sintel_custom.py

STILL OPEN — revisit once you've reviewed your training-data format/pipeline:
  - _CN.image_size          currently [432, 960], Sintel's own resolution.
                            Your custom data may need a different crop size.
  - _CN.max_flow            400 px/frame cutoff used to discard bad GT flow
                            during loss computation. Fine for Sintel-scale
                            motion; may need changing for your data's motion range.
  - _CN.batch_size          6, tuned for the original multi-GPU Sintel run.
                            Adjust to fit your GPU/VRAM.
  - _CN.trainer.num_steps / canonical_lr   see NOTE near the bottom.

WARNING — checkpoint overwrite risk (lives in train_FlowFormer.py, not here):
  The official train_FlowFormer.py hardcodes its final save path as
  checkpoints/{--stage}.pth. If you launch training with `--stage sintel`,
  it WILL overwrite your existing checkpoints/sintel.pth once training
  finishes — it doesn't know this config is "custom", it just uses whatever
  --stage string you pass on the command line. Two options once you're ready
  to actually run training:
    1) Back up checkpoints/sintel.pth first, or
    2) Add a new branch to train_FlowFormer.py, e.g.
       `elif args.stage == 'sintel_custom': from configs.sintel_custom import get_cfg`
       and change the final-save line to use a name that won't collide.
  Not touched here since you only asked for the config file for now.
"""

from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = 'sintel_custom'          # CHANGED from 'default'
                                     # -> logs go to logs/sintel_custom/... instead of
                                     #    logs/default/..., so they don't mix with any
                                     #    other run's logs.
_CN.suffix = 'sintel_custom'        # CHANGED from 'sintel'
                                     # -> just a label appended to the log dir name;
                                     #    changed to match `name` above for clarity.
_CN.gamma = 0.85
_CN.max_flow = 400
_CN.batch_size = 6
_CN.sum_freq = 100
_CN.val_freq = 5000000
_CN.image_size = [432, 960]
_CN.add_noise = True
_CN.critical_params = []

_CN.transformer = 'latentcostformer'
_CN.restore_ckpt = 'checkpoints/sintel.pth'   # CHANGED from 'checkpoints/things.pth'
                                               # -> this is the core change: start from
                                               #    the Sintel-finetuned weights instead
                                               #    of the FlyingThings checkpoint, so
                                               #    training continues ON TOP OF the
                                               #    Sintel-finetuned model as you asked.

# latentcostformer
_CN.latentcostformer = CN()
_CN.latentcostformer.pe = 'linear'
_CN.latentcostformer.dropout = 0.0
_CN.latentcostformer.encoder_latent_dim = 256 # in twins, this is 256
_CN.latentcostformer.query_latent_dim = 64
_CN.latentcostformer.cost_latent_input_dim = 64
_CN.latentcostformer.cost_latent_token_num = 8
_CN.latentcostformer.cost_latent_dim = 128
_CN.latentcostformer.arc_type = 'transformer'
_CN.latentcostformer.cost_heads_num = 1
# encoder
_CN.latentcostformer.pretrain = True
_CN.latentcostformer.context_concat = False
_CN.latentcostformer.encoder_depth = 3
_CN.latentcostformer.feat_cross_attn = False
_CN.latentcostformer.patch_size = 8
_CN.latentcostformer.patch_embed = 'single'
_CN.latentcostformer.no_pe = False
_CN.latentcostformer.gma = "GMA"
_CN.latentcostformer.kernel_size = 9
_CN.latentcostformer.rm_res = True
_CN.latentcostformer.vert_c_dim = 64
_CN.latentcostformer.cost_encoder_res = True
_CN.latentcostformer.cnet = 'twins'
_CN.latentcostformer.fnet = 'twins'
_CN.latentcostformer.no_sc = False
_CN.latentcostformer.only_global = False
_CN.latentcostformer.add_flow_token = True
_CN.latentcostformer.use_mlp = False
_CN.latentcostformer.vertical_conv = False

# decoder
_CN.latentcostformer.decoder_depth = 12
_CN.latentcostformer.critical_params = ['cost_heads_num', 'vert_c_dim', 'cnet', 'pretrain' , 'add_flow_token', 'encoder_depth', 'gma', 'cost_encoder_res']

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 12.5e-5     # NOT CHANGED YET -- see NOTE below
_CN.trainer.adamw_decay = 1e-5
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 120000         # NOT CHANGED YET -- see NOTE below
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'

# NOTE on canonical_lr / num_steps:
# OneCycleLR ramps the LR up from a low value, peaks near canonical_lr partway
# through, then anneals back down to ~0 over exactly num_steps steps. It's
# built to run once, start-to-finish, over the schedule length you give it.
# Since you're continuing from an already-fine-tuned checkpoint rather than
# starting fresh, re-running this same 120k-step / 12.5e-5-peak schedule
# would re-apply the full aggressive fine-tuning ramp on top of weights that
# are already converged -- likely knocking the model out of its current
# optimum before slowly annealing it back down, rather than doing a gentle
# continuation. Once you've settled on your data format and know roughly how
# much data you have, you'll probably want a shorter num_steps and a lower
# canonical_lr here. Left untouched for now since that depends on your data.

def get_cfg():
    return _CN.clone()
