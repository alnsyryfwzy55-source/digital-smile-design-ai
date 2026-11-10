"""Generation experiment v2: real-tooth prepaint + a light diffusion pass, judged on realism as well as geometry.

Why v2: in v1 the generator kept the flat painted blocks almost unchanged (strength 0.75 on SD 1.5) and the check
only measured "inside the design", so 86% passed while every result looked like plastic.

v2 changes
  1. prepaint from REAL teeth (tooth_bank.py): each designed tooth is a real tooth of the same FDI class from one
     labelled photo (CC BY 4.0), with its own outline and enamel texture, recoloured to the patient's own shade with
     moderate whitening (not paper white);
  2. the diffusion model only harmonises light and edges (low strength); SDXL inpainting is tried next to SD 1.5;
  3. the automatic check adds realism: texture vs the patient's real teeth, colour vs the planned shade, seams;
  4. every sheet shows: before | design | real-tooth prepaint alone | after diffusion, so the eye can judge both.

    python run_v2.py --photos photos/ --out results_v2/            # GPU (Colab T4: about 45-60 min)
    python run_v2.py --photos photos/ --out results_v2/ --dry-run  # no diffusion model (pipeline check only)
"""
import argparse, csv, json, os, platform, random, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsd_gen as G
import tooth_bank as TB
from run_poc import match_color, save_rgb, ensure_models

REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
PROMPT = ('close-up dental photo of a natural smile, natural healthy teeth, realistic enamel with subtle translucency, '
          'soft natural light, consistent lighting, sharp focus, high detail, photorealistic')
NEGATIVE = ('cartoon, painting, 3d render, plastic, deformed teeth, extra teeth, fused teeth, gap, braces, metal, '
            'blurry, lowres, oversaturated, paper white, lipstick, distorted lips, tongue, text')
MODELS = {
    'sd15': dict(base='stable-diffusion-v1-5/stable-diffusion-v1-5', cn='lllyasviel/control_v11p_sd15_canny', width=768),
    'sdxl': dict(base='diffusers/stable-diffusion-xl-1.0-inpainting-0.1', cn='diffusers/controlnet-canny-sdxl-1.0',
                 vae='madebyollin/sdxl-vae-fp16-fix', width=1024),
}


class Renderer:
    """'none' = the real-tooth prepaint as it is; 'sd15' / 'sdxl' = a diffusion pass over it inside the mask."""

    def __init__(self, dry=False):
        self.dry, self.key, self.pipe = dry, None, None
        if not dry:
            import torch
            self.torch = torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        else:
            self.device = 'none'

    def _load(self, key):
        if self.key == key:
            return self.pipe
        import gc
        self.pipe = None; gc.collect()
        if self.device == 'cuda':
            self.torch.cuda.empty_cache()
        from diffusers import ControlNetModel, UniPCMultistepScheduler
        m = MODELS[key]
        def get(cls, repo, **kw):                                  # fp16 weights when the repository has them
            if self.device == 'cuda':
                try:
                    return cls.from_pretrained(repo, torch_dtype=self.dtype, variant='fp16', **kw)
                except Exception:
                    pass
            return cls.from_pretrained(repo, torch_dtype=self.dtype, **kw)
        cn = get(ControlNetModel, m['cn'])
        if key == 'sd15':
            from diffusers import StableDiffusionControlNetInpaintPipeline as P
            p = P.from_pretrained(m['base'], controlnet=cn, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)
            p = p.to(self.device)
        else:
            from diffusers import StableDiffusionXLControlNetInpaintPipeline as P, AutoencoderKL
            vae = AutoencoderKL.from_pretrained(m['vae'], torch_dtype=self.dtype)
            p = get(P, m['base'], controlnet=cn, vae=vae)
            if self.device == 'cuda':
                p.enable_model_cpu_offload()                       # fits the 15 GB of a free T4
            else:
                p = p.to(self.device)
        p.scheduler = UniPCMultistepScheduler.from_config(p.scheduler.config)
        p.set_progress_bar_config(disable=True)
        self.key, self.pipe = key, p
        return p

    def __call__(self, case, seg, key, strength, seed, steps=30, cn_scale=0.6):
        inp, crop = case['inputs'], case['crop']
        h, w = crop.shape[:2]
        if key == 'none' or self.dry:
            gen = inp['prepaint']
        else:
            from PIL import Image
            W = MODELS[key]['width']; H = int(round(h * W / w / 8) * 8)
            img = Image.fromarray(inp['prepaint']).resize((W, H), Image.LANCZOS)
            msk = Image.fromarray((cv2.dilate(inp['mask'], np.ones((5, 5), np.uint8)) * 255).astype(np.uint8)).resize((W, H))
            ctl = Image.fromarray(np.dstack([inp['edges']] * 3)).resize((W, H))
            out = self._load(key)(PROMPT, negative_prompt=NEGATIVE, image=img, mask_image=msk, control_image=ctl,
                                  height=H, width=W, strength=strength, num_inference_steps=steps, guidance_scale=6.0,
                                  controlnet_conditioning_scale=cn_scale,
                                  generator=self.torch.Generator('cpu').manual_seed(seed)).images[0]
            gen = np.asarray(out.resize((w, h), Image.LANCZOS))
            gen = match_color(gen, inp['prepaint'], inp['design_mask'], k=0.8)     # keep the planned shade
        after, alpha = G.composite(crop, gen, inp['mask'])
        chk = G.check_generated(seg, crop, after, inp, case['mouth_poly'])
        return after, alpha, chk


def score(chk):
    """Higher is better: both checks first, then texture closest to the patient's real teeth, then geometry."""
    return (int(chk['pass']), int(chk['pass_geometry']), -abs(np.log(max(chk['texture_ratio'], 1e-3))), chk['iou_union'])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--photos', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--models', default='sd15,sdxl', help='diffusion models to compare with the prepaint alone')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    md = os.path.join(HERE, 'models'); os.makedirs(md, exist_ok=True)
    for name, src in (('teeth_unet_v2.onnx', 'docs/teeth_unet_v2.onnx'), ('face_landmarker.task', 'docs/vendor/face_landmarker.task')):
        if not os.path.exists(os.path.join(md, name)) and os.path.exists(os.path.join(REPO, src)):
            __import__('shutil').copy(os.path.join(REPO, src), os.path.join(md, name))   # from the cloned repository
    ensure_models()
    face = G.FaceFinder(os.path.join(HERE, 'models', 'face_landmarker.task'))
    seg = G.Segmenter(os.path.join(HERE, 'models', 'teeth_unet_v2.onnx'))
    bank = TB.build_bank(os.path.join(REPO, 'data', 'smile_teeth_v2.zip'))
    print(f'tooth bank: {len(bank)} sets of six real teeth', flush=True)
    photos = sorted(p for p in __import__('glob').glob(os.path.join(a.photos, '**', '*'), recursive=True)
                    if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
    cases, skipped = [], []
    for p in photos:
        try:
            rgb = G.load_rgb(p)
            r = G.analyse(rgb, face, seg, bank=bank)
        except Exception as e:
            r = dict(ok=False, reason=f'error: {e}')
        if not r.get('ok'):
            skipped.append(dict(photo=os.path.basename(p), reason=str(r.get('reason') or r.get('problems'))))
            continue
        r['path'], r['rgb'] = p, rgb
        cases.append(r)
    print(f'{len(cases)} photos ready, {len(skipped)} skipped', flush=True)
    if not cases:
        json.dump(dict(error='no usable photos', skipped=skipped), open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
        return 1

    R = Renderer(dry=a.dry_run)
    grid = [('none', 0.0)]
    if not a.dry_run:
        for k in [m for m in a.models.split(',') if m in MODELS]:
            grid += [(k, 0.3), (k, 0.45)]
    # --- sweep on up to 8 photos (one seed): which setting is both faithful and realistic?
    rows = []
    for key, s in grid:
        for i, c in enumerate(cases[:8]):
            t0 = time.time()
            try:
                _, _, chk = R(c, seg, key, s, 0)
            except Exception as e:                              # a model that fails to load or run is reported, not hidden
                print(f'sweep {key} s={s}: ERROR {e}', flush=True)
                rows.append(dict(model=key, strength=s, case=i, error=str(e)[:200])); break
            rows.append(dict(model=key, strength=s, case=i, sec=round(time.time() - t0, 1),
                             **{k: v for k, v in chk.items() if k != 'iou'}))
            print(f'sweep {key} s={s} case={i} pass={chk["pass"]} geo={chk["pass_geometry"]} tex={chk["texture_ratio"]:.2f} '
                  f'dE={chk["color_dE"]:.1f} seam={chk["seam_ratio"]:.2f} {rows[-1]["sec"]}s', flush=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('model', 'strength', 'case'), k))
    with open(os.path.join(a.out, 'sweep.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    groups = {}
    for r in rows:
        if 'error' not in r:
            groups.setdefault((r['model'], r['strength']), []).append(r)
    ranking = sorted(({'model': k[0], 'strength': k[1], 'pass_rate': float(np.mean([r['pass'] for r in v])),
                       'geometry_rate': float(np.mean([r['pass_geometry'] for r in v])),
                       'texture_ratio': float(np.median([r['texture_ratio'] for r in v])),
                       'color_dE': float(np.mean([r['color_dE'] for r in v])), 'seam_ratio': float(np.mean([r['seam_ratio'] for r in v])),
                       'iou_union': float(np.mean([r['iou_union'] for r in v])), 'sec_per_image': float(np.mean([r['sec'] for r in v]))}
                      for k, v in groups.items()), key=lambda d: (-d['pass_rate'], -d['geometry_rate'], abs(np.log(max(d['texture_ratio'], 1e-3)))))
    diff = [r for r in ranking if r['model'] != 'none']
    best = diff[0] if diff else ranking[0]                     # the sheets always show the prepaint alone as well
    print('ranking:', json.dumps(ranking, indent=1), flush=True)

    # --- every photo: prepaint alone + best diffusion setting with 2 seeds (keep the better one)
    seeds = [0] if a.dry_run else [0, 1]
    results = []
    for i, c in enumerate(cases):
        name = f'case_{i:02d}'
        pre_after, pre_alpha, pre_chk = R(c, seg, 'none', 0.0, 0)
        cand = None
        if best['model'] != 'none':
            for seed in seeds:
                after, alpha, chk = R(c, seg, best['model'], best['strength'], seed)
                if cand is None or score(chk) > score(cand[2]):
                    cand = (after, alpha, chk, seed)
        else:
            cand = (pre_after, pre_alpha, pre_chk, 0)
        after, alpha, chk, seed = cand
        save_rgb(os.path.join(a.out, f'{name}_sheet.jpg'),
                 np.concatenate([c['crop'], G.overlay(c['crop'], c['design']), pre_after, after], 1))
        save_rgb(os.path.join(a.out, f'{name}_after_full.jpg'), G.paste_back(c['rgb'], after, c['M'], alpha))
        save_rgb(os.path.join(a.out, f'{name}_prepaint_full.jpg'), G.paste_back(c['rgb'], pre_after, c['M'], pre_alpha))
        save_rgb(os.path.join(a.out, f'{name}_after_crop.jpg'), after)
        save_rgb(os.path.join(a.out, f'{name}_before_crop.jpg'), c['crop'])
        rec = dict(case=name, photo=os.path.basename(c['path']), bank_set=c.get('bank_id'), model=best['model'],
                   strength=best['strength'], seed=seed, **{k: v for k, v in chk.items() if k != 'iou'})
        rec.update({f'prepaint_{k}': v for k, v in pre_chk.items() if k in ('pass', 'texture_ratio', 'color_dE', 'seam_ratio', 'iou_union')})
        results.append(rec)
        print(f'{name} pass={chk["pass"]} tex={chk["texture_ratio"]:.2f} dE={chk["color_dE"]:.1f} seam={chk["seam_ratio"]:.2f} '
              f'iou={chk["iou_union"]:.3f}', flush=True)
    with open(os.path.join(a.out, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)

    # --- blind test for the doctor: real, prepaint and diffusion mouth crops, shuffled under random names
    bd = os.path.join(a.out, 'blind_test'); os.makedirs(bd, exist_ok=True)
    key = []
    items = []
    for r in results:
        items += [('real', r['case']), ('generated', r['case'])]
    random.Random(11).shuffle(items)
    for k, (kind, case) in enumerate(items):
        src = os.path.join(a.out, f'{case}_{"before" if kind == "real" else "after"}_crop.jpg')
        fn = f'img_{k:03d}.jpg'; cv2.imwrite(os.path.join(bd, fn), cv2.imread(src)); key.append((fn, kind, case))
    with open(os.path.join(a.out, 'blind_key_DO_NOT_SHARE.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([('file', 'kind', 'case')] + key)
    with open(os.path.join(bd, 'rating_sheet.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['file', 'realism_1_to_5', 'looks_generated_yes_no', 'notes'])
        for fn, _, _ in key: w.writerow([fn, '', '', ''])

    env = dict(python=platform.python_version(), device=R.device)
    if not a.dry_run:
        import torch, diffusers
        env.update(torch=torch.__version__, diffusers=diffusers.__version__,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    m = lambda k: float(np.mean([r[k] for r in results]))
    summary = dict(experiment='v2 real-tooth prepaint', env=env, photos_total=len(photos), photos_used=len(results), skipped=skipped,
                   best_setting=best, sweep_ranking=ranking, pass_rate=m('pass'), geometry_rate=m('pass_geometry'),
                   realism_rate=m('pass_realism'), prepaint_pass_rate=m('prepaint_pass'),
                   texture_ratio_median=float(np.median([r['texture_ratio'] for r in results])),
                   color_dE_mean=m('color_dE'), seam_ratio_mean=m('seam_ratio'), iou_union_mean=m('iou_union'),
                   outside_change_max=float(np.max([r['outside_change'] for r in results])),
                   criterion='pass_rate >= 0.80 (geometry AND realism) + blind realism rating by the doctor >= 3.5/5',
                   minutes=round((time.time() - t_start) / 60, 1))
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('sweep_ranking', 'skipped')}, indent=1))
    return 0


if __name__ == '__main__':
    import traceback
    out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else '.'
    try:
        sys.exit(main())
    except Exception:                                      # the zip always says why a run stopped
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, 'ERROR.txt'), 'w').write(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)
