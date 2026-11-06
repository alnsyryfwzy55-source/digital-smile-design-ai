"""Generation proof-of-concept as a plain script (same logic as generation_poc.ipynb).

    python run_poc.py --photos photos/ --out results/            # full run (GPU)
    python run_poc.py --photos photos/ --out results/ --quick    # small run (CPU or a weak GPU)
    python run_poc.py --photos photos/ --out results/ --dry-run  # no diffusion model: checks the pipeline only

Writes results/summary.json, results/metrics.csv, results/sweep.csv, a sheet per photo
(before | design | after), the full after-photo, and results/blind_test/ for the doctor.
"""
import argparse, csv, glob, itertools, json, os, platform, random, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsd_gen as G

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = ('close-up photo of a natural smile, healthy natural teeth, realistic enamel texture, '
          'subtle translucency at the edges, natural lighting, sharp focus, high detail')
NEGATIVE = ('cartoon, painting, 3d render, deformed teeth, extra teeth, fused teeth, missing teeth, gap, braces, '
            'metal, blurry, lowres, lipstick, distorted lips, tongue')
BASES = {'A_inpaint': 'stable-diffusion-v1-5/stable-diffusion-inpainting',
         'B_base': 'stable-diffusion-v1-5/stable-diffusion-v1-5'}
CONTROLNET = 'lllyasviel/control_v11p_sd15_canny'


def match_color(gen, ref, mask, k=0.6):
    g = cv2.cvtColor(gen, cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = mask.astype(bool)
    if m.sum() < 50:
        return gen
    g[m] += k * (r[m].mean(0) - g[m].mean(0))
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


class Renderer:
    def __init__(self, dry=False):
        self.dry, self.pipes, self.cn = dry, {}, None
        if dry:
            self.device = 'none'
            return
        import torch
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32

    def pipe(self, key):
        if key in self.pipes:
            return self.pipes[key]
        from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
        self.pipes.clear()
        if self.device == 'cuda':
            self.torch.cuda.empty_cache()
        if self.cn is None:
            self.cn = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=self.dtype)
        p = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            BASES[key], controlnet=self.cn, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)
        p.scheduler = UniPCMultistepScheduler.from_config(p.scheduler.config)
        p.set_progress_bar_config(disable=True)
        self.pipes[key] = p.to(self.device)
        return self.pipes[key]

    def __call__(self, case, seg, base, strength, seed, steps, cn_scale=0.8):
        inp, crop = case['inputs'], case['crop']
        h, w = crop.shape[:2]
        if self.dry:
            gen = inp['prepaint']
        else:
            from PIL import Image
            out = self.pipe(base)(PROMPT, negative_prompt=NEGATIVE,
                                  image=Image.fromarray(inp['prepaint']), mask_image=Image.fromarray(inp['mask'] * 255),
                                  control_image=Image.fromarray(np.dstack([inp['edges']] * 3)),
                                  height=h, width=w, strength=strength, num_inference_steps=steps, guidance_scale=7.0,
                                  controlnet_conditioning_scale=cn_scale,
                                  generator=self.torch.Generator(self.device).manual_seed(seed)).images[0]
            gen = match_color(np.asarray(out.resize((w, h))), inp['prepaint'], inp['design_mask'])
        after, alpha = G.composite(crop, gen, inp['mask'])
        chk = G.check_generated(seg, crop, after, inp, case['mouth_poly'])
        return after, alpha, chk


def save_rgb(path, im, q=92):
    cv2.imwrite(path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])


REPO_RAW = 'https://raw.githubusercontent.com/alnsyryfwzy55-source/digital-smile-design-ai/main/docs/'
MODEL_URLS = {'teeth_unet_v2.onnx': [REPO_RAW + 'teeth_unet_v2.onnx'],
              'face_landmarker.task': [REPO_RAW + 'vendor/face_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task']}


def ensure_models():
    """Model files live in models/; if missing, fetch them from the project repository."""
    import urllib.request
    d = os.path.join(HERE, 'models'); os.makedirs(d, exist_ok=True)
    for name, url in MODEL_URLS.items():
        p = os.path.join(d, name)
        if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
            print('downloading', name, flush=True)
            for u in url:                                   # first working source wins (official MediaPipe as fallback)
                try:
                    urllib.request.urlretrieve(u, p); break
                except Exception as e:
                    print('download failed', u, e)
    assert os.path.getsize(os.path.join(d, 'teeth_unet_v2.onnx')) > 10_000_000, 'teeth_unet_v2.onnx looks wrong'


def main(argv=None, face=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--photos', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--quick', action='store_true', help='5 photos, one setting, 1 seed, 20 steps, width 512')
    ap.add_argument('--dry-run', action='store_true', help='skip the diffusion model (pipeline check only)')
    ap.add_argument('--shape', default='square'); ap.add_argument('--proportion', default='red70'); ap.add_argument('--shade', default='B1')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    ensure_models()
    face = face or G.FaceFinder(os.path.join(HERE, 'models', 'face_landmarker.task'))
    seg = G.Segmenter(os.path.join(HERE, 'models', 'teeth_unet_v2.onnx'))
    photos = sorted(p for p in glob.glob(os.path.join(a.photos, '**', '*'), recursive=True)
                    if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
    out_w = 512 if a.quick else 768
    steps = 20 if a.quick else 30

    cases, skipped = [], []
    for p in photos:
        try:
            rgb = G.load_rgb(p)
            r = G.analyse(rgb, face, seg, shape=a.shape, proportion=a.proportion, shade=a.shade, out_w=out_w)
        except Exception as e:                                   # never hide a failure
            r = dict(ok=False, reason=f'error: {e}')
        if not r.get('ok'):
            skipped.append(dict(photo=os.path.basename(p), reason=str(r.get('reason') or r.get('problems'))))
            continue
        r['path'], r['rgb'] = p, rgb
        cases.append(r)
    if a.quick:
        cases = cases[:5]
    print(f'{len(cases)} photos ready, {len(skipped)} skipped', flush=True)
    if not cases:
        json.dump(dict(error='no usable photos', skipped=skipped), open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
        return 1

    R = Renderer(dry=a.dry_run)
    # --- sweep: which setting keeps to the design best?
    if a.quick or a.dry_run:
        sweep_grid = [('A_inpaint', 0.6, 0)]
    else:
        sweep_grid = list(itertools.product(['A_inpaint', 'B_base'], [0.45, 0.6, 0.75], [0, 1]))
    rows = []
    for base, s, seed in sweep_grid:
        for i, c in enumerate(cases[:5]):
            t0 = time.time()
            _, _, chk = R(c, seg, base, s, seed, steps)
            rows.append(dict(base=base, strength=s, seed=seed, case=i, sec=round(time.time() - t0, 1),
                             **{k: v for k, v in chk.items() if k != 'iou'}))
            print(f'sweep {base} s={s} seed={seed} case={i} iou={chk["iou_union"]:.3f} pass={chk["pass"]} {rows[-1]["sec"]}s', flush=True)
    with open(os.path.join(a.out, 'sweep.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r['base'], r['strength']), []).append(r)
    ranking = sorted(({'base': k[0], 'strength': k[1], 'pass_rate': float(np.mean([r['pass'] for r in v])),
                       'iou_union': float(np.mean([r['iou_union'] for r in v])), 'sec_per_image': float(np.mean([r['sec'] for r in v]))}
                      for k, v in groups.items()), key=lambda d: (-d['pass_rate'], -d['iou_union']))
    best = ranking[0]
    print('best setting:', best, flush=True)

    # --- best setting on every photo, several seeds, keep the one closest to the design
    seeds = [0] if (a.quick or a.dry_run) else [0, 1, 2]
    results = []
    for i, c in enumerate(cases):
        cand = None
        for seed in seeds:
            after, alpha, chk = R(c, seg, best['base'], best['strength'], seed, steps)
            if cand is None or chk['iou_union'] > cand[2]['iou_union']:
                cand = (after, alpha, chk, seed)
        after, alpha, chk, seed = cand
        name = f'case_{i:02d}'
        save_rgb(os.path.join(a.out, f'{name}_sheet.jpg'), np.concatenate([c['crop'], G.overlay(c['crop'], c['design']), after], 1))
        save_rgb(os.path.join(a.out, f'{name}_after_full.jpg'), G.paste_back(c['rgb'], after, c['M'], alpha))
        save_rgb(os.path.join(a.out, f'{name}_after_crop.jpg'), after)
        save_rgb(os.path.join(a.out, f'{name}_before_crop.jpg'), c['crop'])
        results.append(dict(case=name, photo=os.path.basename(c['path']), seed=seed,
                            **{k: v for k, v in chk.items() if k != 'iou'}, iou_per_tooth=json.dumps(chk['iou'])))
        print(f'{name} iou={chk["iou_union"]:.3f} pass={chk["pass"]}', flush=True)
    with open(os.path.join(a.out, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)

    # --- blind test kit: real and generated mouth crops, shuffled under random names
    bd = os.path.join(a.out, 'blind_test'); os.makedirs(bd, exist_ok=True)
    items = [('real', r['case']) for r in results] + [('generated', r['case']) for r in results]
    random.Random(7).shuffle(items)
    key = []
    for k, (kind, case) in enumerate(items):
        src = os.path.join(a.out, f'{case}_{"before" if kind == "real" else "after"}_crop.jpg')
        fn = f'img_{k:03d}.jpg'; cv2.imwrite(os.path.join(bd, fn), cv2.imread(src)); key.append((fn, kind, case))
    with open(os.path.join(a.out, 'blind_key_DO_NOT_SHARE.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([('file', 'kind', 'case')] + key)
    with open(os.path.join(bd, 'rating_sheet.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['file', 'realism_1_to_5', 'looks_generated_yes_no', 'notes'])
        for fn, _, _ in key: w.writerow([fn, '', '', ''])

    env = dict(python=platform.python_version(), platform=platform.platform(), device=R.device)
    if not a.dry_run:
        import torch, diffusers
        env.update(torch=torch.__version__, diffusers=diffusers.__version__,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    summary = dict(mode='dry-run' if a.dry_run else ('quick' if a.quick else 'full'), env=env,
                   photos_total=len(photos), photos_used=len(results), skipped=skipped,
                   best_setting=best, sweep_ranking=ranking,
                   pass_rate=float(np.mean([r['pass'] for r in results])),
                   iou_union_mean=float(np.mean([r['iou_union'] for r in results])),
                   iou_min_mean=float(np.mean([r['iou_min'] for r in results])),
                   outside_change_max=float(np.max([r['outside_change'] for r in results])),
                   criterion='pass_rate >= 0.80 (automatic) + blind realism rating by the doctor',
                   minutes=round((time.time() - t_start) / 60, 1))
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('sweep_ranking', 'skipped')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
