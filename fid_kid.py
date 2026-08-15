import torch, torchvision, torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
import os, glob, json, numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = Inception_V3_Weights.DEFAULT
model = inception_v3(weights=weights).to(device).eval()
model.fc = torch.nn.Identity()  # -> (N, 2048) avgpool features

prep = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def feats_from_files(files, batch=64):
    out = []
    for i in range(0, len(files), batch):
        stack = []
        for f in files[i:i+batch]:
            try:
                stack.append(prep(Image.open(f).convert('RGB')))
            except Exception:
                continue
        if not stack:
            continue
        x = torch.stack(stack, 0).to(device)
        with torch.no_grad():
            fv = model(x)
        out.append(fv.cpu().numpy())
    return np.vstack(out)

def sqrtm_mat(m):
    w, v = np.linalg.eigh(m)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.T

def fid(real, synth):
    mu_r, mu_s = real.mean(0), synth.mean(0)
    s_r = np.cov(real, rowvar=False) + 1e-6 * np.eye(real.shape[1])
    s_s = np.cov(synth, rowvar=False) + 1e-6 * np.eye(synth.shape[1])
    diff = mu_r - mu_s
    covmean = sqrtm_mat(s_r @ s_s)
    return float(diff @ diff + np.trace(s_r + s_s - 2 * covmean))

def kid(real, synth, subset=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    r = rng.choice(len(real), min(subset, len(real)), replace=False)
    s = rng.choice(len(synth), min(subset, len(synth)), replace=False)
    x, y = real[r], synth[s]
    def k(a, b):
        return (0.5 + a @ b.T) ** 3
    m, n = len(x), len(y)
    xx = k(x, x); yy = k(y, y); xy = k(x, y)
    # unbiased MMD^2 (Binkowski et al., 2018)
    kid_val = (xx.sum() - np.trace(xx)) / (m * (m - 1)) \
              + (yy.sum() - np.trace(yy)) / (n * (n - 1)) \
              - 2 * xy.mean()
    return float(kid_val)

# ---- paths ----
real_dir = '/LiKun/crop-detection/pattern/data/images'
synth_tail = '/LiKun/crop-detection/paper6_MotifDiff/synthetic'
synth_mid = '/LiKun/crop-detection/paper6_MotifDiff/synthetic_mid'

real_files = glob.glob(os.path.join(real_dir, '**', '*.jpg'), recursive=True) + glob.glob(os.path.join(real_dir, '**', '*.png'), recursive=True)
tail_files = sorted(glob.glob(os.path.join(synth_tail, '**', '*.png'), recursive=True))
mid_files = sorted(glob.glob(os.path.join(synth_mid, '**', '*.png'), recursive=True))
all_synth = tail_files + mid_files
print('counts', dict(real=len(real_files), tail=len(tail_files), mid=len(mid_files), all=len(all_synth)), flush=True)

R = feats_from_files(real_files)
Tl = feats_from_files(tail_files)
Md = feats_from_files(mid_files)
A = feats_from_files(all_synth)
print('feat shapes', R.shape, Tl.shape, Md.shape, A.shape, flush=True)

res = {
    'n_real': len(real_files), 'n_tail': len(tail_files),
    'n_mid': len(mid_files), 'n_all_synth': len(all_synth),
    'fid_all_vs_real': fid(R, A), 'kid_all_vs_real': kid(R, A),
    'fid_tail_vs_real': fid(R, Tl), 'kid_tail_vs_real': kid(R, Tl),
    'fid_mid_vs_real': fid(R, Md), 'kid_mid_vs_real': kid(R, Md),
}
print(json.dumps(res, indent=2))
with open('fid_kid.json', 'w') as f:
    json.dump(res, f, indent=2)
print('WROTE fid_kid.json', flush=True)
