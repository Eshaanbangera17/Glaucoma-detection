
import os, random, time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import cv2
import argparse
from tqdm import tqdm

# ---------- args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/raw", help="root containing glaucoma/ and normal/ folders")
parser.add_argument("--out_model", type=str, default="models/mobilenet_from_folders.pt")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--freeze_epochs", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_sampler", action="store_true", help="use WeightedRandomSampler for training")
args = parser.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

DATA_DIR = Path(args.data_dir)
GL_DIR = DATA_DIR / "glaucoma"
NL_DIR = DATA_DIR / "normal"
assert GL_DIR.exists() and NL_DIR.exists(), f"Make folders: {GL_DIR} and {NL_DIR}"

# collect images
exts = ("*.jpg","*.jpeg","*.png","*.bmp")
def gather(p):
    files = []
    for e in exts:
        files.extend(list(p.glob(e)))
    return sorted(files)

gl_imgs = gather(GL_DIR)
nl_imgs = gather(NL_DIR)
print(f"Found {len(gl_imgs)} glaucoma and {len(nl_imgs)} normal images.")

rows = []
for p in gl_imgs:
    rows.append({"path": str(p.resolve()), "label":1})
for p in nl_imgs:
    rows.append({"path": str(p.resolve()), "label":0})

df = pd.DataFrame(rows)
# shuffle
df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

# ---------- stratified split ----------
# Keep at least 1 sample per class in val & test if possible.
y = df['label'].values
if len(np.unique(y)) == 1:
    # only one class present — fallback to simple split (will warn later)
    n = len(df)
    n_train = int(0.7*n)
    n_val = int(0.15*n)
    df.loc[:n_train-1, "split"] = "train"
    df.loc[n_train:n_train+n_val-1, "split"] = "val"
    df.loc[n_train+n_val:, "split"] = "test"
else:
    train_idx, temp_idx = train_test_split(df.index, stratify=df['label'], test_size=0.30, random_state=args.seed)
    val_idx, test_idx   = train_test_split(temp_idx, stratify=df.loc[temp_idx,'label'], test_size=0.5, random_state=args.seed)
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

os.makedirs("data", exist_ok=True)
df.to_csv("data/splits_from_folders.csv", index=False)
print("Saved data/splits_from_folders.csv")
print(df.groupby(['label','split']).size())

# ---------- Dataset ----------
class FundusCSV(Dataset):
    def __init__(self, df, train=True, img_size=224):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.img_size = img_size
        if train:
            self.tf = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(img_size, scale=(0.8,1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        else:
            self.tf = T.Compose([
                T.ToPILImage(),
                T.Resize(int(img_size*1.14)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        p = self.df.loc[idx, "path"]
        y = int(self.df.loc[idx, "label"])
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError(f"Cannot read {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tf(img)
        return x, torch.tensor(y, dtype=torch.long)

# dataloaders
train_df = df[df.split=="train"].reset_index(drop=True)
val_df   = df[df.split=="val"].reset_index(drop=True)
test_df  = df[df.split=="test"].reset_index(drop=True)

train_ds = FundusCSV(train_df, train=True, img_size=args.img_size)
val_ds   = FundusCSV(val_df, train=False, img_size=args.img_size)
test_ds  = FundusCSV(test_df, train=False, img_size=args.img_size)

# sampler for imbalance
def make_sampler(df):
    counts = df['label'].value_counts().to_dict()
    weights = df['label'].map(lambda l: 1.0 / counts[l]).values
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

sampler = make_sampler(train_df) if args.use_sampler else None

# Use num_workers=0 on Windows/CPU to avoid DataLoader issues
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=(sampler is None),
                          sampler=sampler, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# model
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
in_f = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_f, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,args.epochs))

# freeze backbone
def set_requires_grad(module, req):
    for p in module.parameters():
        p.requires_grad = req

if args.freeze_epochs > 0:
    set_requires_grad(model.features, False)
    print(f"Backbone frozen for first {args.freeze_epochs} epochs.")

best_auc = 0.0
os.makedirs(Path(args.out_model).parent, exist_ok=True)

for epoch in range(1, args.epochs+1):
    t0 = time.time()
    model.train()
    train_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Train E{epoch}"):
        xb = xb.to(device); yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    if len(train_loader.dataset) > 0:
        train_loss /= len(train_loader.dataset)
    else:
        train_loss = 0.0

    if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
        set_requires_grad(model.features, True)
        print("[INFO] Unfroze backbone; fine-tuning entire model.")

    # validation
    model.eval()
    ys, ps, val_loss = [], [], 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            ys.extend(yb.cpu().numpy().tolist()); ps.extend(probs.tolist())
    if len(val_loader.dataset) > 0:
        val_loss /= len(val_loader.dataset)

    auc = 0.5
    try:
        if len(set(ys))>1:
            auc = roc_auc_score(ys, ps)
    except Exception:
        auc = 0.5

    elapsed = time.time() - t0
    print(f"Epoch {epoch}/{args.epochs} time:{elapsed:.1f}s train_loss:{train_loss:.4f} val_loss:{val_loss:.4f} val_auc:{auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save({"state_dict": model.state_dict(), "epoch": epoch, "auc": auc}, args.out_model)
        print(f"[INFO] Saved best model (AUC={auc:.4f}) to {args.out_model}")

    scheduler.step()

print("Training finished. Best val AUC:", best_auc)

# Evaluate on test set using best saved model (if available)
print("Running final eval on test set...")
ckpt = torch.load(args.out_model, map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
ys, ps = [], []
with torch.no_grad():
    for xb, yb in tqdm(test_loader, desc="Test"):
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        ys.extend(yb.numpy().tolist()); ps.extend(probs.tolist())

if len(set(ys))>1:
    final_auc = roc_auc_score(ys, ps)
    print("Test ROC AUC:", final_auc)
    preds = [1 if p>0.5 else 0 for p in ps]
    cm = confusion_matrix(ys, preds)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
else:
    print("Test set has only one class; cannot compute AUC.")

print("Done.")