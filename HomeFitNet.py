import os
import cv2
import re
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


class Config:
    csv_path = os.environ.get("HOMEFIT_CSV_PATH", "data/annotations.csv")
    frames_base_dir = os.environ.get("HOMEFIT_FRAMES_DIR", "data/frames")
    num_frames = 64
    batch_size = 4
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_subjects = [8, 12, 18, 23]

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"HomeFitNet_Train_Log_{time.strftime('%Y%m%d_%H%M%S')}.txt")


def write_log(text):
    print(text)
    with open(Config.log_file, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def calculate_metrics(preds, targets, compute_average=True):
    preds = np.array(preds)
    targets = np.array(targets)

    metrics = {'PLCC': [], 'SRCC': [], 'RL2': []}

    for i in range(4):
        p = preds[:, i]
        t = targets[:, i]

        val_range = np.max(t) - np.min(t)
        rl2 = np.mean(((p - t) / val_range) ** 2) * 100 if val_range != 0 else 0.0

        if len(np.unique(p)) > 1 and len(np.unique(t)) > 1:
            plcc, _ = pearsonr(p, t)
            srcc, _ = spearmanr(p, t)
        else:
            plcc, srcc = 0.0, 0.0

        metrics['PLCC'].append(plcc)
        metrics['SRCC'].append(srcc)
        metrics['RL2'].append(rl2)

    avg_metrics = {k: np.mean(v) for k, v in metrics.items()} if compute_average else None
    return metrics, avg_metrics


class DualViewAQADataset(Dataset):
    def __init__(self, csv_path, frames_base_dir, split='train', test_subjects=None, num_frames=64):
        self.frames_base_dir = frames_base_dir
        self.num_frames = num_frames

        self.split = split

        full_data = pd.read_csv(csv_path)

        def extract_participant_id(video_name):
            parts = video_name.split('_')
            return int(parts[1].replace('p', ''))

        full_data['p_id'] = full_data['Video_Name'].apply(extract_participant_id)
        if test_subjects is None: test_subjects = []

        if split == 'test':
            self.data = full_data[full_data['p_id'].isin(test_subjects)].copy()
        elif split == 'train':
            self.data = full_data[~full_data['p_id'].isin(test_subjects)].copy()

        self.data = self.data.reset_index(drop=True)
        print(f"[{split.upper()} Set] Loaded {len(self.data)} video samples.")

    def __len__(self):
        return len(self.data)

    def _sample_frames(self, folder_path):
        if not os.path.exists(folder_path):
            return torch.zeros((3, self.num_frames, 224, 224), dtype=torch.float32)

        valid_exts = {'.jpg', '.png', '.jpeg'}
        frames_list = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]

        def extract_number(filename):
            numbers = re.findall(r'\d+', filename)
            return int(numbers[-1]) if numbers else 0

        frames_list.sort(key=extract_number)
        frame_count = len(frames_list)
        if frame_count == 0:
            return torch.zeros((3, self.num_frames, 224, 224), dtype=torch.float32)

        indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            frame_file = os.path.join(folder_path, frames_list[idx])
            frame = cv2.imread(frame_file)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = cv2.resize(frame, (224, 224))

                frames.append(frame)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        frames = np.array(frames).transpose(3, 0, 1, 2)
        frames_tensor = torch.tensor(frames, dtype=torch.float32) / 255.0

        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        return frames_tensor

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        master_name = row['Video_Name']
        sub_name = master_name.replace('master', 'sub')
        action_id = int(master_name.split('_')[2].replace('a', ''))

        master_path = os.path.join(self.frames_base_dir, os.path.splitext(master_name)[0])
        sub_path = os.path.join(self.frames_base_dir, os.path.splitext(sub_name)[0])

        master_video = self._sample_frames(master_path)
        sub_video = self._sample_frames(sub_path)

        scores = [
            row['Q1_Score_FinalScore'] / 100.0,
            row['Q2_Score_FinalScore'] / 100.0,
            row['Q3_Score_FinalScore'] / 100.0,
            row['Q4_Score_FinalScore'] / 100.0
        ]
        return master_video, sub_video, action_id, torch.tensor(scores, dtype=torch.float32)
class LightweightMotionStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

    def forward(self, x):
        diff = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        diff = torch.cat([diff, torch.zeros_like(x[:, :, :1, :, :])], dim=2)
        diff = torch.abs(diff)

        diff_small = F.interpolate(diff, size=(diff.size(2), 112, 112), mode='trilinear', align_corners=False)
        out = self.features(diff_small)
        return out.view(out.size(0), -1)


class HomeFitNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT
        self.backbone_master = r3d_18(weights=weights)
        self.backbone_sub = r3d_18(weights=weights)

        self.backbone_master.fc = nn.Identity()
        self.backbone_sub.fc = nn.Identity()

        self.motion_stream = LightweightMotionStream()

        self.semantic_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.action_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 12)
        )

        self.fused_fc = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.q1_feature_extractor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.q1_regressor = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.q2_scorer = nn.Sequential(
            nn.Linear(256 + 64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.q3_scorer = nn.Sequential(
            nn.Linear(256 + 64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.q4_scorer = nn.Sequential(
            nn.Linear(256 + 64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, master_video, sub_video):
        feat_master = self.backbone_master(master_video)
        feat_sub = self.backbone_sub(sub_video)

        semantic_feat = torch.cat((feat_master, feat_sub), dim=1)
        semantic_feat = self.semantic_fc(semantic_feat)

        action_logits = self.action_classifier(semantic_feat)

        motion_feat = self.motion_stream(master_video)

        fused_feat = torch.cat((semantic_feat, motion_feat), dim=1)
        global_feat = self.fused_fc(fused_feat)

        q1_hidden = self.q1_feature_extractor(global_feat)
        q1_pred = self.q1_regressor(q1_hidden)
        condition_feat = torch.cat((global_feat, q1_hidden), dim=1)
        q2_pred = self.q2_scorer(condition_feat)
        q3_pred = self.q3_scorer(condition_feat)
        q4_pred = self.q4_scorer(condition_feat)

        final_scores = torch.cat((q1_pred, q2_pred, q3_pred, q4_pred), dim=1)

        return final_scores, action_logits


def train_and_eval():
    write_log(f"==========================================")
    write_log(f"Starting multitask dual-stream training | Device: {Config.device}")
    write_log(f"==========================================")

    train_dataset = DualViewAQADataset(Config.csv_path, Config.frames_base_dir, split='train',
                                       test_subjects=Config.test_subjects, num_frames=Config.num_frames)
    test_dataset = DualViewAQADataset(Config.csv_path, Config.frames_base_dir, split='test',
                                      test_subjects=Config.test_subjects, num_frames=Config.num_frames)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = HomeFitNet().to(Config.device)

    for name, param in model.backbone_master.named_parameters():
        if "stem" in name or "layer1" in name or "layer2" in name:
            param.requires_grad = False

    for name, param in model.backbone_sub.named_parameters():
        if "stem" in name or "layer1" in name or "layer2" in name:
            param.requires_grad = False
    write_log("✅ Successfully froze the low-level features in both streams (Stem, Layer1, Layer2).")
    criterion_aqa = nn.SmoothL1Loss()
    criterion_cls = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.backbone_master.parameters()), 'lr': 1e-5},
        {'params': filter(lambda p: p.requires_grad, model.backbone_sub.parameters()), 'lr': 1e-5},
        {'params': model.motion_stream.parameters(), 'lr': 1e-4},
        {'params': model.semantic_fc.parameters(), 'lr': 1e-4},
        {'params': model.action_classifier.parameters(), 'lr': 1e-4},
        {'params': model.fused_fc.parameters(), 'lr': 1e-4},
        {'params': model.q1_feature_extractor.parameters(), 'lr': 1e-4},
        {'params': model.q1_regressor.parameters(), 'lr': 1e-4},
        {'params': model.q2_scorer.parameters(), 'lr': 1e-4},
        {'params': model.q3_scorer.parameters(), 'lr': 1e-4},
        {'params': model.q4_scorer.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=1e-6)
    best_mean_srcc = -1.0

    lambda_cls = 0.04

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.epochs} Train", leave=False)
        for master_v, sub_v, action_ids, scores in train_pbar:
            master_v, sub_v = master_v.to(Config.device), sub_v.to(Config.device)
            scores = scores.to(Config.device)

            action_ids = action_ids.long().to(Config.device) - 1

            optimizer.zero_grad()

            preds, cls_logits = model(master_v, sub_v)

            loss_q1 = criterion_aqa(preds[:, 0], scores[:, 0])
            loss_q2 = criterion_aqa(preds[:, 1], scores[:, 1])
            loss_q3 = criterion_aqa(preds[:, 2], scores[:, 2])
            loss_q4 = criterion_aqa(preds[:, 3], scores[:, 3])

            loss_aqa = 1.0 * loss_q1 + 2.5 * loss_q2 + 1.0 * loss_q3 + 1.5 * loss_q4

            loss_cls = criterion_cls(cls_logits, action_ids)

            loss = loss_aqa + lambda_cls * loss_cls

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"Total Loss": f"{loss.item():.4f}"})

        scheduler.step()

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        correct_cls = 0
        total_samples = 0

        with torch.no_grad():
            for master_v, sub_v, action_ids, scores in tqdm(test_loader, desc="Eval", leave=False):
                master_v, sub_v = master_v.to(Config.device), sub_v.to(Config.device)
                scores = scores.to(Config.device)

                action_ids = action_ids.long().to(Config.device) - 1

                preds, cls_logits = model(master_v, sub_v)

                loss = criterion_aqa(preds, scores)
                val_loss += loss.item()

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(scores.cpu().numpy())

                _, predicted_labels = torch.max(cls_logits, dim=1)
                correct_cls += (predicted_labels == action_ids).sum().item()
                total_samples += action_ids.size(0)

        ind_metrics, avg_metrics = calculate_metrics(all_preds, all_targets)
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)

        val_acc = (correct_cls / total_samples) * 100.0 if total_samples > 0 else 0.0

        log_str = (
            f"\n--- Epoch [{epoch + 1}/{Config.epochs}] ---\n"
            f"Train Total Loss: {avg_train:.4f} | Val AQA SmoothL1: {avg_val:.4f}\n"
            f"🎯 Val Action Recognition Acc: {val_acc:.2f}% 🎯\n"
            f"--------------------------------------------------\n"
            f" Score |   PLCC   |   SRCC   |   R-L2   \n"
            f"--------------------------------------------------\n"
        )
        for i in range(4):
            log_str += f"  Q{i + 1}   |  {ind_metrics['PLCC'][i]:.4f}  |  {ind_metrics['SRCC'][i]:.4f}  |  {ind_metrics['RL2'][i]:.4f}\n"

        log_str += f"--------------------------------------------------\n"
        log_str += f" Mean  |  {avg_metrics['PLCC']:.4f}  |  {avg_metrics['SRCC']:.4f}  |  {avg_metrics['RL2']:.4f}\n"

        if avg_metrics['SRCC'] > best_mean_srcc:
            best_mean_srcc = avg_metrics['SRCC']
            torch.save(model.state_dict(), os.path.join(Config.log_dir, "HomeFitNet_best.pth"))
            log_str += "🌟 New Best Cascade AQA Model Saved! 🌟\n"

        write_log(log_str)


if __name__ == "__main__":
    train_and_eval()