import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from HomeFitNet import HomeFitNet, DualViewAQADataset, Config, calculate_metrics


def evaluate_saved_model(weight_path):
    print(f"==========================================")
    print(f"Starting evaluation-only mode | Loading checkpoint: {weight_path}")
    print(f"==========================================")

    if not os.path.exists(weight_path):
        print(f"❌ Checkpoint not found: {weight_path}. Please verify the path.")
        return

    model = HomeFitNet().to(Config.device)
    model.load_state_dict(torch.load(weight_path, map_location=Config.device))

    model.eval()

    test_dataset = DualViewAQADataset(Config.csv_path, Config.frames_base_dir, split='test',
                                      test_subjects=Config.test_subjects, num_frames=Config.num_frames)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    all_preds = []
    all_targets = []

    print("\nStarting Evaluation...")
    with torch.no_grad():
        for master_v, sub_v, action_ids, scores in tqdm(test_loader, desc="Testing"):
            master_v, sub_v = master_v.to(Config.device), sub_v.to(Config.device)
            scores = scores.to(Config.device)

            preds, _ = model(master_v, sub_v)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())

    ind_metrics, _ = calculate_metrics(all_preds, all_targets, compute_average=False)

    print("\n" + "=" * 50)
    print(" 🏆 Final Evaluation Results on Test Set 🏆")
    print("=" * 50)
    print(f" Score |   PLCC   |   SRCC   |   R-L2   ")
    print("-" * 50)
    for i in range(4):
        print(
            f"  Q{i + 1}   |  {ind_metrics['PLCC'][i]:.4f}  |  {ind_metrics['SRCC'][i]:.4f}  |  {ind_metrics['RL2'][i]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    BEST_WEIGHT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HomeFitNet_best.pth")

    evaluate_saved_model(BEST_WEIGHT_PATH)