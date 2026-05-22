# train_baseline.py 使用說明

APTOS 2019 糖尿病視網膜病變分級 baseline：以 **ResNet50** 做迴歸，訓練時使用 **MSE**，推論時透過 **threshold** 將連續預測轉成整數標籤 `0–4`。

---

## 1. 檔案說明

本專案只需繳交／執行單一腳本：

| 檔案 | 說明 |
|------|------|
| `train_baseline.py` | 訓練程式（模型、資料、訓練迴圈皆在此檔） |

執行後會在 `CHECKPOINT_DIR` 產生：

| 輸出 | 說明 |
|------|------|
| `best_model.pt` | 驗證 MSE 最佳時的模型權重 |
| `history.json` | 每個 epoch 的 `train_mse`、`val_mse`、`val_acc` |

---

## 2. 環境需求

### Python 套件

```
torch
torchvision
pandas
scikit-learn
Pillow
```

### 建議安裝方式（虛擬環境）

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision pandas scikit-learn Pillow
```

### 硬體

- 有 NVIDIA GPU 時會自動使用 CUDA，否則使用 CPU。
- 第一次使用 ImageNet pretrained 權重時需連線下載（除非加上 `--no-pretrained`）。

---

## 3. 資料準備

### 3.1 `train.csv`

需包含欄位：

| 欄位 | 說明 |
|------|------|
| `id_code` | 影像 ID（字串） |
| `diagnosis` | 標籤，整數 `0–4` |

範例：

```csv
id_code,diagnosis
000c1434d8d7,2
001639a390f0,4
```

### 3.2 影像資料夾

- 路徑由 `IMAGE_DIR` 指定。
- 檔名格式：`{id_code}.png`（與 CSV 的 `id_code` 對應）。
- 建議為已前處理之眼底圖（例如 crop + Ben Graham、512×512 灰階 PNG）；程式會轉成 RGB 並 resize 到 `IMAGE_SIZE`（預設 512）。

---

## 4. 執行前設定

開啟 `train_baseline.py`，在檔案開頭填入三個路徑（可為相對或絕對路徑）：

```python
TRAIN_CSV = "train.csv"
IMAGE_DIR = "processed_images/processed_train_images/crop_then_ben"
CHECKPOINT_DIR = "checkpoints"
```

未填寫時執行會結束並提示：

```
Set paths at top of script: TRAIN_CSV, IMAGE_DIR, CHECKPOINT_DIR
```

---

## 5. 執行訓練

在終端機執行（路徑已填好後）：

```bash
python train_baseline.py
```

### 命令列參數（可選）

| 參數 | 預設 | 說明 |
|------|------|------|
| `--epochs` | `15` | 訓練 epoch 數 |
| `--batch-size` | `16` | batch 大小 |
| `--lr` | `1e-4` | Adam 學習率 |
| `--num-workers` | `0` | DataLoader  worker 數（Windows 若出錯可維持 0） |
| `--no-pretrained` | 關閉 | 不使用 ImageNet 預訓練權重 |

範例：

```bash
python train_baseline.py --epochs 20 --batch-size 8 --lr 5e-5
```

### 訓練時畫面輸出

每個 epoch 會印出：

```
epoch 3/15  train_mse=0.8234  val_mse=0.9012  val_acc=0.6548
  saved checkpoints/best_model.pt
```

| 指標 | 意義 |
|------|------|
| `train_mse` | 訓練集 MSE（連續預測 vs 真實標籤） |
| `val_mse` | 驗證集 MSE |
| `val_acc` | 驗證集上，threshold 後標籤與真實標籤一致的比例 |

**最佳模型**依 **val_mse 最小** 儲存，而非 val_acc。

---

## 6. 模型架構與輸出

### 6.1 結構

- **Backbone**：ResNet50（預設 ImageNet pretrained）
- **輸出層**：`Linear(2048, 1)`，單一純量（連續迴歸）
- **Loss**：`MSELoss`（對 `diagnosis` 的浮點值 0–4）
- **Optimizer**：Adam
- **驗證切分**：80% train / 20% val，依 `diagnosis` 分層（`seed=42`）

### 6.2 前處理（無 data augmentation）

1. Resize 至 `IMAGE_SIZE`×`IMAGE_SIZE`（預設 512）
2. `ToTensor`
3. ImageNet `Normalize`

### 6.3 連續值 → 標籤（Threshold）

程式內建：

```python
THRESHOLDS = (1.5, 2.5, 3.5, 4.5)
```

規則：

| 連續預測 `pred` | 輸出標籤 |
|-----------------|----------|
| `pred < 1.5` | 0 |
| `pred < 2.5` | 1 |
| `pred < 3.5` | 2 |
| `pred < 4.5` | 3 |
| 其餘 | 4 |

### 6.4 推論 API（載入權重後）

```python
model = DiagnosisModel(pretrained=True)
checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    labels = model(images)              # 整數標籤 0–4（可直接用於 submission）
    continuous = model.predict_continuous(images)  # 連續值（例如 2.37）
```

| 方法 | 回傳 | 用途 |
|------|------|------|
| `model(x)` | `long`，0–4 | 最終分級標籤 |
| `model.predict_continuous(x)` | `float` | 訓練相同之連續預測 |

---

## 7. 可調超參數（腳本內常數）

在 `train_baseline.py` 頂部修改（命令列未提供的項目）：

| 常數 | 預設 | 說明 |
|------|------|------|
| `IMAGE_SIZE` | `512` | 輸入影像邊長 |
| `VAL_RATIO` | `0.2` | 驗證集比例 |
| `SEED` | `42` | 隨機種子 |
| `EPOCHS` | `15` | 預設 epoch（可被 `--epochs` 覆寫） |
| `BATCH_SIZE` | `16` | 預設 batch（可被 `--batch-size` 覆寫） |
| `LR` | `1e-4` | 預設學習率 |
| `THRESHOLDS` | `(1.5, 2.5, 3.5, 4.5)` | 離散化邊界 |

---

## 8. `best_model.pt` 內容

```python
{
    "model_state_dict": ...,   # 模型權重
    "epoch": int,              # 儲存時的 epoch
    "val_mse": float,
    "val_acc": float,
    "thresholds": (1.5, 2.5, 3.5, 4.5),
}
```

---

## 9. 常見問題

### Q: 路徑填了仍說找不到檔案？

確認 `train.csv` 存在、 `IMAGE_DIR` 內有對應 `{id_code}.png`。路徑可為相對路徑（相對於**執行指令時的工作目錄**）或絕對路徑。

### Q: 只繳交 `train_baseline.py` 可以嗎？

可以。助教／使用者需自行準備資料、安裝套件，並在腳本內填好三個路徑後執行。

### Q: 為何訓練用 MSE，輸出卻是整數標籤？

訓練階段優化連續迴歸較穩定；競賽 submission 需要 0–4 整數，故在 `forward()` 以 threshold 轉換。Threshold 不參與反向傳播。

### Q: 沒有 GPU 可以跑嗎？

可以，速度較慢。可將 `--batch-size` 調小（例如 `8` 或 `4`）。

### Q: 本腳本會產生 `submission.csv` 嗎？

不會。本腳本僅負責**訓練**與儲存權重；若要 Kaggle 提交檔，需另寫推論程式載入 `best_model.pt`，對 test 影像呼叫 `model(x)` 取得標籤後寫入 CSV。

---

## 10. 快速檢查清單

- [ ] 已安裝 `torch`, `torchvision`, `pandas`, `scikit-learn`, `Pillow`
- [ ] 已填寫 `TRAIN_CSV`, `IMAGE_DIR`, `CHECKPOINT_DIR`
- [ ] `train.csv` 含 `id_code`, `diagnosis`
- [ ] 影像檔名為 `{id_code}.png`
- [ ] 執行 `python train_baseline.py`
- [ ] 確認產生 `checkpoints/best_model.pt` 與 `history.json`
