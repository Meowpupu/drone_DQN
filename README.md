# AirSim 無人機視覺追蹤 DQN 訓練

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `airsim_env.py` | AirSim 環境封裝（Gym 風格），含狀態/動作定義 |
| `dqn_agent.py` | TensorFlow/Keras DQN Agent |
| `train.py` | 訓練主程式 |
| `settings.json` | AirSim 設定檔（障礙物感測器 + 相機） |

## 安裝步驟

1. 把 `settings.json` 複製到 `Documents\AirSim\settings.json`（覆蓋原本的檔案，備份一下原本的內容以防萬一）
2. 安裝 Python 套件：
   ```
   pip install airsim tensorflow numpy
   ```
3. 打開 `BlocksV2.uproject`，讓模擬器跑起來、按 Play 進入模擬模式
4. 執行訓練：
   ```
   python train.py
   ```

## 狀態向量（31 維，依序）

| 區段 | 維度 | 內容 |
|---|---|---|
| Vision | 5 | dx_img, dy_img, ratio_current, ratio_initial, target_visible |
| Position | 3 | x, y, z（NED，已除以 100 正規化） |
| Obstacle | 3 | d_left, d_center, d_right（已除以 40m 正規化） |
| Quaternion | 4 | w, x, y, z |
| Linear Velocity | 3 | vx, vy, vz（已除以 10 正規化） |
| Angular Velocity | 3 | p, q, r（rad/s） |
| Linear Acceleration | 3 | ax, ay, az（已除以 20 正規化） |
| Actuation | 4 | F1, F2, F3, F4（馬達推力，已正規化） |
| Torque（估算值） | 3 | Mx, My, Mz（用角速度變化率近似，非精確物理量） |

## 動作空間（13 個離散動作）

| 索引 | 動作 | 對應指令 |
|---|---|---|
| 0 | 強力前進 | Pitch -15% |
| 1 | 精準巡航 | Pitch -5% |
| 2 | 左平移 | Roll -12% |
| 3 | 右平移 | Roll +12% |
| 4 | 爬升 | Throttle +25% |
| 5 | 降落 | Throttle -25% |
| 6 | 左旋轉 | Yaw Rate -20°/s |
| 7 | 右旋轉 | Yaw Rate +20°/s |
| 8 | 前進並左轉 | Pitch -8%, Yaw -15°/s |
| 9 | 前進並右轉 | Pitch -8%, Yaw +15°/s |
| 10 | 前進並爬升 | Pitch -10%, Throttle +15% |
| 11 | 前進並下降 | Pitch -10%, Throttle -15% |
| 12 | 定點懸停 | 全部歸零 |

## 接上 YOLO

打開 `train.py`，找到 `detect_target()` 函式，換成你的模型推論：

```python
def detect_target(image):
    results = your_yolo_model(image)
    if len(results) == 0:
        return None
    x1, y1, x2, y2 = results[0].boxes.xyxy[0]
    return {"bbox": (x1, y1, x2, y2), "img_w": image.shape[1], "img_h": image.shape[0]}
```

## 需要校正的參數（`airsim_env.py` 頂部）

實際訓練前建議先手動測試無人機對這些指令的反應，再微調數值：

- `MAX_TILT_DEG`：100% Pitch/Roll 對應的最大傾角
- `HOVER_THROTTLE`：懸停基準油門（先用小程式測試讓無人機穩定懸停的油門值）
- `THROTTLE_RANGE`：油門百分比對應的實際變化幅度
- `INERTIA_DIAG`：機身慣性矩（僅影響力矩估算特徵，非物理精確值）

## 訓練輸出

- `checkpoints/`：每 50 回合儲存一次模型（`.keras` 格式）
- `training_log.csv`：每回合的獎勵、步數、epsilon、結果紀錄，可用 Excel 或 pandas 畫學習曲線

## 已知限制

1. 目前無障礙物地圖，`d_left/center/right` 會回傳感測器的最大量測範圍（40m），除非場景中有實際物件遮擋
2. 力矩（Torque）為近似估算值，非直接量測，主要作為 RL 的輔助特徵
3. `detect_target()` 是佔位函式，需要接上實際的 YOLO 模型才能使用視覺追蹤功能
