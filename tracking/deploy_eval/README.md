# 另一台電腦驗證步驟 (Stage 1 / Stage 2)

## 檔案
- evaluate_tracking.py          階段 1/2 評估 + 錄影
- evaluate_tracking_stage3.py   階段 3 評估 + 錄影 (權重等訓練結束後再補)
- airsim_env_tracking.py, dqn_agent.py, best.pt   必須與上面兩支放同一資料夾
- checkpoints_tracking/stage1_final.keras, stage2_best.keras
- settings_eval.json   AirSim 設定 (ClockSpeed 1)，覆蓋到 Documents\AirSim\settings.json
- requirements.txt

## 步驟
1. 建 Python 3.11 虛擬環境並安裝套件 (torch 依該機 CUDA 另裝):  pip install -r requirements.txt
2. 備份該機原本的 settings.json，把 settings_eval.json 複製成 Documents\AirSim\settings.json
   (Documents 若被 OneDrive 重新導向，要放在實際位置)。
3. 重新啟動 UE 編輯器 (settings 只在編輯器啟動時讀)，開 BlocksV2 的 FlyingExampleMapV2，按 Play。
4. 先確認物件座標: 評估開頭會印 [賽道資訊]，gate1/gate2/ball 座標應與原電腦一致 (球在世界 X=-20)。
5. 在此資料夾執行:
     python evaluate_tracking.py checkpoints_tracking/stage1_final.keras --stage 1 --episodes 20
     python evaluate_tracking.py checkpoints_tracking/stage2_best.keras  --stage 2 --episodes 20
   先用 --episodes 2 試跑確認沒問題；只要統計不要影片加 --no-video。
6. 結果在 eval_runs/stageN_時間/ : summary.txt、episodes.csv、decisions.csv、frames.csv、mp4、軌跡圖。
   把整個 eval_runs 資料夾帶回來即可。

## 常見問題
- 找不到物件: 看 問題排查報告.md (需要 Play、物件 Mobility 為 Movable、重新校正)。
- ClockSpeed 改成 1 後每步約 0.7 秒，錄影流暢度較好；訓練是在 5 倍速下做的，行為應一致。
