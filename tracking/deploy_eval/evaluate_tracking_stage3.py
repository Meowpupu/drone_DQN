"""
【Stage 3 版本（留存副本）】接近球並懸停的評估 / 錄影腳本。Stage 1、2 請用 evaluate_tracking.py。
=========================================================
執行:  python evaluate_tracking_stage3.py [模型路徑] [--stage 3] [--episodes 5] [--record]
未指定模型時使用 checkpoints_tracking/best_model.keras（不存在則取最新檔）。

--record : 錄影模式。每個決策步（0.5 秒模擬時間）切成 --subframes 小段（預設 5 -> 10 fps），
           每一小段拍一張畫面、跑 YOLO，把偵測框與資訊列畫上去，輸出連續的 mp4（每回合一支）。
           決策仍是每 0.5 秒一次，和訓練時完全相同，只是畫面被切細，所以影片是流暢的。

輸出（eval_runs/<時間>/）:
  decisions.csv  每個決策步一列：位置/速度/朝向、YOLO 看到什麼、深度、DQN 對 13 個動作的 Q 值、選了哪個動作、獎勵、結果
  frames.csv     每個影片畫面一列：畫面時間、位置、該畫面 YOLO 偵測到的所有框（--record 才有）
  ep01_<結果>.mp4 ...

注意：評估與訓練共用同一個模擬器，請先停止訓練再執行。
"""

import argparse
import csv
import glob
import math
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "checkpoints_tracking")

CLASS_COLORS = {"yellow_frame": (0, 220, 255), "pink_ball": (255, 0, 255)}   # BGR
HUD_HEIGHT = 150
VIDEO_FPS_BASE = 2.0      # 每秒模擬時間 2 個決策步 (0.5 s / 步)


def default_model():
    best = os.path.join(MODEL_DIR, "best_model.keras")
    if os.path.exists(best):
        return best
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.keras")), key=os.path.getmtime)
    return files[-1] if files else None


def det_string(dets, names):
    """[(cls, conf, cx, cy, w, h)] -> 'pink_ball:0.95@(320,180,60,60)|yellow_frame:...'"""
    return "|".join(f"{names.get(c, c)}:{conf:.2f}@({cx:.0f},{cy:.0f},{w:.0f},{h:.0f})" for c, conf, cx, cy, w, h in dets)


def draw_detections(img, dets, names):
    out = img.copy()
    for c, conf, cx, cy, w, h in dets:
        name = names.get(c, str(c))
        color = CLASS_COLORS.get(name, (0, 255, 0))
        x0, y0, x1, y1 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, max(th + 6, y0)), color, -1)
        cv2.putText(out, label, (x0 + 2, max(th + 2, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def compose(frame_bgr, dets, names, hud_lines, q_values, chosen, action_names):
    """上方：附 YOLO 框的畫面；下方：文字資訊 + 13 個動作的 Q 值長條圖（選中的標亮）。"""
    h, w = frame_bgr.shape[:2]
    canvas = np.zeros((h + HUD_HEIGHT, w, 3), np.uint8)
    canvas[:h] = draw_detections(frame_bgr, dets, names)
    y = h + 20
    for line in hud_lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    if q_values is not None:
        n = len(q_values)
        step_x, bar_w = 15, 11
        x0, base = w - n * step_x - 8, h + HUD_HEIGHT - 22
        lo, hi = float(np.min(q_values)), float(np.max(q_values))
        span = max(hi - lo, 1e-6)
        for i, q in enumerate(q_values):
            bh = int(8 + 70 * (q - lo) / span)
            color = (0, 255, 0) if i == chosen else (130, 130, 130)
            cv2.rectangle(canvas, (x0 + i * step_x, base - bh), (x0 + i * step_x + bar_w, base), color, -1)
            cv2.putText(canvas, str(i), (x0 + i * step_x, base + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    return canvas


DECISION_COLUMNS = (
    ["episode", "step", "sim_time_s", "x", "y", "z", "altitude", "speed_mps", "yaw_deg",
     "target", "dist_to_target", "yaw_err_deg", "hover_steps",
     "yolo_detections", "target_visible", "target_dx", "target_dy", "target_area_ratio",
     "depth_left", "depth_center", "depth_right"]
)
FRAME_COLUMNS = ["episode", "frame", "step", "sub", "sim_time_s", "x", "y", "z", "n_yellow_frame", "n_pink_ball", "yolo_detections"]


def run_evaluation(env, q_fn, action_names, class_names, args, outdir):
    """q_fn(state_vector) -> 長度 NUM_ACTIONS 的 Q 值。可被離線測試以假環境呼叫。"""
    n_actions = len(action_names)
    os.makedirs(outdir, exist_ok=True)
    dec_f = open(os.path.join(outdir, "decisions.csv"), "w", newline="", encoding="utf-8-sig")
    dec = csv.writer(dec_f)
    dec.writerow(DECISION_COLUMNS + [f"q{i}_{action_names[i]}" for i in range(n_actions)] +
                 ["action_idx", "action_name", "q_best", "q_gap_to_second", "reward", "done", "result"])
    frm_f = frm = None
    if args.record:
        frm_f = open(os.path.join(outdir, "frames.csv"), "w", newline="", encoding="utf-8-sig")
        frm = csv.writer(frm_f)
        frm.writerow(FRAME_COLUMNS)

    wanted_names = {"ball": "pink_ball", "gate": "yellow_frame"}
    successes = 0
    sub_dt = 0.5 / args.subframes
    try:
        for ep in range(1, args.episodes + 1):
            state = env.reset()
            done, step, total, info = False, 0, 0.0, {}
            frames_written, writer, video_path = 0, None, os.path.join(outdir, f"ep{ep:02d}_running.mp4")
            print(f"\n{'=' * 70}\n第 {ep} 回合 (Stage {args.stage})\n{'=' * 70}")

            def grab_frame_dets():
                img = env.grab_scene()
                res = env.yolo_model(img, verbose=False, conf=args.conf)[0]
                dets = [(int(b.cls[0].item()), float(b.conf[0].item()), *[float(v) for v in b.xywh[0].cpu().numpy()])
                        for b in res.boxes]
                return img, dets

            def pose_xyz():
                from airsim_env_tracking import _vec
                return _vec(env.client.simGetVehiclePose().position) + env._frame_offset

            pending = []   # 這個決策步內的 (影像, 偵測, 子段序號, 位置)

            def on_sub(k):
                img, dets = grab_frame_dets()
                pending.append((img, dets, k, pose_xyz()))

            def flush_frames(step_idx, hud_lines, q_values, chosen):
                nonlocal writer, frames_written
                for img, dets, k, p in pending:
                    t = step_idx * 0.5 + k * sub_dt
                    canvas = compose(img, dets, class_names, hud_lines, q_values, chosen, action_names)
                    if writer is None:
                        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 args.subframes * VIDEO_FPS_BASE, (canvas.shape[1], canvas.shape[0]))
                    writer.write(canvas)
                    frames_written += 1
                    n_y = sum(1 for d in dets if class_names.get(d[0]) == "yellow_frame")
                    n_b = sum(1 for d in dets if class_names.get(d[0]) == "pink_ball")
                    frm.writerow([ep, frames_written, step_idx, k, f"{t:.2f}", f"{p[0]:.2f}", f"{p[1]:.2f}", f"{p[2]:.2f}",
                                  n_y, n_b, det_string(dets, class_names)])
                pending.clear()

            while not done:
                # ---------- 觀測（DQN 看到什麼）----------
                raw = env.last_raw_state.copy()
                dets_obs = list(env.last_detections)
                pos = env._world_pos.copy()
                tgt = env._current_target()
                tgt_key = tgt["key"] if tgt else "none"
                vel = raw[14:17]
                speed = float(np.linalg.norm(vel))
                yaw_deg = math.degrees(env._yaw)
                yaw_err = math.degrees(env._yaw_error_to_target()) if tgt else 0.0
                dist_before = env._distance_to_current_target() if tgt else 0.0
                hover_before = getattr(env, "_hover_steps", 0)

                # ---------- DQN 決策 ----------
                q = np.asarray(q_fn(state), dtype=np.float64).reshape(-1)
                order = np.argsort(-q)
                action = int(order[0])
                gap = float(q[order[0]] - q[order[1]])

                # ---------- 執行並反應 ----------
                state, reward, done, info = env.step(action, substeps=args.subframes if args.record else 1,
                                                     on_substep=on_sub if args.record else None)
                total += reward
                result = info.get("result", "running") if done else ""

                row = [ep, step, f"{step * 0.5:.2f}", f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}", f"{-pos[2]:.2f}",
                       f"{speed:.3f}", f"{yaw_deg:.1f}", tgt_key, f"{dist_before:.2f}", f"{yaw_err:.1f}", hover_before,
                       det_string(dets_obs, class_names), int(raw[2] > 0), f"{raw[0]:.3f}", f"{raw[1]:.3f}", f"{raw[2]:.4f}",
                       f"{raw[7]:.2f}", f"{raw[8]:.2f}", f"{raw[9]:.2f}"]
                row += [f"{v:.3f}" for v in q]
                row += [action, action_names[action], f"{q[order[0]]:.3f}", f"{gap:.3f}", f"{reward:.3f}", int(done), result]
                dec.writerow(row)
                dec_f.flush()

                if args.record:
                    hud = [
                        f"EP {ep}  STEP {step}  t={step * 0.5:.1f}s  target: {tgt_key}   dist {dist_before:.1f}m",
                        f"ACTION: {action_names[action]} (#{action})   Q={q[order[0]]:.2f}  gap={gap:.2f}",
                        f"speed {speed:.1f} m/s  alt {-pos[2]:.1f} m  yaw_err {yaw_err:+.0f} deg  hover {hover_before}/20",
                        f"reward {reward:+.2f}  total {total:+.1f}   saw: {len(dets_obs)} box(es)",
                        f"{result}" if done else "",
                    ]
                    flush_frames(step, hud, q, action)

                print(f"Step {step:3d} | {tgt_key:5s} 距離 {dist_before:6.2f}m | 看到 {len(dets_obs)} 個框 | "
                      f"{action_names[action]:18s} (Q {q[order[0]]:7.2f}, 差 {gap:5.2f}) | R={reward:6.2f}")
                step += 1

            if writer is not None:
                writer.release()
                final = os.path.join(outdir, f"ep{ep:02d}_{info.get('result', 'unknown').split(' ')[0]}.mp4")
                os.replace(video_path, final)
            ok = info.get("result", "").startswith("success")
            successes += ok
            print(f">> 結束: {info.get('result')}  總獎勵 {total:.2f}")
    finally:
        dec_f.close()
        if frm_f:
            frm_f.close()
    print(f"\n成功率: {successes}/{args.episodes}")
    print(f"輸出資料夾: {outdir}")
    return successes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--record", action="store_true", help="錄影模式：輸出含 YOLO 框與資訊列的 mp4 與 frames.csv")
    parser.add_argument("--subframes", type=int, default=5, help="每個決策步切成幾個畫面（5 -> 10 fps）")
    parser.add_argument("--conf", type=float, default=0.25, help="畫面上 YOLO 顯示的信心門檻")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    from tensorflow.keras import models
    from airsim_env_tracking import VisionTrackingEnv, ACTION_NAMES
    from dqn_agent import DuelingCombine

    path = args.model or default_model()
    if path is None:
        print("找不到模型檔案！")
        return
    print(f"載入模型: {path}")
    model = models.load_model(path, custom_objects={"DuelingCombine": DuelingCombine})
    q_fn = lambda s: model(np.asarray(s, np.float32).reshape(1, -1), training=False)[0].numpy()

    env = VisionTrackingEnv(curriculum_stage=args.stage)
    class_names = {int(k): str(v) for k, v in env.yolo_model.names.items()}
    outdir = args.outdir or os.path.join(HERE, "eval_runs", time.strftime("%Y%m%d_%H%M%S"))
    run_evaluation(env, q_fn, ACTION_NAMES, class_names, args, outdir)
    env.close()


if __name__ == "__main__":
    main()
