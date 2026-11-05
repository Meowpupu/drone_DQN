"""
Stage 1 / Stage 2 評估 + 錄影腳本
=========================================================
Stage 1：接近並碰到粉球（成功 = 碰球）      Stage 2：穿過第一個黃框（成功 = 穿框）
Stage 3（接近球並懸停）請用 evaluate_tracking_stage3.py。

執行:
  python evaluate_tracking.py checkpoints_tracking/stage1_final.keras --stage 1 --episodes 20
  python evaluate_tracking.py checkpoints_tracking/stage2_best.keras  --stage 2 --episodes 20
  加 --no-video 只輸出 CSV / 統計（速度快很多）。未指定模型時用 best_model.keras（目前是 stage 2 的權重）。

錄影：每個決策步（0.5 秒模擬時間）切成 --subframes 小段（預設 5 -> 10 fps），每段拍一張、跑 YOLO，
      畫上偵測框與資訊列（動作、Q 值長條圖、反應時間…）。決策仍是每 0.5 秒一次，和訓練相同，只是畫面被切細。

輸出（eval_runs/stageN_<時間>/）:
  decisions.csv   每個決策步一列：看到什麼（YOLO 框、深度）、13 個動作 Q 值、選擇的動作、反應時間（感知/推論/總延遲）、
                  動作造成的位移/轉向/速度變化、獎勵、結果
  episodes.csv    每回合一列：結果、步數、模擬時長、總獎勵、距離、路徑長、速度、動作切換率、穿框偏差…
  frames.csv      每個影片畫面一列：畫面時間、位置、該畫面 YOLO 偵測到的所有框
  summary.txt     成功率（含 95% 信賴區間）、各結果次數、動作分布、平均反應時間
  epNN_<結果>.mp4 錄影；trajectory_epNN.png 俯視 + 高度軌跡圖

注意：評估與訓練共用同一個模擬器，請先停止訓練再執行。
"""

import argparse
import collections
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
HUD_HEIGHT = 170
STEP_SIM_SECONDS = 0.5


def default_model():
    best = os.path.join(MODEL_DIR, "best_model.keras")
    if os.path.exists(best):
        return best
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.keras")), key=os.path.getmtime)
    return files[-1] if files else None


# ---------------------------------------------------------------- 繪圖 / 格式化
def det_string(dets, names):
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


def compose(frame_bgr, dets, names, hud_lines, q_values, chosen):
    """上方：附 YOLO 框的畫面；下方：文字資訊 + 動作 Q 值長條圖（選中的標綠）。"""
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


def save_trajectory_png(path, traj, env, title):
    """俯視 (X 北 / Y 東) + 高度 vs X。黃框畫成垂直於賽道的線段（寬 = 洞寬），球畫成圓。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from airsim_env_tracking import GATE_HOLE_WIDTH, BALL_RADIUS
    t = np.asarray(traj)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(t[:, 1], t[:, 0], "-o", ms=2.5, lw=1.2, color="tab:blue")
    ax[0].plot(t[0, 1], t[0, 0], "g^", label="start")
    ax[0].plot(t[-1, 1], t[-1, 0], "rs", label="end")
    for k, p in env.course_pos.items():
        if k.startswith("gate"):
            ax[0].plot([p[1] - GATE_HOLE_WIDTH / 2, p[1] + GATE_HOLE_WIDTH / 2], [p[0], p[0]], color="gold", lw=4)
            ax[0].text(p[1] + GATE_HOLE_WIDTH / 2 + 0.3, p[0], k, fontsize=8)
        else:
            ax[0].add_patch(plt.Circle((p[1], p[0]), BALL_RADIUS, color="m", alpha=0.6))
    ax[0].set_xlabel("Y east (m)"); ax[0].set_ylabel("X north (m)"); ax[0].set_aspect("equal"); ax[0].legend(fontsize=8)
    ax[0].set_title("top view"); ax[0].grid(alpha=0.3)
    ax[1].plot(t[:, 0], -t[:, 2], "-o", ms=2.5, lw=1.2, color="tab:blue")
    for k, p in env.course_pos.items():
        ax[1].axvline(p[0], color="gold" if k.startswith("gate") else "m", ls="--", lw=1)
    ax[1].set_xlabel("X north (m)"); ax[1].set_ylabel("altitude (m)"); ax[1].set_title("altitude"); ax[1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


# ---------------------------------------------------------------- CSV 欄位
DECISION_COLUMNS = [
    "episode", "step", "sim_time_s", "x", "y", "z", "altitude", "speed_mps", "yaw_deg",
    "target", "dist_to_target", "yaw_err_deg",
    "yolo_detections", "target_visible", "target_dx", "target_dy", "target_area_ratio",
    "depth_left", "depth_center", "depth_right",
]
TIMING_COLUMNS = ["obs_ms", "infer_ms", "reaction_ms", "step_wall_ms", "loop_ms"]
RESPONSE_COLUMNS = ["dpos_fwd_m", "dpos_right_m", "dpos_up_m", "dyaw_deg", "dspeed_mps", "dv_fwd", "dv_right", "dv_up"]
FRAME_COLUMNS = ["episode", "frame", "step", "sub", "sim_time_s", "x", "y", "z", "n_yellow_frame", "n_pink_ball", "yolo_detections"]
EPISODE_COLUMNS = [
    "episode", "stage", "result", "success", "steps", "sim_duration_s", "total_reward", "start_dist", "end_dist", "min_dist",
    "path_length_m", "mean_speed", "max_speed", "action_switch_rate", "top_actions",
    "mean_infer_ms", "mean_reaction_ms", "mean_step_wall_ms", "gate_cross_lateral_m", "gate_cross_vertical_m", "video",
]


def body_frame(vec_world, yaw):
    """世界水平向量 -> (前, 右)；z 另外處理。"""
    fwd = vec_world[0] * math.cos(yaw) + vec_world[1] * math.sin(yaw)
    right = -vec_world[0] * math.sin(yaw) + vec_world[1] * math.cos(yaw)
    return fwd, right


def run_evaluation(env, q_fn, action_names, class_names, args, outdir):
    """q_fn(state) -> 各動作 Q 值。env 可為真實環境或離線測試用的假環境。"""
    n_actions = len(action_names)
    os.makedirs(outdir, exist_ok=True)
    video = not args.no_video
    sub_dt = STEP_SIM_SECONDS / args.subframes
    rng = np.random.default_rng(args.seed)

    dec_f = open(os.path.join(outdir, "decisions.csv"), "w", newline="", encoding="utf-8-sig")
    dec = csv.writer(dec_f)
    dec.writerow(DECISION_COLUMNS + [f"q{i}_{action_names[i]}" for i in range(n_actions)] +
                 ["action_idx", "action_name", "q_best", "q_gap_to_second", "explored"] + TIMING_COLUMNS + RESPONSE_COLUMNS +
                 ["reward", "done", "result", "gate_cross_lateral_m", "gate_cross_vertical_m"])
    epi_f = open(os.path.join(outdir, "episodes.csv"), "w", newline="", encoding="utf-8-sig")
    epi = csv.writer(epi_f)
    epi.writerow(EPISODE_COLUMNS)
    frm_f = frm = None
    if video:
        frm_f = open(os.path.join(outdir, "frames.csv"), "w", newline="", encoding="utf-8-sig")
        frm = csv.writer(frm_f)
        frm.writerow(FRAME_COLUMNS)

    # 量測「感知」耗時：包住 env._get_state（影像 + YOLO + 狀態）
    obs_timer = {"ms": 0.0}
    orig_get_state = env._get_state

    def timed_get_state():
        t0 = time.perf_counter()
        s = orig_get_state()
        obs_timer["ms"] = (time.perf_counter() - t0) * 1000
        return s
    env._get_state = timed_get_state

    results, all_actions = collections.Counter(), collections.Counter()
    all_infer, all_react, all_wall, steps_per_ep, rewards, successes = [], [], [], [], [], 0
    from airsim_env_tracking import _vec
    try:
        for ep in range(1, args.episodes + 1):
            state = env.reset()
            obs_ms = obs_timer["ms"]
            done, step, total, info = False, 0, 0.0, {}
            print(f"\n{'=' * 72}\n第 {ep} 回合 (Stage {args.stage})\n{'=' * 72}")

            writer, frames_written = None, 0
            video_path = os.path.join(outdir, f"ep{ep:02d}_running.mp4")
            pending = []
            traj = [env._world_pos.copy()]
            ep_actions, speeds, infer_ms_list, react_list, wall_list = [], [], [], [], []
            start_dist = env._distance_to_current_target()
            min_dist, path_len, gate_cross = start_dist, 0.0, None

            def on_sub(k):
                img = env.grab_scene()
                res = env.yolo_model(img, verbose=False, conf=args.conf)[0]
                dets = [(int(b.cls[0].item()), float(b.conf[0].item()), *[float(v) for v in b.xywh[0].cpu().numpy()])
                        for b in res.boxes]
                p = _vec(env.client.simGetVehiclePose().position) + env._frame_offset
                pending.append((img, dets, k, p))

            def flush(step_idx, hud, q, chosen):
                nonlocal writer, frames_written
                for img, dets, k, p in pending:
                    t = step_idx * STEP_SIM_SECONDS + k * sub_dt
                    canvas = compose(img, dets, class_names, hud, q, chosen)
                    if writer is None:
                        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 args.subframes / STEP_SIM_SECONDS, (canvas.shape[1], canvas.shape[0]))
                    writer.write(canvas)
                    frames_written += 1
                    n_y = sum(1 for d in dets if class_names.get(d[0]) == "yellow_frame")
                    n_b = sum(1 for d in dets if class_names.get(d[0]) == "pink_ball")
                    frm.writerow([ep, frames_written, step_idx, k, f"{t:.2f}", f"{p[0]:.2f}", f"{p[1]:.2f}", f"{p[2]:.2f}",
                                  n_y, n_b, det_string(dets, class_names)])
                pending.clear()

            t_loop_prev = time.perf_counter()
            while not done:
                # ---------- 觀測：DQN 看到什麼 ----------
                raw = env.last_raw_state.copy()
                dets_obs = list(env.last_detections)
                pos0, yaw0 = env._world_pos.copy(), env._yaw
                vel0 = raw[14:17].copy()
                speed0 = float(np.linalg.norm(vel0))
                tgt = env._current_target()
                tgt_key = tgt["key"] if tgt else "none"
                dist0 = env._distance_to_current_target() if tgt else 0.0
                yaw_err0 = math.degrees(env._yaw_error_to_target()) if tgt else 0.0

                # ---------- DQN 決策（量測推論耗時）----------
                t0 = time.perf_counter()
                q = np.asarray(q_fn(state), dtype=np.float64).reshape(-1)
                infer_ms = (time.perf_counter() - t0) * 1000
                order = np.argsort(-q)
                action, explored = int(order[0]), 0
                if args.epsilon > 0 and rng.random() < args.epsilon:
                    action, explored = int(rng.integers(n_actions)), 1
                gap = float(q[order[0]] - q[order[1]])
                reaction_ms = obs_ms + infer_ms          # 感測開始 -> 動作產生（感知到行動延遲）

                # ---------- 執行並反應 ----------
                t_step = time.perf_counter()
                state, reward, done, info = env.step(action, substeps=args.subframes if video else 1,
                                                     on_substep=on_sub if video else None)
                step_wall_ms = (time.perf_counter() - t_step) * 1000
                obs_ms_next = obs_timer["ms"]
                total += reward
                result = info.get("result", "running") if done else ""

                # ---------- 動作造成的反應 ----------
                pos1, raw1 = env._world_pos.copy(), env.last_raw_state
                dfwd, dright = body_frame(pos1 - pos0, yaw0)
                dup = -(pos1[2] - pos0[2])
                dyaw = (math.degrees(env._yaw - yaw0) + 180) % 360 - 180
                vel1 = raw1[14:17]
                dv = vel1 - vel0
                dvf, dvr = body_frame(dv, yaw0)
                dspeed = float(np.linalg.norm(vel1)) - speed0
                gc = env.last_gate_cross
                if gc is not None:
                    gate_cross = gc

                loop_ms = (time.perf_counter() - t_loop_prev) * 1000
                t_loop_prev = time.perf_counter()

                row = [ep, step, f"{step * STEP_SIM_SECONDS:.2f}", f"{pos0[0]:.2f}", f"{pos0[1]:.2f}", f"{pos0[2]:.2f}",
                       f"{-pos0[2]:.2f}", f"{speed0:.3f}", f"{math.degrees(yaw0):.1f}", tgt_key, f"{dist0:.2f}", f"{yaw_err0:.1f}",
                       det_string(dets_obs, class_names), int(raw[2] > 0), f"{raw[0]:.3f}", f"{raw[1]:.3f}", f"{raw[2]:.4f}",
                       f"{raw[7]:.2f}", f"{raw[8]:.2f}", f"{raw[9]:.2f}"]
                row += [f"{v:.3f}" for v in q]
                row += [action, action_names[action], f"{q[order[0]]:.3f}", f"{gap:.3f}", explored]
                row += [f"{obs_ms:.1f}", f"{infer_ms:.2f}", f"{reaction_ms:.1f}", f"{step_wall_ms:.0f}", f"{loop_ms:.0f}"]
                row += [f"{dfwd:.3f}", f"{dright:.3f}", f"{dup:.3f}", f"{dyaw:.1f}", f"{dspeed:.3f}",
                        f"{dvf:.3f}", f"{dvr:.3f}", f"{-dv[2]:.3f}"]
                row += [f"{reward:.3f}", int(done), result,
                        f"{gc[0]:.2f}" if gc else "", f"{gc[1]:.2f}" if gc else ""]
                dec.writerow(row)
                dec_f.flush()

                if video:
                    hud = [
                        f"EP {ep}  STEP {step}  t={step * STEP_SIM_SECONDS:.1f}s  target: {tgt_key}   dist {dist0:.1f}m",
                        f"ACTION: {action_names[action]} (#{action})  Q={q[order[0]]:.2f}  gap={gap:.2f}" + ("  [explore]" if explored else ""),
                        f"speed {speed0:.1f} m/s  alt {-pos0[2]:.1f} m  yaw_err {yaw_err0:+.0f} deg",
                        f"reaction {reaction_ms:.0f} ms (obs {obs_ms:.0f} + dqn {infer_ms:.1f})  step {step_wall_ms:.0f} ms",
                        f"response: fwd {dfwd:+.2f} right {dright:+.2f} up {dup:+.2f} m  yaw {dyaw:+.0f} deg",
                        f"reward {reward:+.2f}  total {total:+.1f}   saw {len(dets_obs)} box(es)   {result}",
                    ]
                    flush(step, hud, q, action)

                print(f"Step {step:3d} | {tgt_key:5s} 距離 {dist0:6.2f}m | 看到 {len(dets_obs)} 框 | "
                      f"{action_names[action]:18s} (Q {q[order[0]]:7.2f}, 差 {gap:5.2f}) | "
                      f"反應 {reaction_ms:5.0f}ms | 位移 前{dfwd:+5.2f} 右{dright:+5.2f} 上{dup:+5.2f} | R={reward:6.2f}")

                ep_actions.append(action)
                speeds.append(speed0)
                infer_ms_list.append(infer_ms)
                react_list.append(reaction_ms)
                wall_list.append(step_wall_ms)
                path_len += float(np.linalg.norm(pos1 - pos0))
                traj.append(pos1)
                if tgt:
                    min_dist = min(min_dist, env._distance_to_current_target())
                obs_ms = obs_ms_next
                step += 1

            # ---------- 回合結算 ----------
            final_result = info.get("result", "unknown")
            ok = final_result.startswith("success")
            successes += ok
            results[final_result] += 1
            all_actions.update(ep_actions)
            all_infer += infer_ms_list; all_react += react_list; all_wall += wall_list
            steps_per_ep.append(step); rewards.append(total)
            switches = sum(1 for a, b in zip(ep_actions, ep_actions[1:]) if a != b)
            hist = collections.Counter(ep_actions).most_common(3)
            video_name = ""
            if writer is not None:
                writer.release()
                video_name = f"ep{ep:02d}_{final_result.split(' ')[0]}.mp4"
                os.replace(video_path, os.path.join(outdir, video_name))
            save_trajectory_png(os.path.join(outdir, f"trajectory_ep{ep:02d}.png"), traj, env,
                                f"ep{ep} stage {args.stage}: {final_result}  steps={step}  R={total:.1f}")
            epi.writerow([ep, args.stage, final_result, int(ok), step, f"{step * STEP_SIM_SECONDS:.1f}", f"{total:.2f}",
                          f"{start_dist:.2f}", f"{env._distance_to_current_target():.2f}", f"{min_dist:.2f}", f"{path_len:.2f}",
                          f"{np.mean(speeds):.2f}", f"{np.max(speeds):.2f}",
                          f"{switches / max(1, len(ep_actions) - 1):.2f}",
                          ";".join(f"{action_names[a]}x{n}" for a, n in hist),
                          f"{np.mean(infer_ms_list):.2f}", f"{np.mean(react_list):.1f}", f"{np.mean(wall_list):.0f}",
                          f"{gate_cross[0]:.2f}" if gate_cross else "", f"{gate_cross[1]:.2f}" if gate_cross else "", video_name])
            epi_f.flush()
            print(f">> 結束: {final_result}  總獎勵 {total:.2f}  步數 {step}")
    finally:
        env._get_state = orig_get_state
        for f in (dec_f, epi_f, frm_f):
            if f:
                f.close()

    n = args.episodes
    lo, hi = wilson_interval(successes, n)
    lines = [
        f"Stage {args.stage} 評估：{n} 回合（epsilon={args.epsilon}，seed={args.seed}）",
        f"成功率: {successes}/{n} = {successes / n:.0%}   95% 信賴區間 [{lo:.0%}, {hi:.0%}]",
        f"平均步數 {np.mean(steps_per_ep):.1f}（模擬 {np.mean(steps_per_ep) * STEP_SIM_SECONDS:.1f} s）   平均總獎勵 {np.mean(rewards):.1f}",
        "結果分布: " + ", ".join(f"{k} x{v}" for k, v in results.most_common()),
        "動作分布: " + ", ".join(f"{action_names[a]} {c / max(1, sum(all_actions.values())):.0%}" for a, c in all_actions.most_common()),
        f"反應時間（感知+DQN 推論）平均 {np.mean(all_react):.0f} ms；DQN 推論平均 {np.mean(all_infer):.2f} ms；"
        f"每步（含 0.5 s 模擬）平均 {np.mean(all_wall):.0f} ms",
    ]
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"輸出資料夾: {outdir}")
    return successes


def main():
    parser = argparse.ArgumentParser(description="Stage 1/2 評估與錄影")
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-video", action="store_true", help="不錄影，只輸出 CSV / 統計 / 軌跡圖")
    parser.add_argument("--subframes", type=int, default=5, help="每個決策步切成幾個畫面（5 -> 10 fps）")
    parser.add_argument("--conf", type=float, default=0.25, help="畫面上 YOLO 顯示的信心門檻")
    parser.add_argument("--epsilon", type=float, default=0.0, help="隨機動作機率（預設 0 = 純貪婪；可用來測試穩健性）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
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
    outdir = args.outdir or os.path.join(HERE, "eval_runs", f"stage{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_evaluation(env, q_fn, ACTION_NAMES, class_names, args, outdir)
    env.close()


if __name__ == "__main__":
    main()
