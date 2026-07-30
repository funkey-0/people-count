import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

source = sys.argv[1] if len(sys.argv) > 1 else 0
if isinstance(source, str) and source.isdigit():
    source = int(source)

LOG_FILE = "visitor_log.csv"
DAILY_FILE = "daily_counts.csv"

LINE_RATIO = 0.5
CONF_THRESHOLD = 0.5
MAX_MISSING_FRAMES = 30

CAPACITY_LIMIT = 5
REID_WINDOW = 600
REID_THRESHOLD = 0.6

MIN_BOX_AREA = 4000
IOU_THRESHOLD = 0.4


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def merge_overlapping_boxes(boxes, confs, iou_thresh):
    if len(boxes) <= 1:
        return {0: list(range(len(boxes)))}
    parent = list(range(len(boxes)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if compute_iou(boxes[i], boxes[j]) > iou_thresh:
                union(i, j)

    groups = {}
    for i in range(len(boxes)):
        root = find(i)
        groups.setdefault(root, []).append(i)
    return groups


def filter_and_merge(results, min_area, iou_thresh):
    if results.boxes is None or len(results.boxes) == 0:
        return []
    raw = []
    for box in results.boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue
        raw.append((x1, y1, x2, y2, int(box.id[0]), float(box.conf[0])))
    if not raw:
        return []

    boxes = [r[:4] for r in raw]
    groups = merge_overlapping_boxes(boxes, None, iou_thresh)

    merged = []
    for root, members in groups.items():
        best = max(members, key=lambda m: raw[m][4])
        merged.append(raw[best])
    return merged


def log_visitor(timestamp, track_id):
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["thoi_gian", "id"])
        writer.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S"), track_id])


def load_today_count(date_str):
    if not os.path.exists(LOG_FILE):
        return 0
    count = 0
    with open(LOG_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["thoi_gian"].startswith(date_str):
                count += 1
    return count


def save_daily_count(date_str, count):
    daily = {}
    if os.path.exists(DAILY_FILE):
        with open(DAILY_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                daily[row["ngay"]] = row["so_khach"]
    daily[date_str] = count
    with open(DAILY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ngay", "so_khach"])
        for day in sorted(daily):
            writer.writerow([day, daily[day]])


def color_hist(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


model = YOLO("yolov8n.pt")

if isinstance(source, int):
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"Khong mo duoc nguon video: {source}")
    sys.exit(1)

track_history = {}
counted_direction = {}
first_seen = {}
last_hist = {}
recently_left = {}
track_id_alias = {}

count_in = 0
count_out = 0
dwell_times = []

today = datetime.now().strftime("%Y-%m-%d")
today_count = load_today_count(today)
print(f"Ngay {today}: da co {today_count} khach truoc phien nay")

heatmap_acc = None
show_heatmap = False
frame_idx = 0
prev_time = time.time()
failed_reads = 0

while True:
    ret, frame = cap.read()
    if not ret:
        failed_reads += 1
        if failed_reads > 30:
            print("Khong doc duoc frame tu nguon video.")
            break
        continue
    failed_reads = 0

    now_date = datetime.now().strftime("%Y-%m-%d")
    if now_date != today:
        save_daily_count(today, today_count)
        print(f"Chot ngay {today}: {today_count} khach")
        today = now_date
        today_count = 0

    frame_idx += 1
    h, w = frame.shape[:2]
    line_y = int(h * LINE_RATIO)

    if heatmap_acc is None or heatmap_acc.shape != (h, w):
        heatmap_acc = np.zeros((h, w), dtype=np.float32)

    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml",
        classes=[0], conf=CONF_THRESHOLD, verbose=False
    )[0]

    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
    active_ids = []
    now_t = time.time()

    if results.boxes is not None:
        merged = filter_and_merge(results, MIN_BOX_AREA, IOU_THRESHOLD)
        for (x1, y1, x2, y2, raw_track_id, conf) in merged:
            track_id = track_id_alias.get(raw_track_id, raw_track_id)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(heatmap_acc, (cx, cy), 20, 1.0, -1)

            hist = color_hist(frame, x1, y1, x2, y2)

            if raw_track_id not in track_id_alias and track_id not in first_seen and hist is not None:
                best_match = None
                best_similarity = REID_THRESHOLD
                for old_id, (old_hist, left_frame, old_dir, old_first_seen, old_y) in recently_left.items():
                    if old_hist is None or frame_idx - left_frame > REID_WINDOW:
                        continue
                    similarity = cv2.compareHist(hist, old_hist, cv2.HISTCMP_CORREL)
                    if similarity > best_similarity:
                        best_match = (old_id, old_dir, old_first_seen, old_y)
                        best_similarity = similarity

                if best_match is not None:
                    old_id, old_dir, old_first_seen, old_y = best_match
                    track_id_alias[raw_track_id] = old_id
                    track_id = old_id
                    first_seen[track_id] = old_first_seen
                    counted_direction[track_id] = old_dir
                    track_history[track_id] = (old_y, frame_idx - 1)
                    recently_left.pop(old_id, None)

            active_ids.append(track_id)
            if hist is not None:
                last_hist[track_id] = hist

            if track_id not in first_seen:
                first_seen[track_id] = now_t

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            dwell = now_t - first_seen[track_id]
            cv2.putText(frame, f"{dwell:.0f}s", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

            if track_id in track_history:
                prev_y = track_history[track_id][0]
                if prev_y < line_y <= cy:
                    if counted_direction.get(track_id) != "in":
                        counted_direction[track_id] = "in"
                        count_in += 1
                        today_count += 1
                        log_visitor(datetime.now(), track_id)
                elif prev_y >= line_y > cy:
                    if counted_direction.get(track_id) != "out":
                        counted_direction[track_id] = "out"
                        count_out += 1

            track_history[track_id] = (cy, frame_idx)

    stale = [tid for tid, (_, seen) in track_history.items()
             if frame_idx - seen > MAX_MISSING_FRAMES]
    for tid in stale:
        previous_y = track_history[tid][0]
        recently_left[tid] = (last_hist.get(tid), frame_idx,
                               counted_direction.get(tid, ""),
                               first_seen.get(tid, now_t), previous_y)
        track_history.pop(tid, None)
        first_seen.pop(tid, None)
        last_hist.pop(tid, None)

    expired = [tid for tid, (_, left_frame, _, first_time, _) in recently_left.items()
               if frame_idx - left_frame > REID_WINDOW]
    for tid in expired:
        _, _, _, first_time, _ = recently_left.pop(tid)
        dwell_times.append(now_t - first_time)
        for raw_id, persistent_id in list(track_id_alias.items()):
            if persistent_id == tid:
                track_id_alias.pop(raw_id, None)

    now = time.time()
    fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
    prev_time = now

    if show_heatmap:
        heatmap_acc *= 0.99
        norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.6, colored, 0.4, 0)

    present = max(0, count_in - count_out)

    over = present > CAPACITY_LIMIT
    panel_h = 130 if over else 114
    cv2.rectangle(frame, (8, 8), (200, panel_h), (0, 0, 0), -1)
    y = 26
    cv2.putText(frame, f"Khach hom nay: {today_count}", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); y += 20
    cv2.putText(frame, f"IN: {count_in}   OUT: {count_out}", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1); y += 20

    color = (0, 0, 255) if over else (255, 200, 0)
    cv2.putText(frame, f"Dang trong: {present}/{CAPACITY_LIMIT}", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1); y += 20
    if over:
        cv2.putText(frame, "!! QUA TAI !!", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2); y += 20

    avg_dwell = sum(dwell_times) / len(dwell_times) if dwell_times else 0
    cv2.putText(frame, f"Dwell TB: {avg_dwell:.1f}s", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1); y += 18

    cv2.putText(frame, f"FPS: {fps:.1f}  [H]eatmap [ESC]", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    cv2.imshow("People Counting System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in (ord('h'), ord('H')):
        show_heatmap = not show_heatmap

cap.release()
cv2.destroyAllWindows()

save_daily_count(today, today_count)
print("\n========== TOM TAT PHIEN ==========")
print(f"IN = {count_in}, OUT = {count_out}")
print(f"Tong khach ngay {today}: {today_count} (luu {DAILY_FILE})")
if dwell_times:
    print(f"Thoi gian luu lai TB: {sum(dwell_times)/len(dwell_times):.1f}s")
