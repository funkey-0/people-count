"""
Dashboard biểu đồ khách từ dữ liệu người đếm được.
Đọc:
  - visitor_log.csv  (từng lượt khách vào, có cột thoi_gian)
  - daily_counts.csv (tổng khách mỗi ngày)
Vẽ 2 biểu đồ: khách theo giờ trong ngày + khách theo từng ngày.
Lưu ra dashboard.png và mở cửa sổ xem.

Chạy:  py dashboard.py
"""
import csv
import os
from collections import Counter
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.size"] = 10

LOG_FILE = "visitor_log.csv"
DAILY_FILE = "daily_counts.csv"
OUT_IMG = "dashboard.png"


def read_visitor_log():
    """Trả về danh sách datetime của từng lượt khách vào."""
    times = []
    if not os.path.exists(LOG_FILE):
        return times
    with open(LOG_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                times.append(datetime.strptime(row["thoi_gian"],
                                               "%Y-%m-%d %H:%M:%S"))
            except (ValueError, KeyError):
                continue
    return times


def read_daily_counts():
    """Trả về (danh_sach_ngay, danh_sach_so_khach)."""
    days, counts = [], []
    if not os.path.exists(DAILY_FILE):
        return days, counts
    with open(DAILY_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            days.append(row["ngay"])
            try:
                counts.append(int(row["so_khach"]))
            except ValueError:
                counts.append(0)
    return days, counts


def main():
    times = read_visitor_log()
    days, daily = read_daily_counts()

    if not times and not days:
        print("Chua co du lieu. Hay chay people-counting-system.py de sinh CSV truoc.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("THONG KE KHACH - People Counting System",
                 fontsize=14, fontweight="bold")

    # --- Biểu đồ 1: khách theo giờ trong ngày (gộp mọi ngày) ---
    by_hour = Counter(t.hour for t in times)
    hours = list(range(24))
    values = [by_hour.get(h, 0) for h in hours]
    ax1.bar(hours, values, color="#2e86de")
    ax1.set_title("Khach theo gio trong ngay")
    ax1.set_xlabel("Gio")
    ax1.set_ylabel("So khach vao")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(axis="y", alpha=0.3)
    if any(values):
        peak = max(hours, key=lambda h: by_hour.get(h, 0))
        ax1.axvline(peak, color="red", linestyle="--", alpha=0.7)
        ax1.text(peak, max(values), f" cao diem {peak}h",
                 color="red", va="top")

    # --- Biểu đồ 2: khách theo từng ngày ---
    if days:
        ax2.bar(range(len(days)), daily, color="#27ae60")
        ax2.set_title("Khach theo ngay")
        ax2.set_xlabel("Ngay")
        ax2.set_ylabel("Tong khach")
        ax2.set_xticks(range(len(days)))
        ax2.set_xticklabels(days, rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3)
        for i, v in enumerate(daily):
            ax2.text(i, v, str(v), ha="center", va="bottom")
    else:
        ax2.text(0.5, 0.5, "Chua co daily_counts.csv",
                 ha="center", va="center")
        ax2.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_IMG, dpi=120)
    print(f"Da luu bieu do vao {OUT_IMG}")

    # Thống kê nhanh in ra console
    print("\n--- Tom tat ---")
    print(f"Tong luot khach ghi nhan: {len(times)}")
    if times:
        peak = max(range(24), key=lambda h: by_hour.get(h, 0))
        print(f"Gio cao diem: {peak}h ({by_hour.get(peak, 0)} khach)")
    if days:
        busiest = max(range(len(days)), key=lambda i: daily[i])
        print(f"Ngay dong nhat: {days[busiest]} ({daily[busiest]} khach)")

    plt.show()


if __name__ == "__main__":
    main()
