# Hệ thống đếm khách ra vào bằng YOLOv8 và ByteTrack

Đồ án xây dựng hệ thống đếm người ra vào theo thời gian thực từ camera thường. Hệ thống dùng YOLOv8n để phát hiện người, ByteTrack để bám ID, đếm số lượt vượt qua một vạch ảo theo hai chiều vào và ra, đồng thời ghi nhật ký ra file CSV và vẽ biểu đồ thống kê. Chương trình chạy trực tiếp bằng Python trên CPU, nhận nguồn từ webcam, file video hoặc điện thoại làm camera IP.

## Chức năng

- Phát hiện người theo thời gian thực bằng YOLOv8n.
- Bám ID ổn định cho từng người bằng ByteTrack.
- Đếm tách bạch lượt vào và lượt ra qua vạch, mỗi người chỉ đếm một lần cho mỗi hướng.
- Hiển thị số người đang có mặt và cảnh báo khi vượt sức chứa.
- Đo thời gian mỗi người lưu lại trong khung hình (dwell time).
- Chống đếm trùng bằng tái định danh (re-ID) theo màu áo.
- Bật lớp bản đồ nhiệt (heatmap) thể hiện vùng người qua lại nhiều.
- Ghi nhật ký từng lượt và chốt tổng theo ngày ra file CSV.
- Vẽ biểu đồ khách theo giờ và theo ngày bằng Matplotlib.
- Nhận nguồn video từ webcam, file mp4 hoặc điện thoại làm camera IP.

## Mô hình

- Kiến trúc phát hiện: YOLOv8n (Ultralytics).
- Trọng số khởi tạo: huấn luyện sẵn trên bộ COCO (80 lớp).
- Lớp sử dụng: chỉ lớp `person` (class 0).
- Ngưỡng tin cậy tối thiểu: 0.5.
- Thuật toán bám vết: ByteTrack (`bytetrack.yaml`).
- Chế độ chạy: CPU, không cần huấn luyện lại.

Mô hình `yolov8n.pt` được Ultralytics tự tải về trong lần chạy đầu tiên.

## Nguồn video

Chương trình nhận nguồn qua tham số dòng lệnh:

```text
(để trống)                      → webcam mặc định (0)
1                               → camera thứ hai (ví dụ DroidCam/Iriun)
video.mp4                       → file video
http://192.168.x.x:8080/video   → điện thoại làm camera IP (app IP Webcam)
```

## Quy trình hệ thống

```text
Khung hình từ camera / video / điện thoại
      ↓
YOLOv8n phát hiện người (chỉ class 0, độ tin cậy ≥ 0.5)
      ↓
ByteTrack gán ID ổn định
      ↓
Lọc khung quá nhỏ + gộp khung chồng nhau (IoU + Union-Find)
      ↓
Tái định danh bằng histogram màu (chống đếm trùng)
      ↓
Đếm khi tâm người vượt vạch (IN / OUT)
      ↓
Ghi CSV + vẽ bảng số liệu và heatmap lên khung hình
      ↓
Dashboard biểu đồ thống kê (dashboard.py)
```

Toàn bộ luồng xử lý nằm trong một vòng lặp `while True` duy nhất — mỗi khung hình đi hết luồng rồi mới đọc khung tiếp theo.

## File chính

```text
people-counting-system.py
dashboard.py
```

- `people-counting-system.py`: chương trình đếm người theo thời gian thực.
- `dashboard.py`: đọc dữ liệu CSV và vẽ biểu đồ thống kê.

## Cách chạy

### Bước 1: Cài thư viện

```text
py -m pip install ultralytics opencv-python numpy matplotlib
```

### Bước 2: Chạy chương trình đếm

```text
py people-counting-system.py
```

Chạy với nguồn khác:

```text
py people-counting-system.py video.mp4
py people-counting-system.py 1
py people-counting-system.py "http://192.168.1.5:8080/video"
```

Trong lúc chạy:

- Phím `H`: bật/tắt lớp heatmap.
- Phím `ESC`: thoát chương trình và chốt sổ.

### Bước 3: Vẽ biểu đồ thống kê

Sau khi đã đếm và sinh ra file CSV:

```text
py dashboard.py
```

Cả hai file phải chạy trong cùng một thư mục để `dashboard.py` đọc được CSV.

## File đầu ra

```text
visitor_log.csv
daily_counts.csv
dashboard.png
```

Trong đó:

- `visitor_log.csv`: từng lượt khách vào kèm thời gian (`thoi_gian`, `id`).
- `daily_counts.csv`: tổng số khách mỗi ngày (`ngay`, `so_khach`).
- `dashboard.png`: ảnh biểu đồ do `dashboard.py` xuất ra.

Chương trình tự chốt tổng theo ngày khi thoát hoặc khi qua nửa đêm, và không mất số liệu nếu tắt rồi mở lại trong ngày.

## Cấu hình

Các tham số đặt ở đầu `people-counting-system.py`, chỉnh theo hiện trường:

| Hằng số              | Giá trị    | Vai trò                                                        |
| -------------------- | ---------- | ------------------------------------------------------------- |
| `LINE_RATIO`         | 0.5        | Vạch đếm nằm ở 50% chiều cao khung hình                       |
| `CONF_THRESHOLD`     | 0.5        | Nâng từ 0.4 lên 0.5 để bớt phát hiện nhầm                     |
| `MIN_BOX_AREA`       | 4000 px²   | Loại khung bao quá nhỏ như bàn tay, cánh tay                  |
| `IOU_THRESHOLD`      | 0.4        | Hai khung chồng nhau quá mức thì gộp làm một                  |
| `MAX_MISSING_FRAMES` | 30 khung   | Vắng ~1 giây thì chuyển người đó sang danh sách chờ re-ID     |
| `REID_WINDOW`        | 600 khung  | Nhớ người vừa mất dấu ~20 giây để nối lại ID nếu họ quay lại  |
| `REID_THRESHOLD`     | 0.6        | Độ tương quan màu tối thiểu để coi là cùng một người          |
| `CAPACITY_LIMIT`     | 5 người    | Ngưỡng bật cảnh báo quá tải trên màn hình                     |

## Các thuật toán đã sử dụng

- **Phát hiện — YOLOv8** (thư viện): mạng một giai đoạn, không anchor, kèm NMS loại hộp trùng.
- **Bám đối tượng — ByteTrack** (thư viện): dự đoán vị trí bằng bộ lọc Kalman rồi ghép với phát hiện mới theo IoU.
- **Đo chồng lấn — IoU**: tự cài trong `compute_iou()`, dùng cho bước gộp khung.
- **Gom nhóm — Union-Find**: gộp mọi cặp khung có IoU cao về một nhóm, có nén đường đi khi tìm gốc.
- **Tái định danh — Histogram HSV + tương quan**: đặc trưng màu hai chiều 30×32 bin, so bằng hệ số tương quan với ngưỡng 0.6; khớp thì gán lại đúng ID cũ để không đếm trùng.
- **Đếm — phát hiện cắt vạch**: so vị trí tâm ở hai khung liên tiếp với vạch để biết hướng đi.
- **Heatmap — tích lũy có phai dần**: mỗi khung cộng nhiệt tại tâm người rồi nhân toàn bộ ma trận với 0.99 nên vết cũ mờ dần.

## Bảng thông tin trên màn hình

```text
Khach hom nay: 27
IN: 14   OUT: 8
Dang trong: 6/5
!! QUA TAI !!
Dwell TB: 42.3s
FPS: 18.5  [H]eatmap [ESC]
```

Dòng "Dang trong" chuyển đỏ và hiện cảnh báo "QUA TAI" ngay khi số người trong phòng vượt sức chứa.

## Yêu cầu môi trường

Các thư viện chính:

```text
ultralytics
opencv-python
numpy
matplotlib
```

Cấu hình thử nghiệm: laptop Intel Core i5-1135G7, RAM 8GB, Windows 11 — chạy trên CPU với YOLOv8n.

## Hạn chế

- Hiệu năng giảm khi có hơn 10 người cùng lúc trong khung hình.
- Phụ thuộc điều kiện ánh sáng và chất lượng camera.
- Độ trễ của camera IP ảnh hưởng đến độ ổn định khi bám đối tượng.
- Re-ID chỉ dựa vào màu áo nên hai người mặc màu giống nhau có thể bị coi là một.
- Vạch đếm cố định nằm ngang, chưa xử lý được lối đi chéo hoặc nhiều cửa.

## Hướng phát triển

- Dùng YOLOv8m hoặc YOLOv8l với GPU mạnh hơn để tăng độ chính xác.
- Thay histogram màu bằng đặc trưng re-ID học sâu như OSNet.
- Cho phép vẽ vạch hoặc vùng đếm tùy ý bằng chuột ngay trên khung hình.
- Chuyển hệ thống lên giao diện web bằng Flask hoặc Django.
- Triển khai trên thiết bị nhúng như Jetson Nano hoặc Raspberry Pi.
- Tích hợp camera đa góc nhìn (multi-view tracking).
