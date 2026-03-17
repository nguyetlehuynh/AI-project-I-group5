import torch

class Args:
    def __init__(self):
        # --- [SỬA ĐỔI] THIẾT LẬP THIẾT BỊ (DEVICE) ---
        # Thay vì ép buộc dùng "cuda", dòng này sẽ tự kiểm tra máy local.
        # Nếu không có card đồ họa rời, nó sẽ tự chọn "cpu" để không bị báo lỗi đỏ.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- [THÊM MỚI] SỐ VÒNG LẶP (EPOCHS) ---
        # Đặt là 2 để chạy thử nghiệm quy trình (workflow) trên máy local cho nhẹ.
        # Sau này khi nộp bài hoặc dùng Google Colab, chỉ cần sửa số này lên cao hơn.
        self.epochs = 2
        
        # --- [THÊM MỚI] THÔNG SỐ HUẤN LUYỆN ---
        self.batch_size = 4  # Số lượng ảnh xử lý cùng lúc (để nhỏ để tránh treo máy CPU).
        self.lr = 0.001      # Tốc độ học (Learning Rate).
        self.img_size = 416  # Kích thước ảnh chuẩn hóa cho mô hình[cite: 1, 70].
        
        # --- [SỬA ĐỔI] ĐƯỜNG DẪN DỮ LIỆU (PATHS) ---
        # Thêm "data/" vào trước các đường dẫn để khớp với cấu trúc thư mục local.
        self.train_csv = "data/train_df.csv"
        self.val_csv = "data/val_df.csv"
        self.img_dir = "data/images/"
        self.label_dir = "data/labels/"
        
        # --- [THÊM MỚI] LƯU TRỮ KẾT QUẢ ---
        # Tên file sẽ chứa "bộ não" của AI sau khi học xong.
        self.model_save_path = "paper_model.pth"

    def __str__(self):
        return f"Using device: {self.device} | Epochs: {self.epochs} | Batch Size: {self.batch_size}"

# --- [THÊM MỚI] KHỞI TẠO INSTANCE ---
# Giúp các file main.py, dataset.py chỉ cần import "args" là dùng được ngay, rất sạch sẽ.
args = Args()