import torch # Thư viện PyTorch để xây dựng và huấn luyện mô hình học sâu
from PIL import Image # Thư viện để mở và thao tác với các tệp hình ảnh
from torchvision.transforms.functional import to_tensor # Hàm chuyển đổi ảnh PIL hoặc mảng thành Tensor


class ObjDetectionDataset(torch.utils.data.Dataset): # Định nghĩa lớp Dataset tùy chỉnh kế thừa từ PyTorch
    def __init__(self, df):
        # Khởi tạo lớp với một bảng dữ liệu (DataFrame) và làm mới lại chỉ số (index)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        # Trả về tổng số lượng mẫu (dòng) có trong tập dữ liệu
        return len(self.df)

    def __getitem__(self, idx):
        # --- TODO 1: Lấy dòng dữ liệu tại vị trí idx từ dataframe ---
        # (Tại đây bạn cần dùng lệnh để truy xuất dữ liệu hàng thứ idx, ví dụ: row = self.df.iloc[idx])
        # your code here
        row = self.df.iloc[idx]
        # Mở tệp hình ảnh dựa trên đường dẫn trong cột "image_path" và chuyển sang hệ màu RGB
        img = Image.open(row["images"]).convert("RGB")
        # Lấy kích thước chiều rộng (w) và chiều cao (h) của ảnh gốc
        w, h = img.size
        # Chuyển đổi ảnh vừa mở sang dạng Tensor (dữ liệu số mà AI có thể tính toán)
        image = to_tensor(img)

        # Khởi tạo danh sách rỗng để chứa tọa độ khung bao (boxes) và nhãn (labels)
        boxes, labels = [], []
        # Mở tệp chứa nhãn tương ứng dựa trên đường dẫn trong cột "label_path"
        with open(row["labels"]) as f:
            for line in f:
                # Đọc từng dòng, tách các giá trị (class_id, x_center, y_center, width, height)
                cls, xc, yc, bw, bh = map(float, line.split())
                
                # Chuyển đổi tọa độ từ định dạng chuẩn hóa (0 đến 1) sang tọa độ điểm ảnh thực tế (pixel)
                # Tính toán x1, y1 (góc trên bên trái) và x2, y2 (góc dưới bên phải) của khung bao
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                
                # Thêm tọa độ khung bao vào danh sách
                boxes.append([x1, y1, x2, y2])
                # Thêm nhãn lớp vào danh sách (thường cộng 1 để phân biệt với lớp nền/background)
                labels.append(int(cls) + 1)

        # Đóng gói dữ liệu mục tiêu (target) vào một từ điển (dictionary)
        target = {
            # Chuyển danh sách khung bao thành Tensor kiểu số thực 32-bit
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            # Chuyển danh sách nhãn thành Tensor kiểu số nguyên 64-bit
            "labels": torch.tensor(labels, dtype=torch.int64),
            # Lưu lại chỉ số của ảnh để quản lý
            "image_id": torch.tensor([idx]),
        }
        
        # --- TODO 2: return what you need form this class ---
        # Your code here (Thường sẽ trả về cặp dữ liệu: image, target)

        return image, target