from args import get_args  # Nạp hàm lấy tham số từ file args.py đã tạo
import pandas as pd        # Thư viện để làm việc với bảng dữ liệu (DataFrames)
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader
import os                 # Thư viện để xử lý đường dẫn file và thư mục
from model import build_model

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Khởi tạo đối tượng args để lấy các giá trị cấu hình (đường dẫn, batch size, v.v.)
    args = get_args()

    # 1. Đọc các tệp dữ liệu bảng (Dataframes)
    # os.path.join giúp kết hợp đường dẫn thư mục và tên file một cách chính xác
    # Đọc file dữ liệu dùng để huấn luyện
    train_df = pd.read_csv(os.path.join(args.csv_dir, 'train_df.csv'))
    
    # Đọc file dữ liệu dùng để kiểm chứng (đánh giá trong khi học)
    val_df = pd.read_csv(os.path.join(args.csv_dir, 'val_df.csv'))

    # 2. (Giai đoạn tiếp theo thường là khởi tạo Dataset và DataLoader)
    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    var_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    images, targets = next(iter(train_loader))

    print("Success! Got 1 data batch.")
    # In ra các thông tin quan trọng
    print(f"Number of images in this batch: {len(images)} images")
    print(f"Size of the first image (Image number 0): {images[0].shape}")
    print(f"Label structure (target) of the first image: {targets[0]}")

    #4. Initializing the model
    model = build_model(args.backbone)



# Đảm bảo hàm main() chỉ chạy khi file này được thực thi trực tiếp
if __name__ == '__main__':
    main()