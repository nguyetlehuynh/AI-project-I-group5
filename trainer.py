from args import args
import os
import torch
import torch.optim as optim

# Tuần sau, sẽ Project Submission Phases. Upload files to Github respository cá nhân.

def validate_model(model, val_loader, device):
    """
    Hàm đánh giá mô hình trên tập dữ liệu kiểm thử (Validation).
    """
    model.eval() # Chuyển mô hình sang chế độ đánh giá (tắt Dropout, Batchnorm)
    val_loss_sum = 0.0
    val_count = 0
    
    # Tắt tính toán đạo hàm để tiết kiệm bộ nhớ và tăng tốc độ khi test
    with torch.no_grad():
        for images, targets in val_loader:
            # Đưa ảnh và nhãn (boxes, labels) vào GPU/CPU và định dạng kiểu dữ liệu
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]
            
            # Dự đoán và lấy từ điển các giá trị lỗi (loss dict)
            loss_dict = model(images, targets)
            # Tổng hợp các loại lỗi (classification, box regression,...) thành 1 số duy nhất
            loss = sum(loss_value for loss_value in loss_dict.values())
            
            # Cộng dồn lỗi để tính trung bình sau này
            val_loss_sum += loss.item() * len(images)
            val_count += len(images)

    # Tính lỗi trung bình trên toàn bộ tập validation
    val_epoch_loss = val_loss_sum / val_count
    return val_epoch_loss

def train_model(model, train_loader, val_loader, device):
    """
    Hàm chính thực hiện quá trình huấn luyện mô hình qua nhiều Epoch.
    """
    args = args() # Lấy các tham số cấu hình (lr, epochs,...)
    model = model.to(device) # Đưa mô hình vào thiết bị tính toán
    
    # Khởi tạo bộ tối ưu hóa Adam với các tham số từ args
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = float('inf') # Khởi tạo lỗi tốt nhất là vô cùng để so sánh

    for epoch in range(args.epochs):
        model.train() # Chuyển mô hình sang chế độ huấn luyện
        running_loss = 0.0
        
        for images, targets in train_loader:
            # Chuẩn bị dữ liệu cho mỗi Batch (tương tự phần validate)
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]

            optimizer.zero_grad() # Xóa đạo hàm cũ của bước trước
            loss_dict = model(images, targets) # Forward pass: Tính lỗi
            loss = sum(loss_value for loss_value in loss_dict.values())
            
            loss.backward() # Backward pass: Tính toán đạo hàm (lan truyền ngược)
            optimizer.step() # Cập nhật trọng số mô hình

            running_loss += loss.item() * len(images)

        # Tính toán và hiển thị kết quả sau mỗi Epoch
        train_epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validate_model(model, val_loader, device)

        # Nếu mô hình hiện tại tốt hơn (loss thấp hơn), tiến hành lưu lại
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.out_dir, exist_ok=True)
            # Lưu bộ trọng số (state_dict) vào file .pth
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))

        # In thông tin tiến trình ra màn hình
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")