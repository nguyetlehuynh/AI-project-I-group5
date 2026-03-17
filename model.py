# Thư viện lõi của PyTorch dùng để tính toán ma trận (Tensor)
import torch

# Thư viện chuyên dụng của PyTorch cho xử lý ảnh (Computer Vision)
import torchvision

# Import module để chỉnh sửa lớp cuối cùng (lớp dự đoán) của mô hình Faster R-CNN.
# (Bạn sẽ cần dùng module này ở bước ngay sau để khai báo số lượng loại vật thể bạn muốn nhận diện)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Hàm khởi tạo mô hình. Tham số đầu vào 'backbone' là một chuỗi (string) 
# để chọn loại "xương sống" (mạng nơ-ron cơ sở) cho mô hình.
def build_model(backbone:str):
    
    # TRƯỜNG HỢP 1: Chọn xương sống là ResNet50
    # Đây là mô hình to, phức tạp, độ chính xác rất cao nhưng chạy hơi nặng.
    if backbone == 'fasterrcnn_resnet50_fpn':        
        
        # Tải cấu trúc mô hình ResNet50. 
        # pretrained=True có nghĩa là tải về một "bộ não" đã được học sẵn (Transfer Learning)
        # trên hàng triệu bức ảnh (tập dữ liệu COCO), thay vì bắt đầu học lại từ con số 0.
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Lưu lại bộ trọng số (kinh nghiệm đã học) tương ứng với mô hình này
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights
    
    # TRƯỜNG HỢP 2: Chọn xương sống là MobileNet V3
    # Đây là mô hình nhẹ, tối ưu hóa để chạy siêu nhanh (thậm chí trên điện thoại di động), 
    # nhưng độ chính xác có thể thấp hơn ResNet50 một chút.
    else:
        
        # Tải cấu trúc mô hình MobileNet V3 (cũng đã được học sẵn với pretrained=True)
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        
        # Lưu lại bộ trọng số của MobileNet V3
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights
        
    # (Đoạn code hiện tại đang thiếu lệnh return để trả về model sau khi tạo xong)
    # return model