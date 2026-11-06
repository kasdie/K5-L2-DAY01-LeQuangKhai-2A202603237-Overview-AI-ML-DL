# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…

sample_id = traffic
class_id = 468
class_name = cab
rank = 1
score = 0.510915
taxonomy_name = ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

Record là kết quả phân loại cho cả bức ảnh. Mô hình xếp hạng các lớp cho ảnh đầu vào; record không có tọa độ hộp để xác định một chiếc taxi cụ thể. rank = 1 nghĩa là cab có score cao nhất trong các lớp.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

Danh sách lớp đi kèm taxonomy ImageNet-1K mà checkpoint đã được huấn luyện để phân loại. Model chọn và xếp hạng các lớp có sẵn, không tự tạo tên lớp mới.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

class_id = 468 là mã lớp, class_name = cab giúp người đọc hiểu tên lớp, còn taxonomy_name = ImageNet-1K xác định hệ thống lớp đang sử dụng. Mã lớp chỉ có ý nghĩa khi biết nó thuộc taxonomy và cách ánh xạ nào.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

Guideline cần quy định cách chọn một chủ đề chính, chẳng hạn dựa trên đối tượng mục tiêu hoặc mức độ nổi bật, cùng quy tắc xử lý khi nhiều chủ đề có vai trò tương đương. Nếu chưa đủ quy định để chọn, annotator cần hỏi Lab Coach.

6. Vì sao không được lấy `score` của model làm ground truth?

score = 0.510915 là điểm mô hình dùng để xếp hạng prediction, không chứng minh nhãn cab đúng. Ground truth cần được con người xác định theo guideline và kiểm tra chất lượng. Không thể diễn giải score này thành “nhãn đúng 51,09%”.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

sample_id = kitchen
class_name = bowl
score = 0.719062
bbox_xyxy = [32.65, 342.12, 100.16, 384.93]
bbox_width = 67.51
bbox_height = 42.81

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

Hộp dự đoán cho vật thể thuộc lớp bowl có góc trên trái tại (32.65, 342.12) và góc dưới phải tại (100.16, 384.93). Hộp rộng 67.51 pixel và cao 42.81 pixel.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

Ở hai ngưỡng 0.35 và 0.60. Với ảnh kitchen, ở ngưỡng 0.35 model tạo 11 prediction, còn ở ngưỡng 0.60 model tạo 6 prediction.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

Khi hạ ngưỡng, mô hình thường giữ lại nhiều prediction hơn. Reviewer cần đối chiếu ảnh để kiểm tra thêm các trường hợp nhận sai lớp, nhận nhầm nền thành vật thể, hộp lệch hoặc hộp trùng. Không loại prediction chỉ vì score thấp mà cần kiểm tra nội dung ảnh.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

Với vật thể nhìn thấy đầy đủ, vẽ hộp chữ nhật bao sát toàn bộ vật thể, hạn chế lấy thừa nền và không cắt mất phần của vật thể. Trường hợp bị che khuất hoặc cắt mép phải tuân theo quy định riêng trong guideline.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Ở mép trái ảnh kitchen, chỉ nhìn thấy một phần bàn tay/cánh tay, còn phần lớn cơ thể nằm ngoài khung hình. Model dự đoán person với score khoảng 0.61. Nếu guideline chưa quy định cách gán nhãn người bị cắt mép, tôi sẽ hỏi Lab Coach (escalation) để thống nhất có gán nhãn hay không và cách vẽ box, tránh tự suy đoán phần cơ thể ngoài ảnh.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

sample_id = kitchen
instance_id = kitchen-002
class_name = bowl
score = 0.735743
polygon_point_count = 67

5 điểm đầu của polygon_xy:
[[53.0, 344.0],
 [52.0, 345.0],
 [50.0, 345.0],
 [49.0, 346.0],
 [48.0, 346.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

Box chỉ là hình chữ nhật bao quanh vật thể nên có thể chứa cả nền. Polygon mô tả đường biên chi tiết hơn, giúp xác định vùng ảnh thuộc về vật thể.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

instance_id dùng để phân biệt từng vật thể riêng trong output, nghĩa là nhận diện 1 chiếc bát cụ thể, những chiếc bát khác có id riêng dù thuộc cùng lớp bowl.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

Vẽ polygon sát phần rìa nhìn thấy của từng vật thể, không lấy thêm nền hoặc phần của vật thể bên cạnh. Mỗi vật thể cần có mask riêng, kể cả khi chúng cùng lớp. Phần bị che khuất phải xử lý theo guideline, không tự suy đoán.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Ở chiếc bát có nhãn bowl 0.74 phía trước bàn, mép dưới của bát tiếp giáp với mặt bàn. Đây là vị trí cần phân biệt rõ vùng thuộc chiếc bát và vùng thuộc mặt bàn để tránh mask lấy thừa nền. Nếu guideline đã quy định vẽ theo biên nhìn thấy, tôi áp dụng quy tắc đó. Nếu một đoạn biên không đủ rõ và guideline chưa hướng dẫn cách xử lý, tôi sẽ hỏi Lab Coach (escalation) để thống nhất.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|

| Phân loại ảnh | Một nhãn cho toàn ảnh theo taxonomy và guideline của bài toán single-label. | Với ảnh `traffic`, model xếp `cab` hạng 1, score `0.510915`, và `minibus` hạng 2, score `0.164284`. Cần đối chiếu ảnh và guideline để quyết định nhãn toàn ảnh, không chọn chỉ dựa vào score. | Quan sát toàn ảnh, chọn lớp theo quy tắc xác định chủ đề chính; hỏi Lab Coach nếu nhiều chủ đề khó phân định. | Kiểm tra lớp có thuộc taxonomy và cách chọn nhãn có đúng guideline không. |
| Phát hiện vật thể | Một lớp và một bounding box cho mỗi vật thể thuộc phạm vi gán nhãn. | Ở mép trái ảnh `kitchen`, chỉ thấy một phần bàn tay/cánh tay nhưng model dự đoán `person`, score `0.610931`. Đây là trường hợp bị cắt mép cần có quy định rõ. | Áp dụng guideline về vật thể bị cắt mép; hỏi Lab Coach nếu chưa rõ có gán nhãn và cách vẽ box. | Kiểm tra lớp, độ sát của box, vật thể bị bỏ sót và tính nhất quán khi xử lý vật thể bị cắt mép. |
| Instance segmentation | Một lớp và một mask riêng cho mỗi instance. | Chiếc bát `kitchen-002`, score `0.735743`, có mép dưới tiếp giáp mặt bàn. Cần phân biệt vùng của bát với vùng của bàn để tránh lấy thừa nền. | Vẽ mask sát biên theo guideline, tách từng vật thể; hỏi Lab Coach khi biên khó xác định và chưa có quy định. | Kiểm tra mask có lấy thừa nền, thiếu phần vật thể hoặc gộp các instance hay không. |
## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

Tôi chỉ sử dụng ba ảnh công khai được cung cấp trong notebook, giữ thông tin nguồn ảnh và không đưa ảnh cá nhân, dữ liệu khách hàng, dữ liệu nội bộ hoặc thông tin cá nhân không cần thiết vào báo cáo/output công khai.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

Tôi sẽ dừng sử dụng và không tiếp tục tải lên hoặc chia sẻ dữ liệu đó, đồng thời báo GV/Lab Coach để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS