# Tài Liệu Thuyết Minh Trình Bày

> Tài liệu này được viết theo hướng giải thích, không phải lời thoại để đọc nguyên văn. Mục tiêu là giúp cả người đọc lẫn người nghe, dù có nền tảng kỹ thuật hay không, đều hiểu rõ bài toán, cách làm, phần đã hoàn thành, kết quả hiện tại và hướng phát triển tiếp theo. Các số liệu trong tài liệu này bám theo code và các artifact benchmark hiện có trong repository.

## 1. Tên Đề Tài

**Tên đề tài tiếng Việt**

Thiết kế và phát triển chatbot tài chính tiếng Việt tích hợp mô hình dự đoán biến động giá theo hướng đa phương thức.

**Ý nghĩa của tên đề tài**

Tên đề tài có hai lớp mục tiêu.

- Lớp mục tiêu dài hạn là xây dựng một chatbot tài chính tiếng Việt, tức là một giao diện mà người dùng có thể tương tác bằng ngôn ngữ tự nhiên.
- Lớp mục tiêu kỹ thuật cốt lõi ở giai đoạn hiện tại là xây dựng mô hình dự đoán biến động giá có khả năng kết hợp dữ liệu thị trường với dữ liệu tin tức tiếng Việt.

Nói đơn giản, chatbot là phần giao tiếp với người dùng, còn mô hình dự đoán là phần “bộ não” phân tích dữ liệu phía sau. Trong giai đoạn này, luận văn tập trung nhiều nhất vào việc làm cho “bộ não” đó đủ tốt, đủ rõ ràng và đủ kiểm chứng được.

## 2. Tổng Quan Và Bối Cảnh

Thị trường chứng khoán Việt Nam là một môi trường khó dự đoán vì ba lý do chính.

Thứ nhất, dữ liệu giá có tính nhiễu cao và thay đổi theo thời gian. Một mô hình học tốt ở giai đoạn này chưa chắc còn tốt ở giai đoạn khác, vì thị trường có thể đổi trạng thái khi lãi suất thay đổi, chính sách thay đổi, hoặc tâm lý nhà đầu tư thay đổi.

Thứ hai, giá cổ phiếu không chỉ phản ánh dữ liệu lịch sử mà còn phản ứng với thông tin bên ngoài như tin doanh nghiệp, chính sách tiền tệ, tỷ giá, nợ xấu, tăng trưởng tín dụng hoặc biến động vĩ mô. Nếu chỉ nhìn giá quá khứ mà bỏ qua tin tức, mô hình sẽ thiếu một phần quan trọng của bức tranh thực tế.

Thứ ba, trong bối cảnh Việt Nam, ngôn ngữ cũng là một thách thức công nghệ. Tin tức tài chính tiếng Việt có đặc thù riêng về cách dùng từ, thuật ngữ ngành, tên tổ chức, tên doanh nghiệp và cách viết. Nếu dùng mô hình ngôn ngữ quá tổng quát, hệ thống có thể hiểu không đúng sắc thái hoặc không hiểu đủ bối cảnh.

Vì vậy, luận văn chọn cách tiếp cận đa phương thức, nghĩa là kết hợp hai nguồn dữ liệu khác nhau.

- Nguồn thứ nhất là dữ liệu thị trường: giá mở cửa, cao nhất, thấp nhất, đóng cửa, khối lượng, cùng các chỉ báo kỹ thuật.
- Nguồn thứ hai là dữ liệu văn bản: tin tức tài chính tiếng Việt, được biến đổi thành dạng số thông qua mô hình ngôn ngữ và mô hình cảm xúc.

Mục tiêu không phải chỉ để chứng minh rằng “dùng thêm tin tức nghe có vẻ hợp lý”, mà là kiểm tra một cách định lượng xem tin tức tiếng Việt có thực sự cải thiện năng lực dự báo so với mô hình chỉ dùng dữ liệu giá hay không.

## 3. Công Trình Liên Quan

Phần công trình liên quan có thể chia thành bốn hướng chính.

**Hướng thứ nhất: xử lý ngôn ngữ tiếng Việt trong tài chính.** Các nghiên cứu về PhoBERT và các mô hình tiếng Việt cho thấy mô hình được huấn luyện chuyên cho tiếng Việt thường làm tốt hơn các mô hình đa ngôn ngữ tổng quát khi xử lý văn bản tiếng Việt. Điều này rất quan trọng vì dữ liệu văn bản của luận văn là tin tài chính tiếng Việt chứ không phải tiếng Anh.

**Hướng thứ hai: dự báo chuỗi thời gian tài chính.** Những mô hình như Random Forest, LSTM, CNN-LSTM và gần đây là Chronos đại diện cho các trường phái khác nhau trong dự báo: học trên đặc trưng dạng bảng, học phụ thuộc theo thời gian bằng mạng hồi tiếp, học mô típ ngắn hạn bằng convolution kết hợp học phụ thuộc dài hạn bằng LSTM, và học chuyển giao từ mô hình nền tảng chuỗi thời gian.

**Hướng thứ ba: dự báo đa phương thức.** Nhiều nghiên cứu đã cho thấy rằng việc kết hợp chuỗi thời gian với tiêu đề tin tức, sự kiện, hoặc thông tin văn bản có thể tăng chất lượng dự báo. Tuy nhiên, điểm khó nằm ở chỗ ghép hai loại dữ liệu này như thế nào. Nếu ghép quá đơn giản, mô hình có thể không phân biệt được tin nào quan trọng và tin nào chỉ là nhiễu. Vì vậy, các hướng gần đây thường dùng cơ chế attention, gated fusion hoặc cross-attention để mô hình tự học nên tập trung vào thông tin nào.

**Hướng thứ tư: chatbot và hệ đa tác tử trong tài chính.** Các công trình mới cho thấy chatbot tài chính và hệ đa tác tử có thể giúp người dùng tương tác thuận tiện hơn với hệ thống phân tích. Tuy nhiên, nhiều hệ thống chỉ mạnh ở phần giao tiếp mà chưa có lõi dự báo đủ vững. Điều đó dẫn tới rủi ro là hệ thống nói hay nhưng không có bằng chứng định lượng đủ mạnh đằng sau.

Từ bốn hướng trên, có thể thấy luận văn không đứng riêng lẻ mà nằm tại giao điểm của ba vấn đề: hiểu tiếng Việt tài chính, dự báo giá tài sản, và triển khai một hệ thống cuối cùng có thể giao tiếp với người dùng.

## 4. Mục Tiêu Nghiên Cứu Và Khoảng Trống

Mục tiêu tổng thể của luận văn là xây dựng một hệ thống hỗ trợ ra quyết định tài chính bằng tiếng Việt, trong đó phần giao diện cuối cùng có thể phát triển thành chatbot, còn phần phân tích lõi phải có khả năng dự báo, giải thích được và kiểm chứng được.

Ở giai đoạn hiện tại, mục tiêu cụ thể hơn là xây dựng và đánh giá một pipeline dự báo đa phương thức theo kiến trúc Cross-Modal Temporal Fusion, gọi tắt là CMTF. Pipeline này phải làm được bốn việc:

1. Thu thập và chuẩn hóa dữ liệu thị trường và dữ liệu tin tức.
2. Căn thời gian giữa tin tức và dữ liệu giá theo cách tránh rò rỉ thông tin tương lai.
3. Chuyển tin tức tiếng Việt thành tín hiệu có thể dùng cho mô hình học máy.
4. So sánh công bằng mô hình đa phương thức với các mô hình chỉ dùng thị trường.

Khoảng trống nghiên cứu mà luận văn muốn giải quyết gồm ba điểm.

- Nhiều nghiên cứu tiếng Việt dừng lại ở phân loại cảm xúc, nhưng chưa đi tiếp tới bài toán dự báo biến động giá.
- Nhiều nghiên cứu đa phương thức được làm trên dữ liệu không phải tiếng Việt hoặc chưa nhấn mạnh đúng mức tới vấn đề rò rỉ thời gian khi ghép tin tức với dữ liệu giao dịch.
- Nhiều chatbot tài chính ở Việt Nam phục vụ tác vụ dịch vụ hoặc hỏi đáp đơn giản, chưa thật sự là hệ thống hỗ trợ quyết định đầu tư có lõi phân tích được kiểm chứng chặt chẽ.

Vì vậy, điểm trọng tâm của luận văn không chỉ là “phân loại cảm xúc tin tức” hay “làm chatbot”, mà là nối liền các bước đó thành một quy trình dự báo hoàn chỉnh và có thể audit được.

## 5. Ý Tưởng Tiếp Cận Tổng Thể Và Luồng Xử Lý

Toàn bộ quy trình hiện tại có thể hiểu thành bảy bước liên tiếp.

1. Thu thập dữ liệu OHLCV theo ngày cho các mã mục tiêu và lấy thêm dữ liệu VN-Index để làm tín hiệu vĩ mô bổ sung.
2. Thu thập tin tức tài chính tiếng Việt từ nhiều nguồn, sau đó chuẩn hóa thời gian xuất bản, loại tin không liên quan, loại tin trùng lặp.
3. Gán mỗi tin tức vào đúng cây nến thị trường tương ứng theo quy tắc giờ đóng cửa, nhằm tránh việc mô hình “nhìn thấy tương lai”.
4. Tạo đặc trưng cho dữ liệu thị trường như RSI, MACD, Bollinger Bands, ATR, tỷ lệ khối lượng và log return.
5. Mã hóa tin tức thành vector lai gồm phần ngữ nghĩa và phần cảm xúc.
6. Huấn luyện nhiều mô hình nền và mô hình CMTF trên cùng một giao thức chia dữ liệu theo thời gian.
7. So sánh kết quả bằng cả chỉ số sai số dự báo và các chỉ số có ý nghĩa tài chính.

Nếu nói thật ngắn gọn cho người không chuyên, pipeline này làm đúng ba việc lớn: lấy dữ liệu, biến dữ liệu thành tín hiệu có thể học được, rồi kiểm tra mô hình nào thực sự tốt hơn.

Thiết lập benchmark hiện tại tập trung vào hai cổ phiếu ngân hàng là VCB và BID, dùng dữ liệu ngày từ 2022-01-01 đến 2026-03-31, với ba horizon dự báo là 1 ngày, 5 ngày và 20 ngày. Mục tiêu dự báo không phải giá tuyệt đối, mà là **forward log return**, tức là mức biến động giá trong tương lai dưới dạng logarit. Cách biểu diễn này ổn định hơn khi so sánh giữa các mã và giữa các giai đoạn khác nhau.

## 6. Những Phần Việc Đã Hoàn Thành

### 6.1 Toàn Bộ Pipeline Hiện Tại

Đến thời điểm hiện tại, pipeline cốt lõi đã được triển khai đầy đủ từ bước lấy dữ liệu cho tới bước xuất kết quả benchmark. Điều này có nghĩa là luận văn không dừng ở mức ý tưởng hoặc sơ đồ kiến trúc, mà đã có một luồng chạy thực tế, có thể tái lập và kiểm chứng.

Pipeline hiện hỗ trợ các thành phần chính sau đây.

- Lấy dữ liệu thị trường.
- Thu thập tin tức từ nhiều nguồn.
- Căn chỉnh thời gian giữa tin tức và dữ liệu giá.
- Tạo đặc trưng kỹ thuật cho dữ liệu thị trường.
- Mã hóa tin tức và cảm xúc tiếng Việt.
- Ghép dữ liệu thành dataset theo cửa sổ thời gian.
- Chia train, validation, test theo thời gian.
- Huấn luyện mô hình nền và mô hình CMTF.
- Tính benchmark và xuất file kết quả.

Hai điểm vào thực thi quan trọng nhất hiện nay là pipeline dữ liệu và benchmark runner. Điều này cho phép nhóm vừa kiểm tra chất lượng dữ liệu, vừa kiểm tra chất lượng mô hình trong cùng một hệ thống.

### 6.2 Thu Thập Dữ Liệu: Nguồn, Phương Pháp, Kiểm Chứng Và Dữ Liệu Hiện Có

#### Dữ liệu thị trường

Dữ liệu thị trường được lấy thông qua thư viện `vnstock`. Trong benchmark hiện tại, nguồn dữ liệu OHLCV đang dùng là KBS. Ngoài ra hệ thống còn lấy dữ liệu VN-Index để tạo hai biến vĩ mô phụ trợ.

- `vnindex_ret`: log return của VN-Index.
- `vnindex_vol_ratio`: tỷ lệ khối lượng hiện tại so với mức trung bình trượt của VN-Index.

Điểm quan trọng ở đây là mô hình không chỉ nhìn vào cổ phiếu riêng lẻ mà còn được cung cấp một phần tín hiệu của thị trường chung.

#### Dữ liệu tin tức

Ở thời điểm hiện tại, code đang triển khai **5 nhánh thu thập chính và 1 nhánh dự phòng** cho dữ liệu tin tức.

- VNExpress theo từ khóa gắn với từng mã cổ phiếu.
- VNExpress theo từ khóa ngành ngân hàng và vĩ mô dùng chung cho mọi mã.
- CafeF chuyên mục tài chính ngân hàng, sau đó lọc theo mức liên quan tới từng mã.
- Vietstock theo trang tin của từng mã.
- Google News RSS cho cả truy vấn theo mã và truy vấn vĩ mô hoặc địa chính trị.
- Nếu scraping thất bại, hệ thống có thể fallback về tin công ty từ `vnstock` qua nguồn VCI.

Như vậy, hệ thống không phụ thuộc hoàn toàn vào một website duy nhất. Nếu một nguồn yếu hoặc tạm thời lỗi, pipeline vẫn có phương án duy trì dòng dữ liệu.

#### Phương pháp thu thập

Việc thu thập không chỉ là tải dữ liệu về. Pipeline hiện tại có các lớp xử lý sau.

- Retry khi request lỗi.
- Giới hạn tốc độ gọi API để tránh bị chặn.
- Header mô phỏng trình duyệt khi scraping web.
- Chuẩn hóa URL.
- Chuẩn hóa thời gian xuất bản.
- Lưu cache để tái sử dụng dữ liệu cũ.
- Xuất trace CSV để biết tin nào được giữ, tin nào bị loại, và bị loại vì lý do gì.

Điều này quan trọng vì với dữ liệu web, nếu không có cache và trace, rất khó giải thích sau này là dataset đã được hình thành như thế nào.

#### Cách kiểm chứng chất lượng dữ liệu

Pipeline tin tức hiện có nhiều lớp kiểm chứng.

- Chỉ cho phép các mã đang nằm trong phạm vi nghiên cứu, hiện tại là VCB và BID.
- Yêu cầu có thời gian xuất bản hợp lệ.
- Với CafeF và Vietstock, chỉ giữ các bài có nội dung đủ dài để giảm khả năng lấy phải dữ liệu quá nghèo thông tin.
- Loại trùng chính xác theo URL và loại trùng gần đúng theo độ giống tiêu đề, với ngưỡng fuzzy similarity là 85.
- Có test parser và test smoke tùy chọn cho crawler.

Lớp kiểm chứng quan trọng nhất là kiểm chứng theo thời gian. Quy tắc hiện tại là:

- Tin trước 15:00 ngày T được gán cho bar ngày T.
- Tin từ 15:00 trở đi được dời sang bar kế tiếp.
- Tin chỉ có ngày mà không có giờ cụ thể, thường rơi vào 00:00:00, cũng bị dời sang bar kế tiếp để an toàn.
- Tin cuối tuần hoặc ngày nghỉ được gán sang ngày giao dịch gần nhất tiếp theo.

Đây chính là phần giúp hạn chế **leakage**, tức là rò rỉ thông tin tương lai vào dữ liệu huấn luyện.

#### Dữ liệu hiện có

Hiện tại luận văn đang có cả dữ liệu giám sát cho bài toán cảm xúc và dữ liệu vận hành cho pipeline dự báo.

**Bộ dữ liệu cảm xúc giai đoạn 2**

- 1.005 tiêu đề bài báo đã gán nhãn.
- 804 mẫu train, 100 mẫu validation, 101 mẫu test.
- Phân bố nhãn: 187 negative, 249 neutral, 569 positive.

**Dữ liệu tin tức đã scrape gần nhất**

- Trace ngày 2026-05-20 cho VCB có 13.213 dòng tin.
- Trace ngày 2026-05-20 cho BID có 12.657 dòng tin.
- Trong các trace gần nhất, Google News RSS là nguồn chiếm tỷ trọng lớn nhất, sau đó mới tới VNExpress và VNExpress sector.

**Dữ liệu sau khi căn chỉnh và tổng hợp theo bar**

- Các file sentiment-bar gần nhất có khoảng 2.068 đến 2.108 dòng ở cấp bar.
- Tương ứng khoảng 1.034 đến 1.054 bar cho mỗi mã, tùy theo phiên bản cache.

Điểm cần nói rõ khi trình bày là cơ cấu nguồn dữ liệu hiện đã thay đổi so với một số cache cũ. Những bản export cũ cho thấy coverage thực thấp hơn, còn các cache mới có coverage bar rất cao vì Google News bổ sung nhiều tin vĩ mô. Vì thế khi giải thích “mức phủ tin tức”, cần nói rõ mức phủ đó thuộc cấu hình nguồn nào.

### 6.3 Tiền Xử Lý Dữ Liệu Và Cách Dữ Liệu Đi Vào Mô Hình

#### Tiền xử lý dữ liệu thị trường

Với dữ liệu thị trường, pipeline tính các đặc trưng kỹ thuật sau.

- RSI-14.
- MACD, MACD signal, MACD histogram.
- Bollinger Bands.
- ATR-14.
- Volume ratio.
- Log return trong ngày.

Đồng thời pipeline tạo ra nhãn dự báo là **forward log return** cho 1 ngày, 5 ngày và 20 ngày. Việc dùng log return thay vì giá tuyệt đối có hai lợi ích.

- Chuẩn hóa tốt hơn giữa các mã có mức giá khác nhau.
- Phản ánh trực tiếp hướng và cường độ biến động, phù hợp hơn với bài toán ra quyết định.

Sau đó các đặc trưng thị trường được chuẩn hóa bằng thống kê chỉ học từ tập train. Điều này rất quan trọng, vì nếu chuẩn hóa bằng toàn bộ dataset thì mô hình đã vô tình sử dụng thông tin của tương lai.

#### Tiền xử lý dữ liệu văn bản

Sau khi tin tức được căn thời gian vào đúng bar, mỗi bar sẽ có một tập bài báo tương ứng. Tập bài báo đó được xử lý theo hai nhánh song song.

**Nhánh ngữ nghĩa**

Tin tức được mã hóa bằng mô hình `dangvantuan/vietnamese-embedding`, tạo ra vector ngữ nghĩa 768 chiều. Có thể hiểu đơn giản đây là cách biến một đoạn văn bản tiếng Việt thành một điểm trong không gian số để mô hình nhìn ra độ giống hoặc khác nhau giữa các nội dung.

**Nhánh cảm xúc**

Tiêu đề tin tức được đưa qua mô hình PhoBERT đã fine-tune để sinh ra điểm cảm xúc. Từ các điểm cảm xúc ở cấp bài báo, pipeline tổng hợp thành 5 thống kê ở cấp bar.

- Trung bình cảm xúc.
- Độ lớn cảm xúc mạnh nhất.
- Tỷ lệ bài tích cực.
- Tỷ lệ bài tiêu cực.
- Số lượng bài có điểm cảm xúc hợp lệ.

#### Vector tin tức lai và cách dùng trong CMTF

Hai nhánh trên được ghép lại thành một **hybrid news vector**.

- 768 chiều ngữ nghĩa.
- 5 chiều thống kê cảm xúc.

Tổng cộng là **773 chiều cho mỗi bar** khi sentiment handoff được bật.

Ngoài ra còn có một cờ `sentiment_missing_flag` đi qua nhánh thị trường để mô hình phân biệt được hai trường hợp rất khác nhau:

- Không có tin.
- Có tin nhưng tin trung tính.

Về cấp dataset, mỗi mẫu huấn luyện dùng một cửa sổ lùi 30 bar.

- Nhánh thị trường nhận tensor `30 x 23`.
- Nhánh tin tức nhận tensor `30 x 773`.
- Đầu ra là một forward log return cho horizon được chọn.

Hiểu đơn giản cho người không chuyên: mô hình nhìn lại 30 phiên gần nhất, vừa nhìn số liệu giá vừa nhìn diễn biến tin tức, rồi học cách dự đoán biến động trong tương lai gần hoặc trung hạn.

### 6.4 Các Mô Hình Được So Sánh Và Lý Do Lựa Chọn

Luận văn không chỉ huấn luyện một mô hình rồi kết luận mô hình đó tốt. Thay vào đó, nhóm cố ý chọn nhiều mô hình nền để trả lời những câu hỏi phương pháp khác nhau.

| Mô hình | Dữ liệu đầu vào | Mô hình học gì | Đầu ra | Lý do chọn |
| --- | --- | --- | --- | --- |
| Random Forest | Đặc trưng thị trường đã làm phẳng | Quan hệ phi tuyến dạng bảng, không giữ thứ tự thời gian | Forward log return | Kiểm tra xem đặc trưng kỹ thuật tự thân đã đủ mạnh hay chưa |
| LSTM | Cửa sổ 30 bar của dữ liệu thị trường | Phụ thuộc theo chuỗi thời gian | Forward log return | Baseline hồi tiếp kinh điển cho chuỗi tài chính |
| CNN-LSTM | Cửa sổ 30 bar của dữ liệu thị trường | Mô típ ngắn hạn bằng convolution và phụ thuộc dài hạn bằng LSTM | Forward log return | Baseline mạnh và là backbone so khớp để so với CMTF |
| Chronos Zero-Shot | Cửa sổ giá đóng cửa | Tri thức chuyển giao từ foundation model chuỗi thời gian | Forward log return | Đo sức mạnh dự báo thuần chuyển giao, không fine-tune trong miền dữ liệu này |
| Chronos Fine-Tuned với LoRA | Chuỗi giá đã token hóa, có thể kèm nhánh market tabular | Thích nghi miền dữ liệu bằng cập nhật low-rank | Forward log return | Kiểm tra xem pretraining có còn hữu ích sau khi thích nghi nhẹ hay không |
| CNN-LSTM CMTF | `30 x 23` market tensor và `30 x 773` news tensor | Hợp nhất đa phương thức theo kiểu residual | Forward log return | Kiểm tra xem tin tức tiếng Việt có mang thêm giá trị dự báo ngoài backbone thị trường hay không |

Ở tầng cảm xúc văn bản, nhóm cũng so sánh hai mô hình dưới cùng một đầu ra 3 lớp cảm xúc để quyết định backbone nào đáng dùng hơn về sau.

| Mô hình cảm xúc | Đầu vào | Đầu ra | Lý do chọn |
| --- | --- | --- | --- |
| Custom Transformer | Tiêu đề tiếng Việt đã tiền xử lý | Xác suất 3 lớp và điểm cảm xúc kỳ vọng | Baseline gọn nhẹ, huấn luyện từ đầu |
| PhoBERT | Tiêu đề tiếng Việt qua backbone pretrained | Xác suất 3 lớp và điểm cảm xúc kỳ vọng | Kiểm tra giá trị của pretraining tiếng Việt |

Ý nghĩa của bước này là rất thực dụng: nếu PhoBERT không tốt hơn rõ rệt, thì không có lý do gì phải mang một mô hình ngôn ngữ nặng hơn vào pipeline dự báo.

### 6.5 Kiến Trúc CMTF, Cách Học, Dữ Liệu Vào, Đầu Ra Và Lý Do Chọn

Điểm đóng góp chính về mặt mô hình ở giai đoạn hiện tại là CMTF. Đây không phải mô hình ghép dữ liệu kiểu đơn giản, mà là mô hình **dự báo cơ sở rồi hiệu chỉnh bằng tin tức**.

Có thể viết ngắn gọn tư tưởng đó như sau:

**Dự báo cuối = dự báo từ thị trường + phần hiệu chỉnh do tin tức tạo ra**

#### Nhánh thị trường

Nhánh thị trường là nơi tạo ra dự báo nền ban đầu. Nó gồm ba tầng xử lý liên tiếp.

- Chiếu đầu vào để đưa dữ liệu thị trường về không gian ẩn phù hợp cho mô hình.
- Các khối causal dilated convolution để bắt mô típ ngắn hạn nhưng vẫn giữ đúng chiều thời gian.
- LSTM kết hợp temporal attention để tóm tắt 30 bar thành một trạng thái thị trường cô đọng.

Từ trạng thái đó, mô hình sinh ra dự báo nền chỉ dựa trên thị trường.

#### Nhánh tin tức

Nhánh tin tức không thay thế nhánh thị trường mà chỉ đóng vai trò hiệu chỉnh. Chuỗi tin tức 30 bar được đưa qua cross-attention, trong đó trạng thái thị trường đóng vai trò “câu hỏi”, còn chuỗi tin tức đóng vai trò “bộ nhớ để tham chiếu”.

Nói dễ hiểu: mô hình tự hỏi rằng “với trạng thái thị trường hiện tại, mẩu tin nào trong 30 phiên gần đây thực sự đáng chú ý?”.

Ngoài cross-attention, nhánh tin tức còn có các cơ chế ổn định hóa sau.

- Positional embedding học được để mô hình biết tin gần đây và tin cũ khác nhau như thế nào.
- Sigmoid gate theo từng chiều để chặn bớt nhiễu từ nhánh tin tức.
- Recency-aware density gate tập trung nhiều vào 5 bar gần nhất.
- Cơ chế zero-news parity, tức là nếu không có tin, phần residual bằng đúng 0.
- Tham số `news_weight` để điều khiển mức ảnh hưởng tổng thể của nhánh tin tức.

#### Tại sao chọn thiết kế residual

Thiết kế residual có ba ưu điểm lớn.

- **Dễ giải thích**: có thể xem tin tức đã kéo dự báo lên hay đẩy dự báo xuống bao nhiêu.
- **An toàn khi không có tin**: nếu không có tin tức, mô hình quay về dự báo thị trường gốc thay vì sinh ra hành vi bất ổn.
- **Dễ kiểm tra giá trị tăng thêm của văn bản**: khi so với backbone CNN-LSTM cùng cấu trúc thị trường, ta biết phần cải thiện đến từ tin tức chứ không phải chỉ từ việc thay backbone.

#### Cách huấn luyện

CMTF hiện được huấn luyện theo quy trình hai giai đoạn.

- Giai đoạn 1: khởi tạo từ backbone CNN-LSTM đã huấn luyện, đóng băng encoder thị trường, chỉ học nhánh fusion.
- Giai đoạn 2: mở lại encoder và fine-tune toàn mô hình với learning rate nhỏ hơn cho encoder.

Hàm mất mát chính là **sign-aware Huber loss**, nhằm vừa tối ưu sai số hồi quy vừa phạt thêm khi mô hình dự báo sai hướng. Ngoài ra mô hình còn dùng auxiliary loss để giữ cho nhánh thị trường không bị lệch quá mức khi học cùng nhánh tin tức, có gradient clipping, early stopping, và ensemble 3 seed ở giai đoạn suy luận.

Đầu ra cuối cùng của mô hình vẫn là forward log return. Điều này quan trọng vì toàn bộ benchmark đều so sánh trên cùng một loại mục tiêu.

### 6.6 Benchmark: Cách Đánh Giá, Cách Tính Và Ý Nghĩa

Benchmark được thiết kế để công bằng và chặt chẽ. Tất cả mô hình đều dùng cùng universe mã, cùng khoảng thời gian, cùng horizon dự báo và cùng giao thức chia dữ liệu.

#### Cách chia dữ liệu

- Train đến hết 2024-06-30.
- Validation đến hết 2024-12-31.
- Test là phần sau đó.

Ngoài ra còn có **purge buffer** bằng đúng độ dài horizon dự báo ở ranh giới giữa các split. Mục đích của purge buffer là tránh việc nhãn dự báo của tập trước chạm sang khoảng thời gian của tập sau.

Ví dụ, nếu dự báo 20 ngày thì một mẫu ở cuối tập train có thể đang “nhìn” tới giá của 20 ngày sau. Nếu không purge, phần giá đó có thể nằm trong vùng validation, và như vậy đã có leakage.

#### Các chỉ số đánh giá

Benchmark hiện không chỉ đo sai số dự báo thuần túy mà còn đo tính hữu ích về mặt tài chính.

| Chỉ số | Ý nghĩa |
| --- | --- |
| MAE | Sai số tuyệt đối trung bình |
| RMSE | Sai số căn bình phương trung bình, phạt mạnh lỗi lớn |
| DA% | Tỷ lệ đoán đúng hướng tăng hoặc giảm |
| Precision / Recall / F1 | Chất lượng dự đoán hướng tăng dưới góc nhìn phân loại |
| Sharpe | Hiệu quả lợi nhuận đã điều chỉnh theo rủi ro nếu giao dịch theo dấu dự báo |
| IC | Tương quan xếp hạng giữa dự báo và thực tế |
| Temporal Lag | Mức trễ pha của dự báo so với chuỗi thật |

#### CompositeScore hiện tại được tính như thế nào

Trong code hiện tại, CompositeScore được tính theo công thức trọng số sau, với **giá trị càng thấp càng tốt**.

- `0.5 x RMSE`
- `0.15 x MAE`
- `0.15 x (1 - DA/100)`
- `0.12 x (1 - F1)`
- `0.08 x TemporalLag`

File CSV benchmark vẫn còn xuất thêm `ModalDisagreement` như một cột chẩn đoán, nhưng cột này không còn là thành phần chính của công thức chấm điểm tổng hợp hiện tại.

Nói đơn giản cho người không chuyên: benchmark không chỉ hỏi “mô hình dự báo gần đúng bao nhiêu”, mà còn hỏi “mô hình có đoán đúng chiều tăng giảm không”, “nếu dùng để ra quyết định thì có hữu ích không”, và “dự báo có bị trễ nhịp so với thực tế hay không”.

### 6.7 So Sánh Với Một Số Hướng Nghiên Cứu Liên Quan Và Lý Do Chọn Thiết Kế Hiện Tại

Để trả lời câu hỏi “luận văn này giống và khác gì so với các công trình trước”, có thể so sánh theo bảng sau.

| Công trình hoặc hướng nghiên cứu | Điểm giống với luận văn | Điểm khác trong triển khai hiện tại | Lý do nhóm chọn cách làm hiện tại |
| --- | --- | --- | --- |
| Deep fusion giữa headline và time series | Cùng mục tiêu kết hợp văn bản và dữ liệu giá | Luận văn dùng tiếng Việt, có căn giờ đóng cửa, và có thêm 5 đặc trưng cảm xúc vào vector ngữ nghĩa | Bài toán tiếng Việt cần bản địa hóa ngôn ngữ và kiểm soát thời gian chặt hơn |
| MSGCA 2024 về gated cross-attention cho dự báo chuyển động giá | Cùng ý tưởng dùng cross-attention và gating cho hợp nhất đa phương thức | CMTF hiện tại dùng thiết kế residual, zero-news parity, cửa sổ tin 30 bar và backbone CNN-LSTM so khớp | Thiết kế residual giúp dễ giải thích hơn và an toàn hơn khi dữ liệu tin thưa hoặc nhiễu |
| Các nghiên cứu dự báo theo log return | Cùng chọn mục tiêu là return thay vì giá tuyệt đối | Luận văn dùng đồng thời 1D, 5D, 20D và áp dụng chung cho benchmark nhiều mô hình | Log return phù hợp hơn cho so sánh giữa horizon và giữa các mã |
| Chronos 2024 và hướng foundation model cho time series | Cùng dùng mô hình nền tảng chuỗi thời gian làm baseline mạnh | Chronos trong luận văn hiện là baseline thị trường, chưa phải backbone fusion chính hiện tại | Cách này giúp tách bạch giá trị của transfer learning và giá trị của fusion văn bản |
| Các nghiên cứu chỉ báo cáo MAE hoặc RMSE | Cùng đánh giá sai số dự báo | Luận văn bổ sung DA, F1, Sharpe, IC và Temporal Lag | Vì mục tiêu cuối là hỗ trợ quyết định, không thể chỉ nhìn sai số số học |

Từ bảng trên có thể rút ra logic thiết kế của luận văn: điểm mới không nằm ở một thành phần đơn lẻ, mà nằm ở việc kết hợp nhiều quyết định hợp lý trong cùng một hệ thống, gồm tiếng Việt chuyên biệt, căn chỉnh thời gian chống leakage, biểu diễn tin tức lai, fusion dạng residual và benchmark có ý nghĩa tài chính.

## 7. Kết Quả Hiện Tại

### 7.1 Kết Quả Ở Giai Đoạn Mô Hình Cảm Xúc

Ở giai đoạn cảm xúc, kết quả hiện tại khá rõ ràng: **PhoBERT đang là mô hình tốt hơn và đã được chọn làm handoff model cho pipeline dự báo**.

| Mô hình | Macro-F1 validation | Macro-F1 test | Accuracy test | Kết luận |
| --- | ---: | ---: | ---: | --- |
| Custom Transformer | 0.7623 | 0.6782 | 0.7129 | Giữ vai trò baseline |
| PhoBERT | 0.7704 | 0.8410 | 0.8515 | Được chọn làm mô hình cảm xúc đưa sang downstream |

Ý nghĩa của kết quả này là: với dữ liệu tài chính tiếng Việt, pretraining bằng mô hình tiếng Việt thực sự có giá trị, chứ không chỉ là lựa chọn mang tính hình thức.

### 7.2 Kết Quả Benchmark Ở Giai Đoạn Dự Báo

Ở giai đoạn dự báo, benchmark hiện tại cho thấy một bức tranh thực tế và khá thú vị: pipeline đa phương thức đã được triển khai đầy đủ, nhưng lợi ích của nhánh tin tức chưa ổn định trên mọi horizon.

Kết quả trung bình hiện tại trên VCB và BID có thể tóm tắt như sau.

| Horizon | Diễn giải kết quả hiện tại |
| --- | --- |
| 1D | CNN-LSTM đang có CompositeScore tốt nhất là 0.2155. CNN-LSTM CMTF rất gần với 0.2180. LSTM có directional accuracy cao nhất là 54.38%. |
| 5D | LSTM đang có CompositeScore tốt nhất là 0.1999. CNN-LSTM có RMSE thấp nhất là 0.04653. Nhánh CMTF hiện chưa vượt backbone CNN-LSTM ở horizon này. |
| 20D | LSTM đang mạnh nhất về tổng thể với DA 66.73%, CompositeScore 0.1624, Sharpe 1.03 và IC 0.2081. Chronos Fine-Tuned với LoRA có RMSE tốt nhất là 0.10085 và F1 tốt nhất là 0.6183. |

Kết luận trung thực ở thời điểm này không phải là “fusion luôn thắng”. Kết luận đúng hơn là:

- Hạ tầng đa phương thức đã chạy hoàn chỉnh.
- Nhánh tin tức đã có thể kiểm tra định lượng một cách nghiêm túc.
- Benchmark đang cho thấy rõ khi nào tín hiệu tin tức hữu ích và khi nào chưa đủ mạnh.

### 7.3 Cách Hiểu Đúng Về Kết Quả Hiện Tại

Có ba cách diễn giải quan trọng.

**Thứ nhất, nhánh thị trường hiện rất mạnh.** LSTM và CNN-LSTM đang học được nhiều cấu trúc có ích từ dữ liệu giá và chỉ báo kỹ thuật. Điều này cho thấy backbone thị trường của hệ thống không yếu.

**Thứ hai, nhánh tin tức có tiềm năng nhưng chưa ổn định.** Việc CMTF chưa thắng đều ở mọi horizon không phải là thất bại của ý tưởng, mà là tín hiệu cho thấy bài toán fusion vẫn còn cần tinh chỉnh. Đây chính là giá trị của benchmark: nó chỉ ra chính xác chỗ nào còn yếu.

**Thứ ba, chất lượng tín hiệu tin tức hiện phụ thuộc mạnh vào nguồn dữ liệu.** Các trace gần đây cho thấy Google News RSS đang chiếm tỷ trọng rất lớn. Điều đó làm tăng coverage, nhưng cũng đặt ra câu hỏi liệu tín hiệu thêm vào là thông tin thực sự liên quan tới mã cổ phiếu hay chỉ là nhiễu vĩ mô ở mật độ cao.

Nói cách khác, bài toán hiện nay không còn là “có nên dùng tin tức hay không”, mà là “nên lọc, nhóm và dùng tin tức như thế nào để tín hiệu văn bản thật sự giúp ích cho dự báo”.

## 8. Phần Sắp Làm

Từ kết quả hiện tại, các bước tiếp theo đã khá rõ.

**Thứ nhất, siết lại chất lượng dữ liệu tin tức.** Cần audit kỹ hơn cơ cấu nguồn, đặc biệt là tỷ trọng quá lớn của Google News trong các cache mới. Đồng thời cần phân biệt rõ tin gắn trực tiếp với doanh nghiệp, tin ngành ngân hàng và tin vĩ mô.

**Thứ hai, cải thiện nhánh CMTF.** Mục tiêu là làm cho giá trị tăng thêm của tin tức trở nên ổn định hơn so với backbone CNN-LSTM. Hướng cải thiện có thể gồm lọc mức liên quan chặt hơn, tách riêng macro news và company news, hoặc làm ablation sâu hơn trên vector tin tức 773 chiều.

**Thứ ba, đồng bộ hóa phần viết của luận văn với benchmark hiện tại.** Vì benchmark và artifact đã cập nhật, phần mô tả kết quả trong tài liệu viết cũng cần phản ánh đúng trạng thái mới nhất.

**Thứ tư, tiếp tục các giai đoạn sau của đề tài.** Sau khi lõi dự báo ổn định hơn, nhóm sẽ đi tiếp tới phần multi-agent refinement, đánh giá A/B cho bước hậu kiểm dự báo, và cuối cùng là giao diện chatbot tài chính tiếng Việt.

## Kết Luận Ngắn

Giá trị hiện tại của luận văn không nằm ở một khẩu hiệu như “dùng AI cho chứng khoán”, mà nằm ở việc đã xây dựng được một pipeline dự báo đa phương thức tiếng Việt có thể chạy được, kiểm chứng được và phân tích được. Hệ thống này đã có mô hình cảm xúc được chọn rõ ràng, có quy trình căn thời gian tránh leakage, có nhiều baseline mạnh để đối chiếu, và có benchmark đủ chi tiết để chỉ ra chính xác phần nào đang tốt, phần nào còn cần cải thiện.