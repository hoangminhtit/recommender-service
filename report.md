# Báo cáo: Hệ khuyến nghị 2-stage (BPR + BGE Reranker)

## 1) Tổng quan kiến trúc
Hệ thống sử dụng kiến trúc 2-stage cho nền tảng sách:
- Stage 1: BPR (Bayesian Personalized Ranking) để lấy ứng viên nhanh.
- Stage 2: BGE Reranker (BAAI/bge-reranker-base) để rerank theo ý nghĩa ngữ nghĩa từ hồ sơ người dùng và mô tả sách.

Luồng xử lý tổng quát:
1. Tạo bảng tương tác có time-decay từ DWH (data được sử dụng từ dev-seed).
2. Train BPR -> sinh top-k ứng viên cho mỗi user.
3. Xây dựng user profile + item description.
4. Rerank ứng viên bằng BGE -> top-k cuối.
5. Có fallback cho cold/hybrid user.

## 2) Stage 1: Candidate Retrieval (BPR)
### 2.1. Dữ liệu đầu vào
Dữ liệu từ DWH mock:
- dim_books, dim_date, dim_users
- fact_cart, fact_reviews, fact_sales

Notebook [databricks/1_prepare_data.ipynb](databricks/1_prepare_data.ipynb) kiểm tra file và đọc nhanh DWH mock.

### 2.2. Xây dựng bảng interaction có time-decay
Bảng interaction được build từ 3 nguồn hành vi:
- Add to cart (base score 3)
- Purchase (base score 10)
- Review >= 4 (base score 5)

Mỗi tương tác được giảm dần theo thời gian:
$$
\text{decay} = e^{-\lambda \cdot \text{months\_ago}}, \quad \lambda \approx 0.05
$$

Sau đó cộng tổng theo (user, book) để tạo interaction_score. Logic chi tiết nằm trong tài liệu pipeline [recommendation_pipeline_readme.md](recommendation_pipeline_readme.md).

Quy trình còn có bước kiểm tra dữ liệu đầu ra:
- Loại bỏ bản ghi `interaction_score` bị null hoặc <= 0.
- Đảm bảo đủ các cột `user_id`, `book_id`, `interaction_score`.

### 2.3. Train BPR và lưu model
Notebook [databricks/2_train_model.ipynb](databricks/2_train_model.ipynb) gọi:
- `build_interaction_table` để tạo interaction_df.
- `filter_cold_books` để loại sách quá ít tương tác.
- `train_bpr` để học user/item embedding.
- `save_bpr_model` lưu checkpoint vào `models/bpr/model.npz`.

Chi tiết `filter_cold_books`:
- Đếm số tương tác theo `book_id` từ interaction_df.
- Các sách có số tương tác < `settings.cold_start.min_book_interactions` sẽ bị loại khỏi tập train.
- Mặc định ngưỡng là 3 (đã hạ để phù hợp dữ liệu mock nhỏ).
- Hàm trả về (interaction_df đã lọc, danh sách cold_book_ids) để các bước sau biết sách bị loại.

Chi tiết `train_bpr`:
- Dùng BPR pairwise ranking với negative sampling (1 negative cho mỗi positive).
- Hyperparameters chính: factors=64, epochs=30, learning_rate=0.05, reg=0.0025.
- Mỗi epoch sẽ shuffle toàn bộ (user, book) pairs và cập nhật bằng SGD.

Model BPR được đăng ký qua MLflow (pyfunc) với đầu vào `user_id`, `top_k` và đầu ra danh sách ứng viên (book_id, mf_score, rank).

### 2.4. Sinh ứng viên top-k
Sau khi có BPR, hệ thống sinh top-k ứng viên trên tất cả sách để rerank. Output có dạng:
- `user_id`, `candidate_book_id`, `mf_score`.

Chi tiết `generate_candidates`:
- Chấm điểm tất cả item cho từng user bằng `bpr_model.score_items`.
- Lấy Top-(2 * num_candidates), sau đó filter các sách user đã tương tác.
- Cắt lại đúng Top-50 theo `settings.bpr.num_candidates` sau khi filter.

## 3) Stage 2: Semantic Reranking (BGE)
### 3.1. Xây dựng user profile và item description
- `build_user_profiles`: tổng hợp sở thích người dùng (category/author/recency) từ interaction_df.
- `build_item_descriptions`: tạo mô tả sách từ metadata (title, author, category, rating...).

Chi tiết `build_user_profiles`:
- Lấy Top categories và Top authors bằng cách SUM `interaction_score` theo user.
- Lấy 2 sách mua gần nhất và 2 sách được review >= 4.0. Lấy từ fact_sales với item_status = 'completed', nên đó là các sách user đã mua. Nếu user không có đủ 2 bản ghi mua gần nhất thì phần đó sẽ ít hơn hoặc trống (không có dòng “Recently purchased” trong profile)
- Ghép lại thành profile dạng text:
	- Categories (Top-3)
	- Authors (Top-3)
	- Highly rated books (Top-2)
	- Recently purchased (Top-2)

Chi tiết `build_item_descriptions`:
- Lấy metadata từ `dim_books` theo danh sách candidate_book_ids (chia chunk 1000).
- Format mô tả theo template:
	- Title
	- Author
	- Category
	- Average Rating
	- Popularity rank (purchase_count)

### 3.2. Rerank bằng BGE
Notebook [databricks/2_train_model.ipynb](databricks/2_train_model.ipynb) đăng ký BGE reranker bằng MLflow, dùng model `BAAI/bge-reranker-base`.

Khi infer:
- Đưa `user_profile` và `item_description` vào BGE.
- Lấy `semantic_score` (sigmoid logits) để sắp xếp lại danh sách ứng viên.

Chi tiết `rerank_candidates`:
- Dùng cross-encoder BGE để chấm điểm từng cặp (profile, description).
- Batch size mặc định 32, chạy trên `settings.reranker.device` (mặc định cpu).
- Sắp xếp theo `semantic_score` giảm dần và lấy Top-5 (`settings.reranker.top_k`).

## 4) Inference và fallback
Notebook [databricks/3_inference.ipynb](databricks/3_inference.ipynb) mô tả pipeline inference:
- Load BPR + BGE checkpoint.
- Build interaction table.
- Nếu user warm -> rerank full (BPR + BGE).
- Nếu user hybrid -> trộn BPR + popularity.
- Nếu cold -> dùng popularity.

Chi tiết phân nhóm user (cold/hybrid/warm):
- cold: 0 tương tác.
- hybrid: 1-4 tương tác.
- warm: >= 5 tương tác (`settings.cold_start.min_interactions_for_full_bpr`).

Chi tiết hybrid:
- Trộn điểm BPR và popularity theo trọng số 0.3 / 0.7.
- Popularity được lấy từ Top-N pool (mặc định 10) và random sample để tăng diversity.

Ngoài ra notebook này còn đóng gói pipeline 2-stage vào MLflow pyfunc `TwoStageRecommender`, nhận input `user_id` và trả về danh sách khuyến nghị đã rerank.

## 5) Điểm nổi bật trong thiết kế
- Có time-decay để giảm ảnh hưởng tương tác cũ.
- Two-stage giúp cân bằng giữa tốc độ (retrieval) và chất lượng (rerank).
- Cơ chế fallback cho cold/hybrid user tăng độ bao phủ.
- Tích hợp MLflow để versioning và deploy.

## 6) Kết luận
Hệ thống đã được triển khai theo kiến trúc 2-stage rõ ràng:
- BPR phù hợp cho truy hồi ứng viên nhanh với dữ liệu tương tác.
- BGE reranker nâng cao độ chính xác nhờ semantic matching giữa hồ sơ người dùng và mô tả sách.

Báo cáo này được tổng hợp từ notebook Databricks và tài liệu pipeline trong repo.