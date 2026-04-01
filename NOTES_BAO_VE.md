# Notes Bao Ve Du An HARNN

File nay duoc viet de hoc bao ve theo dung code hien tai trong repo. Muc tieu khong phai viet lai ly thuyet chung, ma la giai thich tung khoi quan trong: no de lam gi, vi sao chon nhu vay, neu doi tham so thi co the anh huong the nao.

## 1. Bai toan cua repo

- Bai toan: phan loai van ban tieng Viet theo nhan phan cap 3 muc `L1 -> L2 -> L3`.
- Ve mat code, moi muc duoc bieu dien bang vector nhi phan `vec_l1`, `vec_l2`, `vec_l3`.
- Ve mat du lieu hien tai, moi bai gan nhu chi co 1 nhan o moi muc neu nhan do ton tai.

Can nho de tra loi:
- Repo dang viet theo huong "multi-label co the mo rong".
- Dataset hien tai gan voi "single-label theo tung muc trong cay phan cap".
- Day la ly do code dung `sigmoid + BCELoss`, nhung notebook evaluation lai co buoc `argmax` de xem nhu multiclass.

## 2. Tien xu ly du lieu

Tham chieu:
- `notebooks/preprocessing_data.ipynb`

### `MIN_TOKENS = 20`

De lam gi:
- Bo cac bai qua ngan, vi bai qua ngan thuong it thong tin va de gay nhieu.

Vi sao chon 20:
- 20 token la moc vua du de giu lai noi dung co nghia, nhung khong qua chat khien du lieu giam manh.

Neu chon nho hon:
- Giu lai nhieu mau hon.
- Doi lai co nhieu mau nhieu, tieu de ngan, noi dung rat it thong tin.
- Mo hinh co the hoc nham tu cac tin hieu yeu.

Neu chon lon hon:
- Chat luong mau thuong "sach" hon.
- Doi lai so luong du lieu giam, dac biet hai o cac nhan hiem.
- L3 co the bi anh huong manh hon vi da it du lieu.

### `VOCAB_MIN_COUNT = 3`

De lam gi:
- Loai cac tu xuat hien qua it de giam nhiu va giam kich thuoc tu dien.

Vi sao chon 3:
- 3 la moc nhe, van giu duoc nhieu tu mang tinh chu de, nhung bo bot typo va tu hiem khong on dinh.

Neu chon 1:
- Vocab rat lon.
- Nhieu tu hiem, typo, ten rieng chi xuat hien 1 lan.
- Embedding kho hoc on dinh hon.

Neu chon 5 hoac 10:
- Vocab gon hon, train nhanh hon.
- Doi lai de mat cac tu chuyen nganh, nhan hien hoi, ten nhan vat, ten giai dau, ten benh ly.

### Tach tu bang `underthesea`

De lam gi:
- Bien tieng Viet tu dang van ban thanh token co nghia.
- Cac cum nhu `bong_da`, `bat_dong_san`, `giao_duc` se duoc giu dung hon so voi tach theo tung tieng.

Vi sao can:
- Tieng Viet co rat nhieu tu da am tiet.
- Neu khong tach tu dung, Word2Vec va RNN se hoc bieu dien kem chinh xac.

Neu khong co `underthesea`:
- Notebook co fallback sang `.split()`.
- Cach nay van chay duoc, nhung chat luong token hoa giam ro.

### Stopword

De lam gi:
- Bo cac tu chuc nang xuat hien rat nhieu nhung it gia tri phan biet nhan.

Rui ro:
- Neu danh sach stopword qua tay, co the vo tinh bo luon mot so tu co nghia trong ngu canh bao chi.

## 3. Cau truc nhan

Code tao:
- `labels_l1`, `labels_l2`, `labels_l3`
- `vec_l1`, `vec_l2`, `vec_l3`

Y nghia:
- `labels_*` la dang de doc.
- `vec_*` la dang dua vao model va tinh loss.

Can nho de bao ve:
- Mac du vector la multi-hot, du lieu thuc te hien tai khong phai multi-label day du.
- Day la mot diem manh de mo rong ve sau, nhung cung la diem co the bi hoi: "Tai sao khong dung CrossEntropy?".

Cau tra loi goi y:
- Em chon bieu dien vector va `BCELoss` de giu mo hinh theo huong tong quat, de sau nay co the mo rong sang bai toan co nhieu nhan trong cung mot muc.
- Tuy nhien, voi dataset hien tai, dung `CrossEntropy` cho tung muc cung la mot baseline hop ly can so sanh them.

## 4. Chia du lieu

Tham chieu:
- `notebooks/train_w2v_clean.ipynb`

### Iterative Stratification

De lam gi:
- Chia `train/val/test` sao cho phan bo nhan duoc giu can bang hon so voi random split.

Vi sao dung:
- Du lieu co nhieu nhan hiem va mang tinh mat can bang.
- Random split de lam mot so nhan hiem gan nhu bien mat khoi val/test.

Neu dung random split:
- Ket qua co the giao dong manh giua cac lan chay.
- Cac metric, dac biet o L3, co the khong on dinh.

### Ty le 80/10/10

De lam gi:
- Can bang giua hoc mo hinh va danh gia.

Vi sao chon:
- 80% du lon de train.
- 10% du de chon mo hinh.
- 10% de test doc lap.

Neu test qua nho:
- Ket qua de bi may rui.

Neu val qua nho:
- Chon epoch tot nhat de bi nhieu.

## 5. Dataset va input cho model

### `MAX_LEN = 512`

De lam gi:
- Cat hoac pad moi bai ve cung do dai.
- Giup batch hoa va train on dinh.

Vi sao chon 512:
- 512 la moc pho bien, du dai de giu phan lon noi dung quan trong cua bai bao.

Neu giam xuong 256:
- Nhanh hon, it ton bo nho hon.
- De mat thong tin o cac bai dai.

Neu tang len 1024:
- Giu duoc nhieu thong tin hon.
- Cham hon, ton RAM/VRAM hon.
- RNN dai hon cung de gap van de toi uu kho hon.

### `BATCH_SIZE = 16`

De lam gi:
- Xac dinh so mau moi lan cap nhat gradient.

Vi sao chon 16:
- Thuong la muc an toan voi sequence dai 512 va BiGRU tren GPU tam trung.

Neu tang:
- Train nhanh hon theo wall-clock neu du bo nho.
- Gradient on dinh hon.
- Co the giam kha nang tong quat hoa mot chut.

Neu giam:
- Tieu ton thoi gian hon.
- Gradient nhieu nhieu hon, doi khi tong quat hoa tot hon, nhung khong on dinh bang.

## 6. Word2Vec

### `EMBED_DIM = 100`

De lam gi:
- Kich thuoc vector bieu dien cua moi token.

Vi sao chon 100:
- Day la muc vua du cho corpus tam trung, can bang giua chat luong bieu dien va chi phi train.

Neu chon 50:
- Nhe hon, nhanh hon.
- Co the mat bot kha nang bieu dien.

Neu chon 200 hoac 300:
- Co the bieu dien tinh vi hon.
- Nhung neu corpus khong du lon thi de overfit hoac hoc vector khong on dinh.

### `min_count = 5` trong Word2Vec

De lam gi:
- Bo tu qua hiem khi hoc Word2Vec de vector hoc on dinh hon.

Vi sao khac voi `VOCAB_MIN_COUNT = 3`:
- Vocab cho model va vocab cho Word2Vec co the khac nhau.
- Repo dung nguong 3 de giu tu cho input, nhung dung nguong 5 de vector Word2Vec it nhieu hon.

### `epochs = 10`

De lam gi:
- So vong hoc Word2Vec tren corpus.

Vi sao chon 10:
- Du de hoc cac quan he dong xuat hien co ban.

Neu qua it:
- Embedding hoc chua toi.

Neu qua nhieu:
- Ton thoi gian hon.
- Loi ich tang them co the it.

### `sg = 1`

De lam gi:
- Chon Skip-gram thay vi CBOW.

Vi sao chon:
- Skip-gram thuong hoc tot hon voi tu hiem va quan he ngu nghia tinh hon, doi lai train cham hon.

## 7. Embedding matrix

Code:
- Khoi tao ma tran embedding kich thuoc `VOCAB_SIZE x EMBED_DIM`
- Token `UNK` duoc gan bang trung binh cac vector Word2Vec

Y nghia:
- Neu gap tu ngoai vocab Word2Vec, model van co mot vector hop ly hon la all-zero ngau nhien.

Neu thay bang random:
- Van chay duoc.
- Nhung du doan voi tu la co the kem on dinh hon.

## 8. Kien truc HARNN

### `Embedding`

De lam gi:
- Bien id token thanh vector.

### `BiGRU`

De lam gi:
- Doc sequence hai chieu de lay ngu canh truoc va sau.

Vi sao dung GRU:
- Nhe hon LSTM.
- Thuong de train hon va du tot cho sequence dai vua.

Neu thay bang LSTM:
- Co the hoc duoc tot hon trong mot so truong hop.
- Nhung nang hon, cham hon.

### Attention rieng cho tung level

De lam gi:
- Moi muc nhan co the can nhin vao cac token quan trong khac nhau.

Vi sao khong dung chung 1 attention:
- Tin hieu huu ich de phan biet L1 chua chac da tot nhat cho L3.
- Tach rieng attention lam mo hinh linh hoat hon.

### `HAM = LSTMCell`

De lam gi:
- Luu bo nho xuyen qua cac muc phan cap.
- Muc sau nhan thong tin tom tat tu muc truoc.

Vi sao chon `LSTMCell`:
- Phu hop voi y tuong chay tung buoc qua L1, L2, L3.
- Nhe hon viec dung nguyen mot LSTM sequence khac.

### `hidden_size = 256`

De lam gi:
- Xac dinh do rong bo nho cua BiGRU va HAM.

Vi sao chon 256:
- La muc pho bien, du suc chua cho bai bao tieng Viet tam trung.

Neu 128:
- Nhe hon, nhanh hon.
- Co the chua du suc bieu dien.

Neu 512:
- Manh hon ve nang luc mo hinh.
- Nhung ton bo nho, de overfit hon neu du lieu khong du lon.

### `dropout = 0.5`

De lam gi:
- Chong overfitting.

Vi sao chon 0.5:
- Voi RNN + du lieu van ban, 0.3-0.5 la vung pho bien.
- 0.5 la lua chon an toan khi mo hinh khong qua lon nhung van co nguy co hoc thuoc.

Neu giam ve 0.2:
- De hoc manh hon tren train.
- Co the overfit nhanh hon.

Neu tang len 0.6:
- Chong overfit manh.
- Nhung co nguy co underfit.

## 9. Ham mat mat va trong so theo muc

Code:
- `criterion = nn.BCELoss()`
- `LEVEL_WEIGHTS = [1.0, 1.0, 1.2]`

### Vi sao dung `BCELoss`

De lam gi:
- Tinh loss doc lap cho tung nhan trong vector output.

Vi sao chon:
- Phu hop voi output dang `sigmoid`.
- Cho phep mo rong bai toan thanh multi-label that.

Neu doi sang `BCEWithLogitsLoss`:
- Thuong tot hon ve mat so hoc, vi on dinh hon `sigmoid + BCELoss`.
- Day la cai nen can nhac neu muon cai tien repo.

Neu doi sang `CrossEntropyLoss`:
- Phu hop hon neu chot bai toan la single-label cho tung muc.
- Nhung phai doi output va pipeline metric.

### `LEVEL_WEIGHTS = [1.0, 1.0, 1.2]`

De lam gi:
- Tang trong so cho muc L3.

Vi sao L3 duoc 1.2:
- L3 kho hon, du lieu it hon, muc nay chi tiet hon nen can duoc uu tien nhieu hon mot chut.

Vi sao khong tang qua manh:
- Neu de 2.0 hoac 3.0, model co the qua tap trung vao L3 va lam giam chat luong L1, L2.

Neu doi so:
- 1.0, 1.0, 1.0: can bang don gian, de giai thich, nhung co the chua du quan tam L3.
- 1.0, 1.2, 1.5: uu tien muc sau hon, co the cai thien L3 nhung tong the co the dao dong.
- Khong co bao dam la doi trong so se "khac lon" neu dataset va so epoch ngan, nhung thuong se thay ro trade-off giua cac muc.

Cau tra loi an toan:
- Em chon 1.2 cho L3 nhu mot muc uu tien nhe dua tren dac diem nhan it va kho hon. Em khong tang qua cao de tranh mat can bang giua ba muc. Neu thay doi trong so, ket qua co the se dich theo huong tang hoac giam F1 cua tung muc, nen can grid search hoac validation de xac nhan.

## 10. Toi uu hoa va regularization

### `LR = 3e-4`

Vi sao chon:
- Day la learning rate pho bien cho Adam, du on dinh ma van hoc du nhanh.

Neu 1e-3:
- Hoc nhanh hon.
- Co the dao dong hoac khong on dinh.

Neu 1e-4:
- On dinh hon.
- Nhung cham hon, can nhieu epoch hon.

### `weight_decay = 1e-5`

De lam gi:
- Regularization nhe, giam overfit.

Vi sao nho:
- Van ban voi embedding + RNN rat nhay voi regularization qua manh.

### `PATIENCE = 5`

De lam gi:
- Early stopping neu khong con cai thien.

Vi sao chon 5:
- Du de model co co hoi phuc hoi sau cac epoch dao dong nho.

Neu 2:
- Dung som qua, de bo lo epoch tot.

Neu 10:
- Chac an hon nhung ton thoi gian.

### Gradient clipping `max_norm = 5.0`

De lam gi:
- Han che exploding gradient trong RNN.

Vi sao chon 5.0:
- Day la moc rat pho bien, thuong on dinh ma khong cat gradient qua manh.

## 11. Data augmentation bang random crop

Code:
- `MIN_CROP = 20`
- Lay 30%-100% do dai bai goc

De lam gi:
- Moi epoch, model thay nhung doan khac nhau cua cung mot bai.
- Giam phu thuoc vao vi tri co dinh cua thong tin.
- Tang kha nang tong quat hoa khi input thuc te ngan hon.

Vi sao hop ly:
- Bai bao thuong dai, nhan van chu yeu noi trong mot vai doan.

Rui ro:
- Co the crop mat phan chua tin hieu manh nhat.

Neu crop manh hon nua:
- Augmentation manh hon.
- Nhung de tao nhieu nhieu.

Neu tat augmentation:
- Train "sach" hon.
- Nhung model de phu thuoc vao mau toan bo bai.

## 12. Chon checkpoint

Code:
- Luu model theo `global_f1 = (F1_L1 + F1_L2 + F1_L3) / 3`

Y nghia:
- Khong uu tien rieng mot muc.
- Co gang giu can bang ba muc.

Diem can nho:
- `scheduler.step(val_f1s[0])` lai dang theo L1.
- Nghia la co mot su khong hoan toan dong nhat giua "tieu chi dieu chinh LR" va "tieu chi luu checkpoint".

Neu bi hoi:
- Day la mot han che cua implementation hien tai.
- Neu can chuan hon, nen scheduler theo `global_f1` hoac `val_loss`.

## 13. Metric trong notebook train/test

Code:
- `precision`, `recall`, `f1`, `auprc`
- `threshold = 0.3` trong `compute_metrics`

### Vi sao co threshold

Do output la xac suat sau sigmoid, can nguong de quyet dinh gan nhan hay khong.

### Vi sao chon 0.3

- 0.3 de mo hinh "de nhan" hon, thuong tang recall.
- Dieu nay hop ly khi output sigmoid chua sac net, nhat la o muc L3.

Neu chon 0.5:
- Precision thuong tang.
- Recall thuong giam.

Neu chon 0.2:
- Recall co the tang tiep.
- Nhung de du doan du nhan, precision giam.

Cau tra loi an toan:
- Threshold la sieu tham so can tune tren validation. 0.3 duoc chon de can bang precision va recall tot hon cho bo du lieu hien tai. Neu thay threshold, ket qua co the thay doi ro, dac biet la F1 va recall.

## 14. Metric trong notebook evaluation

Code:
- Chuyen `labels` va `probs` sang `argmax`
- Tinh `accuracy`, `micro/macro precision-recall-f1`
- Ve confusion matrix

Y nghia:
- Notebook nay dang xem bai toan nhu multiclass tren tung muc de de doc va phan tich loi.

Vi sao hop ly mot phan:
- Du lieu thuc te gan nhu 1 nhan/muc.
- Confusion matrix can dang nhan roi rac de de nhin.

Nhung can nho:
- Day khong hoan toan cung setting voi notebook train/test dang threshold theo sigmoid.
- Khi bao ve, phai noi ro repo co 2 goc nhin danh gia:
  - mot ben theo multi-label style
  - mot ben theo multiclass style de phan tich confusion matrix

## 15. Micro-F1 va Macro-F1

### Micro-F1

De lam gi:
- Gop tat ca du doan lai de tinh.

Khi nao huu ich:
- Muon biet hieu qua tong the.

Han che:
- Bi anh huong manh boi cac nhan dong mau.

### Macro-F1

De lam gi:
- Tinh F1 tung nhan roi lay trung binh deu.

Khi nao quan trong:
- Khi du lieu mat can bang.
- Phan anh tot hon hieu qua tren nhan hiem.

Cau de hoc thuoc:
- Neu micro-F1 cao ma macro-F1 thap, co the mo hinh dang tot o lop lon nhung kem o lop hiem.

## 16. AUPRC

De lam gi:
- Danh gia quan he precision-recall tren nhieu threshold.

Vi sao quan trong:
- Huu ich khi du lieu mat can bang.
- Tot hon accuracy trong bai toan nhieu nhan hoac nhan hiem.

## 17. Hierarchy diagnostics

Code:
- `parent consistency`
- `strict hierarchical accuracy`

### Parent consistency

De lam gi:
- Kiem tra nhan L3 du doan co "hop cha" voi nhan L2 du doan hay khong.

Neu cao:
- Mo hinh ton trong cau truc phan cap tot hon.

### Strict hierarchical accuracy

De lam gi:
- Kiem tra dong thoi dung ca L2 va L3.

Y nghia:
- Day la metric kho hon F1 rieng tung muc.
- Phu hop khi bai toan can dung dung cay nhan.

## 18. Tai sao L3 kho hon

Can hoc thuoc:
- It du lieu hon.
- Nhan chi tiet hon, de nham hon.
- Nhieu cap nhan gan nghia.
- Loi o L2 co the anh huong den L3 ve mat logic, du code hien tai chua ep cung cha-con.

## 19. Nhung diem hoi dong de hoi van

### Tai sao khong dung Transformer nhu PhoBERT?

Tra loi ngan:
- Muc tieu cua repo hien tai la xay dung baseline co the giai thich ro, tu pipeline den embedding va attention phan cap. HARNN + Word2Vec nhe hon, de huan luyen va de phan tich. Huong mo rong tiep theo la thu PhoBERT de so sanh.

### Tai sao khong dung CrossEntropy?

Tra loi ngan:
- Vi repo duoc thiet ke theo huong mo rong multi-label. Tuy nhien, voi dataset hien tai thi CrossEntropy la mot baseline hop ly va nen duoc so sanh them.

### Trong so L3 = 1.2 co co so gi?

Tra loi ngan:
- Do L3 kho va it du lieu hon nen can uu tien them mot chut. 1.2 la muc tang nhe de tranh lam lech toan bo qua trinh hoc sang L3. Neu doi trong so, ket qua co the thay doi theo huong trade-off giua cac muc, vi vay can xac nhan bang validation.

### Threshold 0.3 co tuy tien khong?

Tra loi ngan:
- Co tinh chat heuristic ban dau, nhung co co so tu validation va trade-off precision-recall. Day la sieu tham so can tune.

## 20. Han che cua implementation hien tai

- Dang co su pha tron giua goc nhin multi-label va multiclass.
- Chua ep cung ranh buoc cha-con khi du doan.
- Dung `BCELoss` thay vi `BCEWithLogitsLoss`, ve mat so hoc chua toi uu.
- Scheduler theo `F1_L1`, checkpoint theo `Global F1`.
- Chua co baseline so sanh ro rang voi cac mo hinh khac.

## 21. Neu muon noi "vi sao chon" mot cach an toan

Mau cau co the dung:

- "Em chon tham so nay vi no giu can bang giua chat luong va chi phi huan luyen trong bo du lieu hien tai."
- "Neu thay doi tham so, ket qua co the khac, nhung huong thay doi thuong la trade-off giua kha nang bieu dien, toc do huan luyen va nguy co overfitting."
- "Gia tri hien tai duoc chon theo kinh nghiem thuc nghiem va tinh on dinh, khong khang dinh la toi uu toan cuc."

## 22. Thu tu hoc nhanh de di bao ve

Neu ban chi con it thoi gian, hoc theo thu tu nay:

1. Bai toan va du lieu
2. Tien xu ly
3. Word2Vec
4. HARNN
5. Loss va LEVEL_WEIGHTS
6. Metric: micro/macro F1, AUPRC, confusion matrix
7. Threshold 0.3 va vi sao L3 kho
8. Han che va huong cai tien

## 23. Cac cau bat buoc phai tra loi duoc

- He thong cua em nhan input gi va output gi?
- Tai sao can tach tu tieng Viet?
- Tai sao dung Word2Vec?
- Tai sao dung HARNN?
- Attention trong model de lam gi?
- Tai sao dung BCELoss?
- Tai sao L3 duoc tang trong so 1.2?
- Vi sao khong chi dung accuracy?
- Micro-F1 va Macro-F1 khac nhau the nao?
- Tai sao ket qua L3 thuong thap hon?
- Han che lon nhat cua mo hinh hien tai la gi?

