# Günlük - Kaggle S6E8 Akıllı Telefon Bağımlılığı

## 2026-08-30 gündüz (son gün) — temsil/FE iyileştirme arayışı + LB teyidi

Kullanıcı yönlendirmesi: "hangi feature'ı daha iyi temsil edebilirdik; veride gözden kaçan temizleme/FE var mı → deneylerle test et".

- **EDA sonuçları (hepsi kesin):**
  - Tüm saat kolonları tam 0.01-grid (1.000 oran), sayımlar tamsayı, age int 18-35. **0.01-offset/sub-grid "identity digit" yok** (offsetler tüm 10 basamakta uniform).
  - Kayıp desenleri tek-kolon ağırlıklı (structureless), `nmiss` vs P(y) düz (~0.71) → maske sinyali yok.
  - ORIG (7500 satır) user_id'ler benzersiz → satırlar arası gizli kullanıcı/latent yok.
  - Per-value determinizm çok yüksek: app_opens split-half r=0.9965, notifications r=0.9931 → neredeyse deterministik lookup, ama additif tekbaşına o kadar güçlü değil: app_opens one-hot LR OOF 0.7349, 6-feature one-hot LR 0.9233, tüm-feature one-hot LR 0.9587 → additif tavan 0.959; kalan 0.011'lik marj ETKİLEŞİMLERDEN, lattice modelleri zaten kapsıyor.
- **Meta+ham özellikler deneyi (S6E6 tarzı meta'ya raw):** CatBoost meta = [98 logit + 43 raw/eng] OOF **0.96985** < logit-only kontrol **0.96988** → meta'ya raw eklemek ZARAR/kazançsız (üyeler ham sinyali tüketmiş).
- **İmputasyon-varyant deneyi (kullanıcı fikri):** 3 imputer (LGB/XGB/Cat) + multi-view (3'ü birden, 27 kolon), hepsi aynı 113-feature pipeline (TE/freq/ORIG ortak), tek seed:
  - `imp_var_lgb` 0.96888 (corr ref 1.0000), `imp_var_xgb` 0.96888 (0.9993), `imp_var_cat` 0.96888 (0.9993), `imp_var_multi` 0.96885 (0.9993) vs referans `gbdt_abd_origfeat` 0.96889.
  - **Karar: imputer modeli / çoklu-görünüm sonucu değiştirmiyor (Δ≤0.00003), ek üye değer vermez.**
  - Script: `scripts/stack_impute_variants_2026-08-30.py` (aşamaları `nn_cache/impvar/*.pkl`'e cache'li, resume-destekli), çıktılar `nn_cache/imp_var_{lgb,xgb,cat,multi}_{oof,test_pred}.npy` + `sub/2026-08-30/imp_var_*_2026-08-30.csv`.
- **Sonuç:** Feature/veri temsilinde kaçış yok — özellik uzayı doymuş. Yol skorda değil, submission seçiminde.
- **LB teyidi (manuel yükleme):** exp_meta3=**0.97108**, exp_meta6=**0.97108**, stack_exp_xgb=**0.97108**, exp_ax=**0.97106** (+0.00104/# OOF 0.97002 — transfer katsayısı meta3 ailesiyle birebir tutarlı), stack_nnls=0.97048 (dekorrelasyonlu, düşük OOF 0.96943 → beklenen). En iyi: **exp_meta3 ortağı 0.97108**.

## 2026-08-30 gece (9. Gün, YARIŞMA SON GÜNÜ — 31 Ağustos 23:59 UTC bitiyor) — otonom gece oturumu

### ⚠️ KRİTİK: Kaggle API key YANLIŞ HESABA bağlı — CLI ile submission ATILMADI
Kullanıcı gece Kaggle API key'ini verdi (`~/.kaggle/kaggle.json`, ACCESS_TOKEN yöntemi) ve
"gece boyunca kendi kendine çalış, submission at" dedi. **Test ederken kritik bir uyumsuzluk
bulundu**: `kaggle config view` bu key'in bağlı olduğu hesabın kullanıcı adının **`dante29`**
olduğunu gösteriyor. Ama gerçek yarışma ilerlemesi (LB=0.97035, 37 submission, teamId
16706543) **`talhatursun` ("Talha Tursun")** adlı FARKLI bir hesapta duruyor (leaderboard
CSV'sinde doğrulandı: `nn_cache/playground-series-s6e8-publicleaderboard-*.csv`, satır 664).
**Bu key ile `kaggle competitions submit` çalıştırmak "dante29" hesabına gider, gerçek
ilerlemeyle hiçbir ilgisi olmaz — bu yüzden gece boyunca CLI'dan HİÇBİR submission atılmadı.**
Genel/açık veri indirme (dataset download, leaderboard okuma) hesap bağımsız olduğu için
sorun değil, bunlar yapıldı. **Sabah ilk iş: kullanıcıdan ya `talhatursun` hesabının doğru
API key'i istenecek, ya da hazırlanan aday submission CSV'leri kullanıcı tarafından web
arayüzünden manuel yüklenecek.** Referans: 1. sıra Chris Deotte, LB=0.97207 (3272 takım).

### Dış OOF kütüphanesi indirildi ve doğrulandı — kullanıma HAZIR, stacking script'i HENÜZ YAZILMADI
`kaggle datasets download -d szymonkapiski/s6e8-oof-library-47-models` ile indirildi,
`data/oof_library/` altına açıldı (manifest.csv, hyperparameters.json, README.md,
train_keys.parquet, test_keys.parquet, oof/oof_<isim>.npy + oof/test_<isim>.npy, 74 model).
**Hizalama doğrulandı**: `train_keys.parquet`'in `id` sırası bizim `train.csv` ile BİREBİR
aynı (pozisyonel hizalama, id-merge'e bile gerek yok), `oof_lookup.npy`'nin AUC'si yeniden
hesaplanınca manifest'teki 0.96853 ile birebir eşleşti — veri bütünlüğü sağlam.
**En iyi bireysel modeller** (manifest.csv, 74 model tam listesi dosyada): naji03/naji05
(OOF 0.96881, yazarın kendi pipeline'ı, kaynak paylaşılmamış), tabm_seed3 (OOF 0.96867,
public LB 0.96967, TabM/pytabkit), **lookup** (OOF 0.96853, tamerlanomralinov'un mimarisi
retrain edilmiş, **blend'e TEK BAŞINA +0.000109 katkı — projedeki EN BÜYÜK tekil katkı**,
diğer modellerle max korelasyon 0.9869), tabm_x12 (0.96849), tabm_deeper (0.96846),
pub_rmlp (0.96844, en güçlü PUBLIC tekil model).
**Kütüphanenin kendi tam 54-model blend'i: OOF=0.96943, LB=0.97062** (README quick-start
kodu: mantıksal-uzayda (logit) LogisticRegression meta-model, honest nested-CV).
**Önemli çelişki notu**: manifest'te "lattice TE" (pair/triple joint target-encoding)
ailesi onlarda GERÇEK kazanç vermiş (lat_lgbm +0.00108, latwide +0.0002 daha) — bizim
08-20'de aynı fikri (`pair-lattice TE`) denediğimizde CV+0.00027 ama LB-0.00018 (net negatif,
projenin İLK CV/LB çelişkisi) bulmuştuk. Çelişkiyi çözmeye çalışmaya gerek yok — onların
sonucu kendi LB'lerinde (0.97062) doğrulanmış, muhtemelen bizim 4-hand-picked-pair
uygulamamız onların 36-pair "latwide" + üçlü versiyonundan metodolojik olarak farklıydı
(smoothing/örneklem büyüklüğü/hücre sayısı). Bizim kendi sonucumuza güvenmeye devam,
ama onların OOF'unu (zaten hesaplanmış, bize maliyeti sıfır) blend'e almakta sakınca yok.

**SIRADAKİ OTURUMUN İLK İŞİ (kullanıcı "dur, yarın devam ederiz" dedi, gece burada durdu):**
1. `data/oof_library/`'deki 74 model + bizim 2 kendi OOF'umuzu (gbdt_abd_origfeat OOF=0.96889,
   nn_missingaug_featfull OOF=0.96791) BİRLEŞTİREN gerçek bir stacking/greedy-blend script'i
   yaz (README'nin quick-start kodu iyi bir başlangıç noktası: logit-uzayında nested-CV
   LogisticRegression meta-model). Hedef: kütüphanenin kendi 0.96943'ünü VE bizim mevcut
   en iyimizi (0.96927) aşan bir kombinasyon bulmak.
2. Kaggle hesap sorunu çözülmeli: kullanıcıdan `talhatursun` hesabının doğru API key'i
   alınmalı (CLI submission için) — o zamana kadar hazırlanan CSV'ler web'den manuel
   yüklenebilir.
3. 3-seed bagging (`scripts/multiseed_gbdt_origfeat_2026-08-30.py`, arka planda bırakıldı,
   `nn_cache/multiseed_gbdt_origfeat_2026-08-30.log`) sonucuna bak — muhtemelen sabaha
   tamamlanmış olacak.
4. **Yarışma BUGÜN (31 Ağustos 23:59) bitiyor** — zaman çok kısıtlı, önceliği stacking'e ver.

## 2026-08-29 (8. Gün, 08-21'den 8 gün sonra devam)

### Oturum baglami: yarisma bitisine 2 gun kala, hiz-oncelikli mod
Kullanici 08-21 sonundaki acik maddelerden **#1 (NN'e GBDT'nin 113-ozellik setini eksiksiz
verme) ve #3'u (ResNet-tarzi mimari-cesitlilik) yapmaya karar verdi, #2'yi (3-seed bagging)
bu turda atladi**. Onemli metodoloji karari: **submission hakki bol oldugu ve sure kisitli
oldugu icin local CV/OOF tarama-once-LB-sonra kurali gevsetildi - kucuk kod-dogrulugu
sanity-check disinda, dogrudan LB'de test etme stratejisine gecildi.** (Bu, negatif/net
sonuclari hala LB'ye atmama kararini engellemedi - asagida gorulecegi gibi ResNet'in net
negatif OOF sonucu LB'ye atilmadi, sadece "borderline pozitif" sinyal LB'de dogrulandi.)

### Madde 1: NN'e GBDT'nin eksiksiz feature setini vermek - BASARILI, LB DOGRULANDI
`scripts/nn_data_prep_kfold_featfull_2026-08-29.py`: Lookup-Transformer'in PLR-only turetilmis
sutun sayisi 8 -> 62'ye cikarildi (eski 8 + kategori-A'nin NN'de eksik kalan 8 orani +
diff_daily_sum_clean + 4 gap/range + 12 decimal-lattice + 29 ORIG-CDF feature, hepsi
`lgbm_orig_features_lb_2026-08-21.py` ile BIREBIR AYNI kod kullanilarak). TE/freq-encoded
sutunlar BILEREK eklenmedi (NN'in lookup embedding'iyle ayni bilgiyi baska yoldan kodluyorlar,
korelasyonu arttirma riski). ORIG-CDF hesaplamasi (KDE score_samples 987K satirda) ~11 dk surdu.
9 ham surekli sutun + 4 kategorik sutun DEGISMEDI (NN'in kendine ozgu exact-deger lookup+PLR+
attention mekanizmasi korundu).
`scripts/nn_train_kfold_missingaug_featfull_2026-08-29.py`: ayni production ayarlari
(mask_prob=0.40, epochs=40/patience=8, GBDT ile ayni StratifiedKFold(5,seed=42)) - 76 token'lik
(CLS+9+62+4) sequence nedeniyle fold basi ~28 dk (eski 22 token'lik ~11 dk'nin ~2.5 kati),
toplam 142 dk.
- **NN solo OOF: 0.96717 -> 0.96791 (+0.00074)** - dar-feature missingaug NN'den net iyi.
- **GBDT+NN blend OOF: 0.96914 -> 0.96927 (+0.00013)**, W_GBDT=0.656. Spearman korelasyonu
  neredeyse DEGISMEDI (0.9814 -> 0.9813) - yani kazanc gercek YENI bilgiden geliyor, "NN'i
  iyilestir = GBDT'ye daha cok benze" tuzagina bu sefer DUSMEDI (feature'lar GBDT'nin zaten
  bildigi bilgiyi TE/freq formatinda degil, ham-oran/CDF formatinda verdigi icin olabilir).
- Δ=0.00013, kalibrasyon bandinda (0.00005-0.00017) - **kullanici karariyla dogrudan LB'de
  test edildi**: `sub/2026-08-29/blend_gbdt_origfeat_nn_featfull_w66_2026-08-29.csv`.
- **LB SONUCU: 0.97025 -> 0.97035 (+0.00010) - YENI EN IYI SKOR.** OOF'un ongordugu yonde ve
  yakin buyuklukte. **Kalibrasyon oruntusunun 4. TEKRARI** (missingaug-NN: OOF+0.00014->
  LB+0.00018, ORIG-CDF solo: OOF+0.00009->LB+0.00013, ORIG-CDF+NN blend: OOF+0.00005->
  LB+0.00009, simdi bu: OOF+0.00013->LB+0.00010) - bu bant artik guvenilir sekilde erken
  eleme yapilmadan LB'de test edilmeli.
- **PRODUCTION artik bu submission.** Script'ler kalici: `nn_data_prep_kfold_featfull_2026-08-29.py`,
  `nn_train_kfold_missingaug_featfull_2026-08-29.py`, `blend_gbdt_origfeat_nn_featfull_2026-08-29.py`.

### Madde 3: ResNet-tarzi tabular NN (mimari-cesitlilik) - NET NEGATIF, KAPANDI
Hipotez: GBDT'nin AYNI 113-ozellik setini (raw+TE+freq+imp+ORIG-CDF, `resnet_data_prep_2026-08-29.py`
- `lgbm_orig_features_lb_2026-08-21.py`'nin feature-muhendisligi kismi BIREBIR kopyalandi,
model egitimi HARIC) attention'siz, sadece residual-MLP bloklu duz bir feed-forward age
(`resnet_model.py`: ResidualBlock yiginlari) ile vermek, Lookup-Transformer'dan daha DUSUK
GBDT-korelasyonlu (gercek mimari-cesitliligi) bir model uretebilir mi diye test edildi.
- **v1** (`resnet_train_kfold_2026-08-29.py`, hidden=256, n_blocks=5, epochs=60): 5-fold
  egitim sadece **6.6 dk surdu** (Lookup-Transformer'in 142 dk'sindan cok ucuz). Solo
  OOF=0.96452. GBDT ile Spearman korelasyonu **0.9791** - Lookup-Transformer'in 0.9814'unden
  GERCEKTEN DUSUK (hipotez dogru yonde calisti). Ama blend agirlik taramasi W_GBDT=0.98'e
  cikti (ResNet'e ~sifir agirlik), blend OOF=0.96889 = GBDT-solo ile AYNI, production'dan
  **-0.00025 KOTU**. Sebep: solo kalite acigi (GBDT 0.96889 vs ResNet 0.96452, fark 0.00437)
  korelasyon avantajini yutuyor.
- **v2** (`resnet_train_kfold_v2_2026-08-29.py`, hidden=384, n_blocks=7, epochs=100,
  dropout=0.25): kapasite arttirma DENENDI, solo OOF=0.96380 - v1'den DAHA KOTU (asiri
  dropout/yetersiz veri-tekrari, overfitmeden once erken plato). Bu yon (kapasite artirma)
  daha fazla zorlanmadi.
- **3'lu blend taramasi** (GBDT+NN-featfull+ResNet, coarse grid): optimum ResNet agirligi
  **0 cikti** - NN-featfull zaten var oldugunda ResNet'in hicbir ek katkisi yok.
- **LB'ye ATILMADI** (net negatif OOF sonucu, kullanicinin "borderline pozitif direkt LB'de
  test" karari sadece OOF-pozitif sinyaller icin gecerliydi, bu net negatifti).
- **Ders**: mimari-cesitlilik hipotezi DOGRU yonde calisti (korelasyon gercekten dustu) ama
  yeterli degildi - cesitlilik ancak minimum bir solo-kalite esigini gecen bir modelde blend'e
  yansiyor, cok zayif bir model ne kadar "farkli" olursa olsun agirlik alamiyor. Bu acik
  madde artik KAPALI, tekrar denenmeyecek (belki cok daha buyuk bir mimari yatirimi - ornegin
  FT-Transformer/TabNet tam egitimi - farkli sonuc verebilir ama bu turun kapsaminda degil).

### GUN SONU OZET (2026-08-29/30)
- Gun basi LB: 0.97025 -> **Gun sonu LB: 0.97035** (+0.00010)
- Madde 1 (NN feature-completeness): BASARILI, LB dogrulandi, kalibrasyon oruntusunun 4. tekrari
- Madde 2 (3-seed bagging): bu turda ATLANDI (kullanici karari)
- Madde 3 (ResNet mimari-cesitlilik): calisti ama yetersiz kaldi, NET NEGATIF, KAPANDI
- **PRODUCTION**: `sub/2026-08-29/blend_gbdt_origfeat_nn_featfull_w66_2026-08-29.csv`
- **YARISMA BITISINE 1 GUN KALDI (31 Agustos)** - final 2 submission secimi gundemde,
  CV'ye guven kurali hala gecerli (bkz. 08-21 girdisindeki madde 7).
- Acik maddeler (oncelik sirasiyla, sonraki oturum icin):
  1. Okunmamis yuksek-skorlu kaynaklar: Ripon C. Malo'nun "55-Model Stack" (LB 0.97068),
     tomasa2'nin "Stacking" bolumu, Georgy Mamarin'in gaming_hours notu.
  2. 3-seed bagging (GBDT icin) - hala denenmedi, dusuk risk.
  3. **Final submission secimi**: yarisma bitimine cok az sure kaldigi icin, secilecek 2
     submission'in birinin mevcut PRODUCTION (LB=0.97035, en yuksek dogrulanan CV+LB) olmasi
     gerekiyor - LB rank'ine degil CV'ye guven kurali hala gecerli.
  4. Kapanmis yonler: grup-bazli/kaskad missingness-aug, pair-lattice TE, NN
     hiperparametre/Optuna eksen, **ResNet-tarzi mimari-cesitlilik (bu oturumda kapandi)**.

İlgili: [[feedback_kaggle_data_first_workflow]], [[feedback_kaggle_local_f1_no_push]]

## 2026-08-21 (7. Gün)

### Missingness-augmentation — açık madde #1, İLK büyük solo-NN kazancı ama blend'e katkısı sınırda
2026-08-20 gün sonunun "ilk iş" maddesi test edildi: egitim sirasinda rastgele EK deger
maskeleme (zaten eksik olmayan hucreleri de eksik gibi gostermek), bir egitim MEKANIZMASI
degisikligi. Uygulama: `nn_common.py`'ye `augment_missingness()` + `train_model(...,
missing_aug_prob=...)` eklendi (sadece cont-lookup ve plr-only akislarina, sadece egitim
batch'lerinde, val'a uygulanmiyor).
- **Izole tek-split tarama** (`nn_experiment_missingaug_2026-08-21.py` + round2,
  referans mask_prob=0: val AUC=0.96507): mask_prob 0.05→0.20 monoton arttı
  (+0.00046→+0.00112), 0.30-0.50 arasında plato (+0.00133/+0.00141/+0.00135).
  **En iyi: mask_prob=0.40, val AUC=0.96648 (+0.00141)** — şimdiye kadarki en büyük
  tekil NN kazancı (feature-genişletmenin +0.00193'üyle kıyaslanabilir, HPO'nun
  +0.0005'inden kat kat büyük).
- **Tam 5-fold k-fold retrain** (`nn_train_kfold_missingaug_2026-08-21.py`, mask_prob=0.40,
  epochs=40/patience=8, GBDT ile AYNI StratifiedKFold): **NN OOF AUC=0.96717**, eski NN
  k-fold'un 0.96576'sından **+0.00141**, honest OOF'ta da doğrulandı (tek-split'le birebir
  tutarlı). Dış referans notebook'un (0.96853) açığı yarı yarıya kapandı (0.00277→0.00136).
  Süre: 55.2 dk (5 fold, ~11 dk/fold).
- **Blend potansiyeli** (`blend_gbdt_nn_missingaug_2026-08-21.py`): GBDT+missingaug-NN
  rank-blend en iyisi **OOF=0.96909 (W_GBDT=0.718)**, eski blend'in 0.96895'inden
  **+0.00014 — kalibrasyon eşiğinin (0.00015) TAM SINIRINDA**, "güvenilir" değil.
  **Sebep bulundu**: GBDT-NN Spearman korelasyonu **0.9806**'ya çıktı (eskisinden bile
  yüksek) — NN kalitesi artarken GBDT'ye daha çok benzedi, çeşitlilik payı büyümedi.
  **"NN'i iyileştir ≠ blend'e katkı sağla" deseninin 4. tekrarı**, artık güvenilir bir
  proje-kuralı: NN'in KENDİ AUC'sini yükseltmek otomatik olarak blend'e yansımıyor,
  GBDT'nin YAKALAMADIĞI bilgi kanallarına ihtiyaç var (henüz bulunamadı).
- **Kullanıcı kararı**: gürültü-sınırı riskine rağmen tek başına LB'de test edilecek
  (`sub/2026-08-21/blend_gbdt_nn_missingaug_w72_2026-08-21.csv`, W_GBDT=0.718). Kaggle
  CLI bu makinede kurulu değil — kullanıcı manuel yükledi.
  **LB SONUCU: 0.97016 — YENİ EN İYİ SKOR.** Eski en iyiden (0.96998) **+0.00018**,
  LB gürültü tabanının (±0.00014) ÜSTÜNDE — gerçek bir kazanç, OOF'un (+0.00014) tahmin
  ettiğinden bile biraz büyük çıktı (aynı yönde, birbirine yakın — sinyal gerçekmiş, sadece
  kalibrasyon eşiğinin tam sınırındaymış, "noise" diye elenmemeli). **Bu skor, "dürüst
  tavan" sandığımız tomasa2'nin 0.97014'ünü de (+0.00002, gürültü içinde ama nominal
  olarak) GEÇTİ.** **PRODUCTION artık bu submission: `blend_gbdt_nn_missingaug_w72_
  2026-08-21.csv`.** Missingness-augmentation (mask_prob=0.40) kalıcı olarak NN üretim
  hattına alınmalı (nn_model.py/nn_train_kfold.py'ye de yansıtılabilir, düşük öncelik -
  şu an için augmented k-fold script'i ayrı duruyor, çalışıyor).
- Script'ler: `scripts/nn_experiment_missingaug_2026-08-21.py`,
  `scripts/nn_experiment_missingaug_round2_2026-08-21.py`,
  `scripts/nn_train_kfold_missingaug_2026-08-21.py`,
  `scripts/blend_gbdt_nn_missingaug_2026-08-21.py`.
  Checkpoint'ler: `nn_cache/missingaug_model_fold{0-4}.pt`,
  OOF/test: `nn_cache/nn_missingaug_kfold_oof.npy`, `nn_cache/nn_missingaug_kfold_test_pred.npy`.

### Grup-bazlı (kaskad) missingness-augmentation — hipotez ÇÜRÜTÜLDÜ, kapandı
Hücre-bazlı missingaug'ın blend katkısının sınırda kalma sebebi (GBDT korelasyonu 0.9806'ya
çıkması) icin bir düzeltme denendi: türetilmiş oran/fark sütunlarının (ratio_social_daily vb.)
missing bayrağını ham sütundan BAĞIMSIZ rastgele maskelemek yerine, gerçek pandas NaN-yayılma
semantiğini taklit eden bir KASKAD kurala bağladık (`augment_missingness_grouped()`,
`nn_common.py` — `PLR_DEPENDENCIES` haritası, any/all kuralları). Hipotez: hücre-bazlı
maskeleme "kaçak bilgi" bırakıyordu (ham sütun maskeliyken türetilmiş oranı hâlâ görünürdü,
model gerçek değeri oradan çıkarabiliyordu) — bunu kapatmak GBDT'den farklı bir çözüm yoluna
zorlar, çeşitlilik kazandırır sanılmıştı.
- İzole tek-split sweep (`nn_experiment_missingaug_grouped_2026-08-21.py`, mask_prob
  0.1/0.2/0.3/0.4): en iyi mask_prob=0.30, val AUC=0.96646 — hücre-bazlı en iyiye (0.96648)
  neredeyse eşit (solo kalite açısından fark yok).
- **Tam 5-fold k-fold** (`nn_train_kfold_missingaug_grouped_2026-08-21.py`, mask_prob=0.30):
  **NN OOF=0.96684** — hücre-bazlının 0.96717'sinden **-0.00033 KÖTÜ** (grup-bazlı kısıt,
  modelin işini zorlaştırmış, solo kaliteye net zarar vermiş).
  **GBDT-NN Spearman korelasyonu: 0.9808 — hücre-bazlının 0.9806'sından DEĞİŞMEDİ** (fark
  gürültü seviyesinde). **Hipotez ÇÜRÜTÜLDÜ**: kaskad maskeleme çeşitlilik kazandırmadı.
  Blend OOF=0.96902 (W_GBDT=0.756) — hücre-bazlı blend'in 0.96909'undan **-0.00007, hafif
  kötü**. Hem solo hem blend her açıdan hücre-bazlıdan geride — **LB'ye atmaya gerek yok,
  bu yön KAPANDI.**
- **Ders**: blend'in düşük çeşitlilik sorununun kökeni türetilmiş-sütun "kaçağı" değilmiş —
  başka bir yerden geliyor olmalı (muhtemelen: NN ve GBDT'nin ikisi de aynı ham sinyale
  (raw+TE feature'lar) erişiyor olması, mimari farkına rağmen benzer karar sınırlarına
  yakınsıyorlar). Gerçek çeşitlilik için muhtemelen NN'e GBDT'nin GÖRMEDİĞİ bir bilgi
  kanalı gerekir (feature değil, mimari/veri kaynağı düzeyinde bir fark) — kolay bir sonraki
  adım değil, düşük öncelikli araştırma konusu.
  Script'ler: `scripts/nn_experiment_missingaug_grouped_2026-08-21.py`,
  `scripts/nn_train_kfold_missingaug_grouped_2026-08-21.py`,
  `scripts/blend_gbdt_nn_missingaug_grouped_2026-08-21.py`.

### ORIG-CDF feature'ları (başka bir Kaggle notebook'undan esinlenildi) — NEGATİF, kapandı
Kullanıcının paylaştığı bir notebook ("Feature Engineering: What Moved the LB Scores"),
ORIG'i (jayjoshi37, 7500 satır gerçek veri) harici referans dağılım olarak kullanan 29
feature ekliyordu: empirical CDF pozisyonu, class-conditional CDF gap (y0/y1 dağılımlarına
göre ayrı CDF farkı), ORIG medyanlarına uzaklık, ORIG quantile-binned target ortalaması,
KDE log-likelihood ratio. Kaynak notebook'ta bu ailenin tamamı kendi CV'sinde pozitif
bildirilmişti, feature importance'ta ilk 5'te 2 tanesi vardı.
- Kullanıcı ORIG csv'sini indirip `data/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`
  olarak ekledi. Aynı 29 feature (kaynak notebook'un ablation'da onayladığı sütun seçimiyle
  birebir) üretilip PRODUCTION feature setimize (K1+A+B+D, 84 özellik) eklendi.
  **Leak kontrolü**: ORIG, train ile TAM ÖRTÜŞEN satırlar için hash'lendi — **0 örtüşme
  bulundu** (7500→7500), jeneratörün ORIG'i doğrudan örneklemediğini doğruluyor (08-19
  bulgusuyla tutarlı).
- İzole LightGBM-solo 5-fold CV (`scripts/lgbm_orig_features_2026-08-21.py`):
  Baseline (ORIG'siz, 84 özellik) OOF=0.96856, ORIG-eklenmiş (113 özellik) OOF=0.96858.
  **Delta: +0.00002 — kesin gürültü, kalibrasyon eşiğinin (0.00015) çok altında.**
  **LB'ye atılmadı, bu yön KAPANDI.**
- **Ders**: kaynak notebook'ta işe yaramış olması muhtemelen onların TE/frekans kurulumunun
  bizimkinden daha zayıf olmasından — bizim exact-value TE'imiz zaten 691K satırın TAMAMINDAN
  her exact değer için doğrudan hedef oranını öğreniyor, bu 7500 satırlık ORIG'den türetilen
  dolaylı istatistiklerden (92 kat daha az veri, in-domain değil) çok daha güçlü. "tomasa2'nin
  TE-everything kaldıracının büyük kısmı bizde zaten vardı" bulgusunun bir tekrarı (2026-08-19).

### ORIG-CDF feature'ları — kullanıcı kararıyla yine de LB'de test edildi, KÜÇÜK POZİTİF
CV'de gürültü (+0.00002) çıkmasına rağmen kullanıcı LB'de test etmek istedi (missingness-aug
örneğindeki gibi CV'nin küçümsediği gerçek sinyal ihtimaline karşı). Tam 3-model blend
(`scripts/lgbm_orig_features_lb_2026-08-21.py`, LGB+XGB+CatBoost, K1+A+B+D+29 ORIG):
- OOF=0.96888 (referans 0.96879'dan +0.00009)
- **LB=0.96998 (referans 0.96985'ten +0.00013)** — yine gürültü eşiğinin (0.00015) altında
  ama pozitif, OOF'un öngördüğünden biraz büyük. **Kalibrasyon örüntüsünün 2. tekrarı**
  (missingness-aug'da da OOF+0.00014 → LB+0.00018 olmuştu) — bu bandın "muhtemelen küçük
  ama gerçek sinyal" olduğu tezini güçlendiriyor.
- **Feature importance ilginç**: 5 ORIG feature'ı global top-20'de (`social_media_hours__
  orig_cdf` 7., `daily_screen_time_hours__orig_cdf` 8., `weekend_screen_time__orig_cdf` 13.,
  `social_media_hours__orig_q50_dist_y1` 14., `weekend_screen_time__orig_kde_llr` 17.) —
  GBDT'nin gerçekten değerli bulduğu ama çoğunlukla TE/imputed sütunlarla REDUNDANT olan
  bir sinyal (importance yüksek, net AUC katkısı küçük çünkü zaten örtüşüyor).
- **NN ile yeniden blend edildi** (`scripts/blend_gbdt_origfeat_nn_missingaug_2026-08-21.py`):
  GBDT+ORIG OOF=0.96889 (LB=0.96998 ile tutarlı), korelasyon NN ile 0.9814 (eskisi 0.9806,
  neredeyse degismedi — yine "kaliteyi artır, çeşitliliği artırmaz" örüntüsü). Blend
  OOF=0.96914 (W_GBDT=0.734), eski blend'in 0.96909'undan **+0.00005 — çok küçük**.
  **Kullanıcı yine de LB'de test etti: LB=0.97025 — YENİ EN İYİ, eski en iyiden (0.97016)
  +0.00009.** OOF'un öngördüğünden (+0.00005) yine büyük çıktı — **kalibrasyon örüntüsünün
  3. TEKRARI** (missingaug OOF+0.00014→LB+0.00018, ORIG-solo OOF+0.00009→LB+0.00013, bu
  OOF+0.00005→LB+0.00009). Bu bant artık güvenle "küçük ama gerçek sinyal" kabul edilmeli.
  **PRODUCTION artık bu submission**: `sub/2026-08-21/blend_gbdt_origfeat_nn_missingaug_w73_
  2026-08-21.csv` (W_GBDT=0.734). ORIG-CDF feature'ları kalıcı olarak GBDT feature setine
  eklendi (`scripts/lgbm_orig_features_lb_2026-08-21.py` üretim reçetesi).

### Yan iş: discussion + companion notebook paylaşıldı (Kaggle topluluğuna)
Kullanıcı missingness-augmentation bulgusunu Kaggle Discussion olarak paylaştı (jenerik/
mimari-bağımsız anlatım, bizim GBDT/NN/blend detaylarımızı açık etmeden). Companion notebook
da (`scripts/missingness_augmentation_demo.ipynb`, basit/jenerik MLP + missing-flag mimarisi,
150K/30K alt-örneklem) hazırlanıp CPU'da uçtan uca doğrulandı — ilk halinde (10 epoch, LR
schedule yok) sonuç TERS çıkmıştı (augmentation zararlı görünüyordu), 30 epoch + LR scheduler
eklenince gerçek örüntüyle (mask_prob 0.1-0.2'de fayda, 0.4'te tersine dönme) tutarlı hale
geldi. Ders: kısa/sabit epoch bütçesiyle test edilen missingness-augmentation yanıltıcı
negatif sonuç verebilir — yeterli epoch + LR decay olmadan güvenilir ölçülemez.

### GÜN SONU ÖZET (nihai — oturum burada bitti, kullanıcı "yarın bakarız" dedi)
- Gün başı LB: 0.96998 → **Gün sonu LB: 0.97025** (missingaug-NN ile +0.00018, sonra
  ORIG-CDF feature'larıyla +0.00009 daha) — "dürüst tavan" tomasa2'yi (0.97014) rahat geçtik.
- **3 kez doğrulanan kalibrasyon dersi (bugünün en değerli çıktısı)**: Δ=0.00005-0.00017
  aralığındaki OOF sinyalleri "gürültü" diye elenmemeli — üçü de (missingaug-NN blend,
  ORIG-CDF GBDT-solo, ORIG-CDF+NN blend) LB'de pozitif çıktı, hatta OOF'un öngördüğünden
  büyük. Kullanıcının ısrarla "yine de dene" demesi bu üç kazancın (+0.00027 toplam)
  hepsini sağladı — eski kuralımız (Δ<0.00015 atma) hepsini kaçırırdı.
- **PRODUCTION (güncel, doğru): `sub/2026-08-21/blend_gbdt_origfeat_nn_missingaug_w73_
  2026-08-21.csv`** (GBDT: K1+A+B+D+29 ORIG-CDF feature, W_GBDT=0.734, missingaug-NN
  mask_prob=0.40 ile blend). GBDT üretim reçetesi: `scripts/lgbm_orig_features_lb_2026-08-21.py`.
  NN üretim reçetesi: `scripts/nn_train_kfold_missingaug_2026-08-21.py` (mask_prob=0.40).
- **YARIN İÇİN BAŞLANGIÇ NOKTASI — açık maddeler (öncelik sırasıyla):**
  1. NN'e GBDT'nin 84-özellik setini (artık +29 ORIG-CDF ile 113) daha eksiksiz vermek —
     dikkat: blend çeşitliliğini bozmadan, NN'in kendine özgü lookup+PLR+attention
     mekanizmasını koruyarak (geçmişte "feature genişletme" NN kalitesini artırmış ama
     blend'e yansımamıştı — aynı riske dikkat).
  2. 3-seed bagging (GBDT için) — düşük risk, bu turda hâlâ denenmedi, geçmişte küçük
     gerçek kazanç vermişti.
  3. ResNet-tarzı bir tabular NN (attention'sız, residual MLP blokları) — gerçek blend
     çeşitliliği için mimari-seviyesi alternatif, ciddi yatırım, düşük öncelik.
  4. **Kapanmış (tekrar deneme):** grup-bazlı/kaskad missingness-augmentation
     (korelasyonu hiç değiştirmedi, hem solo hem blend kötüleşti), ORIG-CDF'in kendisi
     DEĞİL ama "ORIG verisiyle daha fazla ne yapılabilir" sorusu sorulmadan önce önce
     "bizim TE'nin zaten neyi kapsadığı" kontrol edilmeli (ORIG-CDF'in kendisi zaten
     production'da, kapanmış olan şey CV'nin küçümsediği-ama-gerçek-çıkan sinyal değil,
     grup-bazlı maskeleme denemesiydi).
  5. Kullanıcı bugün ayrıca bir Kaggle discussion + companion notebook paylaştı
     (missingness-augmentation bulgusu, jenerik/mimari-bağımsız anlatım) — takip:
     yorum/tepki gelirse bakılabilir, aktif bir açık madde değil.

İlgili: [[feedback_kaggle_data_first_workflow]], [[feedback_kaggle_local_f1_no_push]]

## 2026-08-20 (6. Gün)

### GÜN SONU ÖZET (oturum yarıda bırakıldı, devam ediyor)
- **Gün başı LB: 0.96996 → Gün sonu LB: 0.96998** (+0.00002, istatistiksel olarak DÜZ —
  LB'nin çözünürlük tabanının (±0.00014) altında, "yeni en iyi" demek yanıltıcı olur).
- Bugün denenen **9 fikrin 8'i negatif/nötr çıktı** — ama hepsi ucuza (~5-6 saat toplam,
  çoğu arka planda) test edildi ve net kapatıldı, tekrar denemeye gerek yok. Detaylar aşağıda.
- **En önemli stratejik bulgu (gün sonunda ortaya çıktı):** kendi Lookup-Transformer NN'imiz,
  AYNI mimari fikrini kullanan dış referanslardan ciddi geride — tamerlanomralinov'un
  orijinali LB=0.97041, bir topluluk notebook'unun (74-model OOF-library) kendi foldunda
  retrain'i OOF=0.96853, bizimki sadece OOF=0.96576 (k-fold). **~0.003-0.005 AUC'lik bu açık,
  bugün denenen HER feature-engineering fikrinden (hepsi <0.0003) kat kat büyük** — asıl
  kaldıraç burada, GBDT feature mühendisliği değil. Gün sonunda buna yöneldik, ARAŞTIRMA
  YARIDA KALDI (aşağıda "yarın için başlangıç noktası"na bak).

### 1. NN için gerçek k-fold OOF (açık madde #1'di, tamamlandı)
Önceki oturumun tek 90/10 split'i yerine GBDT ile BİREBİR AYNI
`StratifiedKFold(n_splits=5, seed=42)` üzerinde 5-fold NN eğitildi.
Script'ler: `scripts/nn_data_prep_kfold.py`, `scripts/nn_train_kfold.py` (checkpoint'li,
her fold sonunda model+kısmi OOF kaydediyor, ~31 dk sürdü).
- **NN k-fold OOF AUC = 0.96576** (90/10 split'in 0.96507'sinden daha iyi, beklenen).
- Tam 691K satırlık OOF üzerinde GBDT (`nn_cache/gbdt_abd_oof.npy`, 0.96879) ile ağırlık
  taraması: **W_GBDT=0.82 optimal, blend OOF=0.96895** (GBDT-solodan +0.00016).
  Script: `scripts/blend_gbdt_nn_kfold_optimal.py`.
- **LB: 0.96998** — istatistiksel olarak düz (eski ad-hoc yöntemin 0.96996'sından ölçülemez
  fark). Sebep: NN'in kendi kalitesi arttı ama GBDT ile korelasyonu değişmedi (Spearman
  hâlâ ~0.973) — "NN'i iyileştir ≠ blend'e katkı sağla" deseninin 3. tekrarı.
- **Asıl kazanç altyapı**: artık tüm veri için gerçek NN-OOF var, gelecekte 3. bir model
  eklenirse (ya da NN iyileştirilirse) doğru ölçüm yapılabilir.

### 2. Pair-lattice TE — CV'de iyi, LB'de KÖTÜ (nadir CV/LB çelişkisi, KAPANDI)
Kullanıcının paylaştığı bir Kaggle notebook'undan (74-model OOF-library, S6E8) esinlenerek:
9 ham sayısal sütunun TÜM C(9,2)=36 çiftine joint target-encoding eklendi ("wide pair
lattice", o notebook'ta LGB/XGB/CatBoost'ta tutarlı +0.0002 civarı vermişti).
- İzole LGBM testi (36 çift): CV 0.96856→0.96863 (+0.00007, gürültü sınırında ama pozitif).
- Yoğunluk-filtreli daraltma denendi (sadece age-ağırlıklı 10 "yoğun" çift) — **-0.00012,
  hipotez TERSİNE çıktı**: age-çiftleri zaten var olan tekil age-TE ile redundant, zarar
  veriyor. Sinyal aslında 36 seyrek çiftin TOPLAMINDAN geliyormuş (genişlik > yoğunluk).
- Tam 3-model blende taşındı (`scripts/lgbm_xgb_cat_ABD_pairlattice_2026-08-20.py`):
  CV **0.96879→0.96906 (+0.00027)**, kalibrasyon eşiğine yakın, güvenilir bekleniyordu.
  XGBoost'ta özellikle büyük sıçrama vardı (+0.00077).
  **LB: 0.96967 — eski GBDT-solo referansından (0.96985) -0.00018, KÖTÜLEŞTİ.**
  **Bu projede İLK KEZ görülen gerçek CV/LB çelişkisi** (LB gürültü tabanının, ±0.00014,
  biraz dışında ve CV'nin işaret ettiği yönün tersi). Muhtemel sebep: 36 çiftin 28'i çok
  seyrek (hücre başına 2.7-18 satır), OOF-TE metodolojik olarak leakage'a karşı temiz olsa
  bile GBDT bu seyrek hücrelerdeki CV-setine-özgü rastgele örüntüleri ezberleyebiliyor -
  test setinin farklı spesifik değer kombinasyonlarında tekrarlanmıyor. **KAPANDI, tekrar
  denemeye gerek yok** (ne tam 36 çift ne yoğunluk-filtreli alt kümesi).

### 3. Üç küçük ek deneme, hepsi negatif (KAPANDI)
- **Missingness-regime blend ağırlığı**: GBDT/NN ağırlığını eksik-veri miktarına göre
  (tam/1-3 eksik/4+ eksik) ayrı ayrı optimize etmek — 74-model notebook'unda işe yaramıştı
  ama bizim 2-modelli setimizde global ağırlık (W=0.82) zaten her rejimde optimale çok
  yakın çıktı, kazanç **~0.000003** (ölçülemez). Sebep: onlarda 74 farklı modelin gerçek
  güvenilirlik farkları vardı, bizde sadece 2 model var, regime-bağımlı fark yok.
- **Logit-uzayında lojistik regresyon stacking** (rank-blend yerine): honest OOF AUC=0.96885,
  mevcut rank-blend'den (0.96895) **-0.00010 daha kötü**. 2-modelli basit durumda rank-blend
  + ağırlık taraması, lojistik meta-modelden daha iyi çalışıyor.
- **Doğrusal-skor feature** (ham+imputed sütunlar üzerinde honest k-fold OOF lojistik
  regresyon skorunu GBDT'ye ekstra feature olarak vermek — "ağaçlar diyagonal/çapraz karar
  sınırlarında zayıf" ilkesinden esinlenilen yeni bir fikir): **-0.00002, gürültü.**
  LR'nin kendi solo gücü zayıftı (AUC=0.905 vs ağaçların 0.969), yeni bilgi katmadı.
  Script: `scripts/lgbm_linearscore_2026-08-20.py`.

### 4. İki araştırma fork'u — outlier ve kombinasyon taraması (KAPANDI, actionable bulgu yok)
Kullanıcı "kasıtlı outlier" ve "tüm kombinasyonları tara" fikirlerini attı, 2+1 paralel
fork ile kapsamlı araştırıldı:
- **Outlier/enjekte-değer analizi**: klasik outlier temizliği VE fiziksel-tutarsızlık
  temizliği **ölü yön** (jeneratör sınırları net kırpılmış, hiçbir kısıt ihlali yok,
  `daily≥social+gaming+work` %0 ihlal). İlginç ama actionable olmayan bulgu:
  `weekend_screen_time=7.67` değerinde n=2986, hedef oranı 0.336 (komşu 0.449, prior 0.709)
  — ama zaten mevcut raw+TE bunu tam yakalıyor (TE tahmini 0.33628, gerçek 0.33590, doğrulandı).
  "Eşzamanlı spike" etkileşimi de test edildi, korelasyon r=0.005, kapalı.
- **Sistematik kombinasyon taraması**: kategorik-üçlü joint encoding (gender×stress_level×
  academic_work_impact, 48 hücre) additive'den sapmıyor (AUC 0.5135 vs 0.5142, fark yok).
  Kategorik×residual etkileşimi yok (korelasyon her grupta 0.301-0.308, sabit). 3+ sütunlu
  yeni bir aritmetik kısıt bulunamadı (`sleep+daily+work≤24` sanılan kısıt aslında %1.8
  ihlalli, jeneratörde enforce edilmiyor). Kalan ikili sürekli-sütun çiftleri tükenmiş.
  **Yan bulgu (actionable değil)**: `age` monotonik değil, `notifications_per_day` gibi
  lookup-table arketipi sergiliyor (23→%67, 24→%77, 25→%67 gibi salınım, ~%10 genlik,
  gürültünün ~40 katı) — ama exact-value TE zaten tam yakalıyor.
- **SONUÇ: tek-satır feature engineering damarı (ratio/diff/joint-encoding ailesi) bu
  veri setinde GERÇEKTEN TÜKENMİŞ.** 3 bağımsız araştırma (outlier, kombinasyon, pair-lattice)
  aynı yere çıktı. Bu yöne artık zaman harcamayın.

### 5. Feature-selection / budama — CV'de nötr, LB'de de nötr (KAPANDI, ama bir bug bulundu)
Fork ile 84 özellikli ABD referansının LightGBM gain'i en düşük 8/16/24/32 özelliği
çıkarma denendi. **Gerçek bir dead-code bug'ı bulundu**: `gender`/`stress_level`/
`academic_work_impact` "ham" sütunları `pd.to_numeric(errors='coerce').fillna(-999)` ile
işleniyor ama bunlar string olduğu için hepsi NaN'a dönüp SABİT -999 oluyor — zararsız
(gerçek sinyal zaten `te_gender` vb. üzerinden geliyor) ama gereksiz/ölü kod.
CV'de budama hiçbir yerde fark yaratmadı (84→52 özellikte bile AUC 0.96856→0.96855).
Bugünkü pair-lattice olayının (CV+/LB-) SİMETRİĞİNİ test etmek için (belki karmaşıklık
azaltmak CV'de görünmeyen bir LB faydası sağlar) `drop_lowest_16` (68 özellik) tam 3-model
blende taşınıp LB'ye atıldı: **CV 0.96879→0.96880 (+0.00001), LB 0.96985→0.96986 (+0.00001)
— TAM NÖTR, hem CV hem LB aynı fikirde.** Hipotez doğrulanmadı ama zarar da yok.
Script: `scripts/lgbm_xgb_cat_ABD_pruned68_2026-08-20.py`, çıkarılan 16 özellik listesi
script içinde `DROP_COLS`.

### 6. Extra-Trees/RandomForest/HistGB — zaten 2026-08-14'te test edilmiş, KAPALI
Kullanıcı "farklı model ailesi" önerince Extra-Trees'i tekrar denemeden önce
`model_family_battery_run.log` (2026-08-14) kontrol edildi: RF/ET/HistGB'nin hepsi
LightGBM ile eşit-ağırlıklı blend'de **daha kötü** sonuç veriyor (ET: solo 0.95684,
LGB'den 0.0115 geride, korelasyon en düşük 0.952 olmasına rağmen blend -0.00291).
Ders: düşük korelasyon (çeşitlilik) tek başına yetmiyor, solo kalite açığı çok büyükse
(GBDT-NN farkının, 0.003, 3.7 katı) eşit-ağırlıklı blend zayıf modeli aşağı çekiyor.

### 7. TabM denendi, donanım kısıtı yüzünden YARIDA BIRAKILDI
Kullanıcının paylaştığı 74-model notebook'unun credits bölümünde "en değerli bulgu" diye
geçen TabM (Yandex Research, batch-ensemble MLP mimarisi, resmi `tabm` pip paketi) denendi.
- `pip install tabm` başarılı, resmi API kullanıldı (`tabm.TabM.make(...)`).
- Veri hazırlığı: ABD 84-özellik seti reuse edildi (TabM'in kendi önerisi zaten "her sütunu
  TE'le" olduğu için), standardize edilip GBDT ile aynı k-fold split'e hizalandı.
  Script: `scripts/tabm_data_prep_2026-08-20.py`.
- Smoke test (1 fold, 3 epoch) başarılı: val AUC 0.96078→0.96283, yükseliş trendinde,
  ~34s/epoch (NN ile benzer hız).
- Tam 5-fold koşusu başlatıldı (`scripts/tabm_train_kfold_2026-08-20.py`, k=32, d_block=512,
  n_blocks=3) ama **fold 0, 28 dk sürdü (val AUC=0.96595, NN'imize yakın ama üstün değil),
  fold 1 60+ dakika takıldı kaldı.** Teşhis: GPU %100 kullanımda ama sadece 50W/80W güçte
  (P3 performans durumu) — muhtemelen laptop-sınıfı RTX 4090 (80W güç sınırı) + TabM'in
  k=32 "aynı anda 32 mini-model" mimarisinin kernel-launch overhead'i bu donanımda verimsiz
  çalışıyor. Donmadı (GPU aktifti) ama saatler sürebilirdi. **Kullanıcı kararıyla durduruldu**
  (PID kill). Fold 0'ın tek sonucu (0.96595) NN'imizden belirgin üstün değildi, devam etmeye
  değmeyebilir ama kesin kapanmadı — belki k=8 gibi çok daha küçük bir konfigürasyonla
  tekrar denenebilir (denenmedi).

### 8. NN Optuna hiperparametre taraması — İLK KEZ yapıldı, mütevazı kazanç
Gün sonunda ortaya çıkan stratejik bulgu (madde 0'a bak: NN'imiz aynı mimarinin dış
versiyonlarından 0.003-0.005 AUC geride) üzerine NN hiç görmediği bir hiperparametre
taraması aldı (GBDT gün-1'de 40 Optuna denemesi görmüştü, NN hiç görmemişti).
- 20 deneme, hız için tek 90/10 split (`nn_cache/prepped.npz`) üzerinde, 18 epoch üst
  sınır + patience=4, MedianPruner ile kötü gidenler erken kesildi (9 deneme pruned).
  Script: `scripts/nn_optuna_search_2026-08-20.py`, SQLite study `nn_cache/optuna_study.db`
  (kalıcı, kesintiye dayanıklı), ~1.5 saat sürdü.
- **En iyi val AUC = 0.96556** (90/10 referansı 0.96507'den +0.0005, gerçek ama küçük).
  **20 denemenin TAMAMI dar bir bantta kaldı (0.961-0.966)** — yani mimari/lr/dropout
  taraması bile bizi referans notebook'ların 0.96853'üne YAKLAŞTIRAMADI. Bu, açığın
  hiperparametrelerden değil BAŞKA bir yerden geldiğini gösteriyor (muhtemelen: (a) NN'e
  giden feature seti GBDT'nin 84'ünden çok daha dar - 17 sürekli+4 kategorik, 21 token;
  (b) missingness-augmentation eksikliği - GBDT'nin NN'e üstünlüğü eksik-veri arttıkça
  büyüyordu, madde 3'te bahsedilen henüz denenmemiş fikir; (c) bizim implementasyonumuzun
  bilmediğimiz başka bir detayı).
- En iyi parametreler (`nn_cache/optuna_best_params.json`): d_token=160 (eskisi 64),
  n_layers=3 (eskisi 2), n_freq=16 (eskisi 8), dropout=0.194, n_head_blocks=3,
  base_lr=0.00056 (eskisi 0.001), weight_decay=1.3e-5, batch_size=4096 (eskisi 2048),
  warmup_epochs=2.
- **Bu en iyi konfigürasyonla tam k-fold retrain BAŞLATILDI ama OTURUM BURADA YARIDA
  KESİLDİ** (`scripts/nn_train_kfold_tuned_2026-08-20.py`, 30 epoch üst sınır, patience=6,
  checkpoint'li). Yarın ilk iş bunun sonucuna bakmak.

### 9. Tuned NN k-fold retrain TAMAMLANDI (yeni oturum, aynı gün devamı) — NEGATİF, KAPANDI
Script'e resume mantığı eklendi (fold0 checkpoint'i varsa training'i atlayıp sadece test
tahmini için modeli yükler) — kalan 4 fold ~33.5 dk'da tamamlandı (GPU, fold başı ~5-7 dk).
- **5-fold TUNED NN OOF AUC = 0.96569** — eski (tune edilmemiş) NN k-fold'un **0.96576'sından
  hafifçe KÖTÜ** (-0.00007, gürültü). Optuna'nın bulduğu "en iyi" hiperparametreler (d_token
  160, 3 katman, n_freq=16 vb.) tam k-fold'da hiçbir gerçek iyileşme getirmedi.
- Blend potansiyeli de ölçüldü: GBDT+tuned-NN rank-blend en iyi OOF=**0.96897** (W_GBDT=0.81),
  eski GBDT+NN blend'in 0.96895'inden (W_GBDT=0.82) sadece **+0.00002** — kalibrasyon
  kuralının gürültü eşiğinin (Δ<0.00015) çok altında. **LB'ye ATILMADI** (feedback:
  local-only/gürültü-seviyesi kazancı LB'ye taşıma kuralına uyuldu).
- **SONUÇ: NN hiperparametre/mimari taraması (Optuna dahil) bu açığı KAPATAMAYACAĞINI
  kesin olarak doğruladı.** Madde 8'deki hipotez (a/b/c) arasında (a) dar feature seti ve
  (b) missingness-augmentation eksikliği artık en olası adaylar — mimari/LR/dropout/derinlik
  ekseni tükendi, tekrar denemeye gerek yok (KAPANDI).

### YARIN İÇİN BAŞLANGIÇ NOKTASI (öncelik sırasıyla)
1. **NN hiperparametre taraması artık KAPANDI (madde 9) — tekrar deneme.** Sıradaki kaldıraç:
   missingness-augmentation'ı dene (eğitim
   sırasında rastgele ek değer maskeleme, referans notebook'un `lookup` üyesinin muhtemelen
   yaptığı şey) — bu bir hiperparametre değil, eğitim MEKANİZMASI değişikliği, HENÜZ HİÇ
   denenmedi. GBDT'nin NN'e üstünlüğünün eksik-veri arttıkça büyümesiyle (madde 3, tam veri
   +0.0025, 4+ eksik +0.0068) doğrudan motive.
2. **NN'e GBDT'nin 84-özellik setinin bir kısmını (özellikle 8 türetilmiş oran/fark +
   frekans encoding) daha eksiksiz vermek** — şu an NN sadece 21 token görüyor, GBDT 84
   özellik. Daha önce "feature genişletme" denenmişti (val AUC +0.00193 vermişti,
   2026-08-19) ama tüm 84'ü değil, sadece 8 tanesini. Genişletmenin devamı düşünülebilir
   (dikkat: blend'e katkısı geçen sefer neredeyse sıfırdı çünkü GBDT'ye benzeşti — bu sefer
   amaç NN'in KENDİ kalitesini GBDT seviyesine çıkarmak, blend çeşitliliğini korumak için
   NN'in kendine özgü mekanizmasını [exact-value lookup+PLR+attention] koruyarak yapmalı).
3. **3-seed bagging** (GBDT için, açık madde, hiç denenmedi bu oturumda) — düşük risk,
   geçmişte küçük ama gerçek kazanç vermişti (K0→K1 geçişi).
4. **Kapanmış yönler (TEKRAR DENEME):** pair-lattice (tam veya yoğunluk-filtreli), missingness-
   regime blend ağırlığı, logit-space stacking, doğrusal-skor feature, kategorik-üçlü joint
   encoding, kategorik×residual etkileşimi, Extra-Trees/RF/HistGB eşit-ağırlıklı blend,
   klasik outlier/fiziksel-tutarsızlık temizliği, tüm ikili sürekli-sütun kombinasyonları,
   **ve artık NN hiperparametre/mimari taraması (Optuna dahil, madde 9'da kapandı).**
5. **Üretim durumu değişmedi**: en iyi submission hâlâ `sub/2026-08-20/
   blend_gbdt_nn_kfold_w82_2026-08-20.csv` (LB=0.96998). Budanmış-68 (LB=0.96986) ve
   pair-lattice (LB=0.96967) submission'ları test amaçlıydı, production'a alınmadı.
6. **Hatırlatma**: dürüst tavan (tomasa2, LB=0.97014) bizden sadece 0.00016 uzakta — LB
   gürültü tabanının içinde. Beklentileri düşük tut, yarışma sonu (31 Ağustos) final 2
   submission'ı CV'ye göre seç, LB rank'ine göre değil.

İlgili: [[feedback_kaggle_data_first_workflow]], [[feedback_kaggle_local_f1_no_push]]

## 2026-08-19 (5. Gün, gün-4'ten 5 gün sonra devam)

### GÜN SONU ÖZET / SIRADAKİ OTURUM İÇİN BAŞLANGIÇ NOKTASI
- **Gün başı LB: 0.96957 (K1 3-seed blend) → Gün sonu LB: 0.96985** (+0.00028, A+B+D birleşik, tek-seed)
- Kullanıcı "büyük açığı kovala" dedi (1.'nin LB'si 0.97500). Kaggle'da (claude-in-chrome ile)
  topluluk taraması yapıldı: yarışmanın küçük bir gerçek veri setinden (7500 satır,
  `jayjoshi37/smartphone-usage-and-addiction-prediction`) esinlenerek üretildiği, jeneratörün
  hard-rule'unun (broccoli beef, discussion #732428) çözüldüğü, ve 0.970+ LB'li birkaç public
  notebook (tomasa2 LB=0.97014, "S6E8: What Moved the Score, and What Didn't") bulundu.
  Bir fork bu kaynakları taradı (ilk deneme boş döndü — 0 tool_uses, resume edildi, ikinci
  denemede 22 tool_uses ile gerçek bulgular getirdi).
- **ÖNEMLİ DÜZELTME**: fork'un "sürekli sütunları TE'lemiyoruz" iddiası YANLIŞ çıktı —
  `lgbm_raw_te.py`'yi (gün-1'den beri production'da) kontrol edince `daily_screen_time_hours`
  gibi tüm sürekli sütunların zaten gün-1'den beri exact-value TE'lendiği görüldü. tomasa2'nin
  "TE everything" kaldıracının büyük kısmı bizde zaten vardı — bu yüzden fork'un tahmin ettiği
  +0.004 toplam potansiyelin çoğu gerçekleşmedi (gerçek transfer: +0.00028).
- **5 izole teknik K1 (48-özellik, LB=0.96957) referansına eklenip TEK TEK LB'de test edildi**
  (kullanıcı: "local f1'i boşver, direkt LB'de test edelim" - K1 tek-seed referans CV=0.96843):

  | Teknik | CV Δ | LB | LB Δ (K1'e göre) |
  |---|---|---|---|
  | A: frekans (count) encoding | +0.00025 | 0.96970 | +0.00013 ✓ |
  | B: imputation-augment (raw yanına 'imp_' kopya) | +0.00011 | 0.96959 | +0.00002 (gürültü) |
  | D: decimal lattice (frac/ilk-ondalık-basamak, 6 sütun) | +0.00012 | 0.96958 | +0.00001 (gürültü) |
  | E: jenerator hard-rule flag (broccoli beef kurali) | +0.00006 | 0.96951 | -0.00006 (gürültü/hafif negatif) |
  | C: CatBoost native categorical (manuel TE yerine) | -0.00057 | 0.96883 | **-0.00074 kesin negatif** |

  Kendi kalibrasyon kuralımız (CV Δ>0.0003 güvenilir, Δ<0.00015 gürültü) burada da dogrulandi:
  sadece A tek başına net pozitifti, B/D/E CV'de zaten esigin altindaydi ve LB bunu birebir
  yansitti (hepsi ±0.00006 bandi). C'nin buyuk CV negatifi (-0.00057) LB'ye ~1.3x oraninda
  (kotu yonde) tasindi - buyuk-CV-farki-guvenilir-tasinir kurali burada da gecerliydi.

- **A+B+D birlestirme denendi (E ve C haric tutuldu - net negatiflerdi) - BASARILI.**
  Kullanicinin sezgisi dogru cikti: B ve D tek basina gurultu seviyesindeydi ama farkli bilgi
  kanallari (frekans/imputation/ondalik) tasidiklari icin birlikte toplandiklarinda olculebilir
  hale geldiler. CV: K1 0.96843 -> A+B+D 0.96879 (+0.00036, guvenilir esigin uzerinde).
  **LB DOGRULANDI: 0.96985 (+0.00028 K1'e gore) - GUNUN EN IYI SKORU.**
  Script: `scripts/lgbm_xgb_cat_ABD_combined_2026-08-19.py`, submission:
  `sub/2026-08-19/lgbm_xgb_cat_ABD_combined_2026-08-19.csv`.
  **Yeni metodolojik ders**: gurultu-seviyesindeki (Δ<0.00015) birden fazla BAGIMSIZ-kanalli
  kucuk pozitif sinyal, TEK TEK gonderilmeye degmese de BIRLIKTE test edilmeye deger olabilir -
  eskiden "Δ<0.00015 ise atma" kurali artik "Δ<0.00015 ise TEK BASINA atma, ama benzer boyutta
  birkac BAGIMSIZ kucuk pozitif varsa BIRLESTIRIP dene" olarak inceltildi.

- **Kullanıcı yeni bir topluluk tekniği paylaştı: "Single Model Feature Engineering" (Hazmah'ın
  discussion'ı) - meta-feature stacking + polinom etkilesim.** Aslinda `StackingClassifier
  (passthrough=True)`'nin yeniden adlandirilmisi (broccoli beef bunu dogru tespit etti).
  **Bu TEKNIK BIZIM GUN-4 BULGUMUZLE BIREBIR ORTUSUYOR**: gun-4'te ayni tekniği kendi 6
  OOF'umuzla (genuinely diverse: eski 42-ozellik LGB/XGB/Cat + RF/ExtraTrees/HistGB) test
  edip mekanizmanin gercek oldugunu kanitlamistik (+0.0012 CV) ama kendi EN IYI 4 modelimizle
  (hepsi ayni K1 feature set'inde, >0.98 korele) denedigimizde SIFIR kazanc bulmustuk. Sonuc:
  teknik gercek ama SADECE gercek cesitlilikle (farkli insanlarin farkli FE/model kararlariyla
  urettigi DIS OOF'lar) calisiyor - bu, gun-4'ten beri "tek acik kalan buyuk kaldirac" dedigimiz
  seyin ta kendisi. Kullanici onayladi: **plana zaten dahildi, ayri tartismaya gerek yok,**
  sirasi gelince (dis OOF kaynaklari saglaninca) uygulanacak.

- **Yapay sinir ağı (Lookup-Transformer) kuruldu, gerçek çeşitlilik kaynağı olarak doğrulandı.**
  Kullanıcının isteğiyle (`nn_data_prep.py`, `nn_model.py`, `nn_submission.py`, hepsi Read/
  py_compile ile doğrulandı, gerçekten çalıştı) tamerlanomralinov'un Kaggle'da paylaştığı
  "lookup transformer" mimarisinden (solo LB=0.97041, bu yarışmaya özel) esinlenen bir NN
  kuruldu: 9 sürekli sütun için exact-değer lookup embedding + PLR (öğrenilmiş Fourier
  frekanslı periyodik-linear trend), 3 kategorik sütun için sadece lookup embedding, CLS+12
  token → TransformerEncoder (d=64, 2 katman, 8 head) → ResidualBlock'lu head. İlk basit
  embedding+MLP denemesi (val AUC=0.93965) çok zayıftı; Lookup-Transformer'a geçince
  **val AUC=0.96346'ya sıçradı (+0.0238)** - GBDT bandına (~0.968) çok daha yakın.
  - **NN solo LB=0.96518** (val AUC ile tutarlı transfer).
  - GBDT (A+B+D) ile korelasyon: Spearman=0.979, Pearson=0.882 - gerçek ama tam-ayrık değil
    (referans notebook'un kendi NN-GBDT karşılaştırması 0.968'di, bizimki biraz daha yüksek,
    muhtemelen v1'in türetilmiş oran feature'larını içermemesinden - bkz. açık madde).
  - **Eşit-ağırlıklı (%50/50) blend LB=0.96900 - GBDT-solo'nun (0.96985) ALTINDA, zarar verdi.**
    Ders: farklı kalitede iki modeli eşit ağırlıkla karıştırmak güçlü modeli aşağı çekiyor.
  - **Ağırlık optimizasyonu (LB harcamadan, GBDT OOF + NN val split aynı satırlarda):** GBDT
    (OOF)=0.96771, NN (val)=0.96346, tarama sonucu en iyi w_gbdt=0.80 (AUC=0.96791, plato
    0.75-0.90 bandında düz). Script: `lgbm_xgb_cat_ABD_combined_2026-08-19.py`'ye OOF-kaydetme
    eklendi (`nn_cache/gbdt_abd_oof.npy`), ağırlık taraması ayrı bir one-off komutla yapıldı.
  - **%85/15 ağırlıklı blend (optimuma çok yakın) GÖNDERİLDİ: LB=0.96990 - GÜNÜN/PROJENİN
    YENİ EN İYİ SKORU** (önceki 0.96985'ten +0.00005 - gürültü eşiğinin altında ama yönü
    tutarlı: hem held-out validasyonda hem LB'de aynı yönde, kör tahmin değil optimize edildi).
  - Script'ler: `scripts/nn_data_prep.py`, `scripts/nn_model.py`, `scripts/nn_submission.py`,
    `scripts/blend_gbdt_nn_weighted_2026-08-19.py`. Checkpoint: `nn_cache/best_model.pt`.

- **AÇIK MADDELER - SIRADAKİ OTURUM:**
  1. **Dış OOF/submission kaynakları topla** (örn. `tcspecialist/oof-and-sub-preds-for-6-runs`,
     Trish Cornelissen'in yorumunda geçen 6 run'lık dataset) - Kaggle indirme gerektiriyor,
     kullanıcının kendi hesabından indirip `data/` altına koyması en temiz yol. Dikkat: bazı
     "vault" tarzı dataset'ler (AnthonyTherrien'inki gibi) dürüst olmayan submission-recycling
     içeriyor olabilir (Naji'nin discussion'ı bunu ifşa etmişti) - kaynağın arkasında açık bir
     notebook/metodoloji olan dataset'leri tercih et.
  2. Dış OOF'lar gelince: `id` üzerinden birleştir (sıra değil), meta-feature'lar arası polinom
     etkileşim üret, mevcut raw+TE+ratio (A+B+D dahil) setine ekleyip tek güçlü model (Hazmah
     gibi tek XGB, ya da bizim 3-model blend) ile yeniden eğit.
  3. A+B+D kombosunu 3-seed bagging'e taşımak henüz yapılmadı (K0→K1 geçişindeki gibi küçük
     ek kazanç beklenebilir) - dış-OOF denemesinden önce ya da sonra düşük maliyetle yapılabilir.
  4. Okunmamış kalan yüksek-skorlu kaynaklar: Ripon C. Malo'nun "55-Model Stack" (LB 0.97068),
     tomasa2'nin "8. Stacking: logits kullan, combiner çıkarsın" + "8.2 Diversity has a floor"
     bölümleri (fork zaman bütçesi yetmedigi icin okuyamadi), Georgy Mamarin'in "why
     gaming_hours helps but adds nothing new".
  5. E ve C kapandı (E: jenerator-region flag, gurultu/hafif negatif; C: CatBoost native
     categorical, kesin negatif) - tekrar denemeye gerek yok.
  6. **NN (Lookup-Transformer) iyileştirme fırsatları (henüz denenmedi):**
     - Kanıtlanmış GBDT ratio/fark feature'larını (ratio_screen_sleep, diff_weekend_daily vb.)
       NN'e de ek token/girdi olarak eklemek - v1 bilinçli olarak sadece 9 ham sürekli + 3
       kategorik sütunla sınırlıydı, GBDT-NN korelasyonunun (0.979) referans notebook'tan
       (0.968) yüksek çıkmasının sebebi bu olabilir.
     - Şu an NN sadece TEK bir 90/10 train/val split ile eğitiliyor (hızlı iterasyon için) -
       gerçek k-fold OOF üretilmedi. Meta-stacking'i (Hazmah'ın polinom-etkileşim tekniği)
       düzgün yapmak için k-fold NN OOF'a geçmek gerekecek.
     - Mimari büyütme (d_token, katman sayısı artırma), daha uzun eğitim/farklı LR - val AUC
       0.96346'da early-stop oldu, GBDT bandına (0.968) daha yakınlaşma potansiyeli var.
     - Kendi K1 LGB OOF'unu NN'e ek feature olarak verme (vault notebook'un fikri, henüz
       hiçbir yerde denenmedi - Lookup-Transformer referansı bile bunu yapmamış).

### GÜN SONU ÖZET (2026-08-19) / YARIN İÇİN BAŞLANGIÇ NOKTASI

**LB progresyonu (bugün, tam):** 0.96957 (K1, gün başı) → 0.96970 (A: frekans-encoding) →
0.96985 (A+B+D: +imputation-augment+decimal-lattice) → 0.96990 (GBDT+NN-v1 %85/15 blend) →
**0.96996 (GBDT+NN-v2 %80/20 blend) — GÜN SONU / PROJE EN İYİ SKORU.**

**Bugünün 3 büyük başarısı:**
1. GBDT tarafında A+B+D kombosu (+0.00028, ayrıntı yukarıda).
2. Kendi Lookup-Transformer NN'imizi sıfırdan kurduk (tamerlanomralinov'un bu yarışmaya özel
   mimarisinden esinlenerek) — GERÇEK model çeşitliliği (Spearman rank-corr GBDT ile 0.979,
   Pearson 0.882) + makul standalone kalite (val AUC 0.93965→0.96507, 2 iterasyonda).
3. GBDT+NN ağırlıklı blend'i (LB harcamadan, held-out train verisiyle) optimize etme yöntemi
   kanıtlandı - kör tahmin yerine dürüst, sistematik bir ağırlık seçimi.

**NN hiperparametre/mimari deneyi sonucu (3 izole test, hepsi val AUC 0.96346 referansına göre):**
- Feature genişletme (türetilmiş oran/fark + dominant_activity): **+0.00193, TEK NET KAZANÇ**
- Kapasite artırma (d_token 128, 3 katman) + LR warmup: **-0.00085, KÖTÜLEŞTİ, terk edildi**
- Label smoothing (eps=0.02): izolede +0.00019 ama feature genişletmeyle BİRLİKTE **-0.00033
  (etkiler toplanmadı)**, terk edildi
- **Önemli/beklenmedik ders:** NN'i iyileştirmek için eklediğimiz feature'lar GBDT'nin ZATEN
  kullandığı feature'lardı — NN'in kendi kalitesini artırdı (0.96346→0.96507) ama aynı zamanda
  GBDT'ye daha çok benzemesine sebep olup çeşitlilik payını törpüledi. Sonuç: blend optimumu
  neredeyse hiç ilerlemedi (0.96791→0.96792) — LB'de de bunu doğrulayan çok küçük bir kazanç
  (0.96990→0.96996, +0.00006). **"NN'i iyileştir" ile "blend'e daha çok katkı sağla" burada
  birbirine ters düşen iki hedef oldu.** Gelecekte NN'i büyütmek istersek GBDT'nin KULLANMADIĞI
  bilgi kanallarına (örn. satır-sırası/batch yapısı, ham metin yok ama belki feature
  etkileşimleri farklı şekilde) odaklanmak daha yüksek potansiyelli olabilir.

**Script/dosya envanteri (NN tarafı, production - güncel):**
- `scripts/nn_common.py` - paylaşılan sınıflar (PLREmbedding, ResidualBlock,
  LookupTransformerNet, LookupDataset, train_model, predict_probabilities)
- `scripts/nn_data_prep.py` - PRODUCTION veri hazırlık (9 ham+8 türetilmiş sürekli + 4
  kategorik), `nn_cache/prepped.npz` üretir
- `scripts/nn_model.py` - PRODUCTION eğitim (feature genişletme ALINDI, kapasite artırma VE
  label smoothing REDDEDİLDİ), `nn_cache/best_model.pt` üretir
- `scripts/nn_submission.py` - test tahmini üretir
- `scripts/nn_experiment_features.py`, `nn_experiment_capacity.py`, `nn_experiment_labelsmooth.py`
  - tarihsel izole deney script'leri (artık production'da değiller, referans olarak duruyor)
- `scripts/blend_gbdt_nn_weighted_2026-08-19.py` - ağırlıklı blend üretici (W_GBDT parametresi)
- GBDT OOF kaydı: `scripts/lgbm_xgb_cat_ABD_combined_2026-08-19.py`'ye eklendi
  (`nn_cache/gbdt_abd_oof.npy`, `nn_cache/gbdt_abd_test_pred.npy`)

**AÇIK MADDELER - YARIN İÇİN BAŞLANGIÇ NOKTASI:**
1. NN için gerçek k-fold OOF üretmek (şu an tek 90/10 split) - daha güvenilir blend-ağırlık
   optimizasyonu ve düzgün meta-stacking (Hazmah'ın polinom-etkileşim tekniği) için gerekli.
2. GBDT'nin kullanmadığı bilgi kanallarına odaklanan NN iyileştirmesi (yukarıdaki ders) -
   feature'ları GBDT'den kopyalamak yerine NN'e özgü bir şey bulmak (örn. NN'in kendi embedding
   uzayında GBDT'nin göremediği bir etkileşim türü).
3. A+B+D GBDT kombosu henüz 3-seed bagging'e taşınmadı.
4. Okunmamış kalan kaynaklar: Ripon C. Malo'nun "55-Model Stack" (LB 0.97068), tomasa2'nin
   "8. Stacking" bölümleri, Georgy Mamarin'in "why gaming_hours helps" notebook'u.
5. Yarışma sonu (31 Ağustos) submission-seçimi CV'ye göre yapılacak, public LB rank'ine göre
   değil (bkz. yukarıdaki Kaggle discussion notu - Georgy Mamarin/Tilii/broccoli beef).

## 2026-08-14 (4. Gün)

### GÜN SONU ÖZET / YARIN İÇİN BAŞLANGIÇ NOKTASI
- **Gün başı LB: 0.96955 → Gün sonu LB: 0.96955** — K1 bundle iki adımda LB'de
  doğrulandı (izole 0.96940, sonra 3-model×3-seed blend'e taşınınca **0.96957**,
  şu anki en iyi skor, kayıtlarda 0.96955 idi düzeltme: gerçek en iyi **0.96957**).
- **Bugünün net sonucu: pek çok yol denendi, hemen hepsi kapandı.** Kapanan (tekrar
  denemeye gerek yok): RandomForest/ExtraTrees/HistGB model-ailesi çeşitliliği,
  class_weight/scale_pos_weight/is_unbalance, focal loss, num_leaves kapasite
  retuning (zaten optimal), kendi modellerimizle meta-feature+polinom stacking,
  pseudo-labeling (dün), train/test tam-satır leakage ihtimali.
- **Tek açık kalan büyük kaldıraç: dışarıdan gerçek model çeşitliliği.** Meta-stacking
  mekanizması ispatlandı (zayıf+farklı modellerle +0.0012 verdi) ama bizim kendi
  modellerimiz hepsi aynı feature set'te ve >0.98 korele olduğu için işe yaramadı.
  Gerçek kazanç için Kaggle'daki güçlü public notebook'ların OOF/submission
  dosyalarına ihtiyaç var — **yarın kullanıcıdan bunları indirip vermesi istenecek**
  (örn. Trish Cornelissen'in yorumundaki `tcspecialist/oof-and-sub-preds-for-6-runs`
  dataset'i gibi, veya başka güçlü public notebook'lar).
- Bugün ayrıca kapsamlı bir Kaggle discussion taraması yapıldı (LB gürültü analizi,
  missingness/distribution-shift teyidi, jeneratör derin analizi, TE-CV-leakage
  metodolojik notu, final-submission-seçimi public-LB'ye değil CV'ye dayanmalı
  uyarısı) — detaylar aşağıda madde 6'da.
- **Yarınki ilk adım**: kullanıcıdan dış OOF/submission dosyaları istenecek; gelirse
  gerçek meta-stacking denemesi yapılacak. Gelmezse sıradaki aday: jeneratörü daha
  derin tersine mühendislik ya da ensemble'ı ölçek büyüterek genişletmek (daha
  fazla seed/model, literatürdeki "onlarca model" örüntüsüne yaklaşmak).

### Durum (gün başı)
- Devralınan en iyi LB: 0.96955 (3-seed bagging blend)
- sub/ klasörü tarihe göre organize edildi: 2026-08-13'ün 19 submission dosyası
  `sub/2026-08-13/` altına taşındı (2026-08-11/12 ile aynı düzen).

### Bugün yapılanlar
1. **K1 bundle LB'ye atıldı** (izole tek-LGB, 48 özellik: K0/AB 42 − `diff_daily_sum`
   + `diff_daily_sum_clean` + Kategori I dominant-activity/rank): **LB 0.96940.**
   - Önceki en iyi izole tek-model LB: Kategori A, 0.96934. K1 bunu **+0.00006** geçti,
     yeni en iyi tek-model LB skoru.
   - CV'deki K0→K1 kazancı (+0.00009) LB'de de tutarlı pozitif çıktı — dünkü şüpheli
     "belki gürültüdür" değerlendirmesi yanlış çıktı, gerçek sinyal.
   - **Sıradaki adım**: K1'i `scripts/blend_multiseed.py`'nin 42-özellik FE bloğuna göm,
     3-model×3-seed bagging ile tam blend CV al, pozitifse LB'ye at.

2. **K1 tam 3-model×3-seed blend'e taşındı** (scripts/blend_multiseed_K1.py, K0/AB'nin
   42-özellik bloğu yerine K1'in 48-özellik bloğu, aynı 3 seed 42/43/44):
   - Tek-seed(42) blend: K0=0.96843 → K1=0.96849 (+0.00006)
   - 3-seed bagging blend: K0=0.96869 → K1=**0.96874 (+0.00005)**
   - Üç ayrı ölçümde (izole tek-LGB +0.00009, tek-seed blend +0.00006, 3-seed blend +0.00005)
     aynı yönde tutarlı küçük kazanç — kalibrasyon eşiğinin (0.00015) altında ama tekrarlanan
     bir sinyal olduğu için LB'ye atıldı.
   - Saved: sub/2026-08-14/lgbm_xgb_cat_multiseed_K1_2026-08-14.csv
   - **LB DOĞRULANDI: 0.96955 → 0.96957 (+0.00002). YENİ EN İYİ SKOR.**
     CV Δ+0.00005'in LB transfer oranı ~0.4x — düşük ama pozitif yönde tutarlı, dünkü
     multiseed bagging'in düşük transfer örüntüsüyle (~0.19x) benzer.
   - **GÜN İÇİ EN İYİ SKOR: 0.96957**

3. **Stratejik değerlendirme**: LB'de 1.'nin skoru 0.97500 — aramızdaki fark (0.00543)
   bugüne kadarki FE kazançlarından (0.0001-0.0005) çok büyük, küçük optimizasyonlarla
   kapanmaz. Önce ucuz bir hipotez elendi: **train/test arasında tam satır eşleşmesi
   (leakage) yok** — test satırlarının sadece %0.0007'si train'de birebir eşleşiyor
   (sürekli değişkenlerin 0.01 hassasiyeti yüzünden pratikte imkansız).

4. **Pseudo-labeling denendi** (scripts/pseudo_label_k1.py, K1 bundle tek-LGB, seed=42,
   5-fold; baseline OOF=0.96831 referans, aynı fold'larda pseudo-test satırları sadece
   eğitime eklendi, validation değişmedi — güvenli karşılaştırma):
   - Eşik 0.98/0.02 (en muhafazakâr, test'in %52.6'sı): 0.96820 (Δ=-0.00010)
   - Eşik 0.95/0.05: 0.96798 (Δ=-0.00033)
   - Eşik 0.90/0.10 (en gevşek): 0.96780 (Δ=-0.00051)
   **SONUÇ: NET NEGATİF, eşik gevşedikçe monotonik kötüleşiyor — gürültü değil, gerçek
   bozulma.** Muhtemel sebep: model kendi en emin olduğu tahminlerini pseudo-label
   olarak katıyor, yeni bilgi eklemek yerine mevcut önyargıyı pekiştiriyor (klasik
   self-training tuzağı); pozitif/negatif oranı da çok dengesiz seçiliyor. LB'ye
   atılmadı (CV net negatif, kalibrasyon kuralına göre gerek yok). Bu yöntem bu
   problemde işe yaramıyor, tekrar denemeye gerek yok.
   **Sıradaki büyük-kaldıraç adayları (denenmemiş, öncelik sırasıyla)**: farklı model
   ailesi eklemek (MLP/embedding, ağaç modellerinin >0.99 OOF korelasyonunu kırmak
   için gerçek çeşitlilik), jeneratörü tam tersine mühendislik, Kaggle discussion/
   notebook tarama (1.'nin yöntemine dair ipucu olabilir).

5. **Model-family çeşitliliği: RandomForest / ExtraTrees / HistGradientBoosting**
   (scripts/model_family_battery_k1.py, K1 48-özellik, seed=42, 5-fold, LGB'yle
   korelasyon + 2-model rank-blend testi):
   - RandomForest: standalone 0.96207 (LGB'den -0.00624), corr=0.975, blend Δ=**-0.00158**
   - ExtraTrees: standalone 0.95684 (-0.01146), corr=0.952, blend Δ=**-0.00291**
   - HistGB (sklearn): standalone 0.96732 (-0.00098), corr=0.996 (LGB kadar korele), blend Δ=**-0.00021**
   **SONUÇ: ÜÇÜ DE NET NEGATİF.** Düşük korelasyon (RF/ET) yeterli standalone kaliteyle
   gelmiyor; yeterli kaliteyle gelen (HistGB) LGB kadar korele, ek çeşitlilik katmıyor.
   "Farklı model ailesi ekle" fikri bu ayarlarla ölü — bu veri setinde boosting'in
   (lookup-table TE + grid-yapı sömürüsü) sağladığı avantajı bagging/histogram-only
   yöntemler kapatamıyor. Tekrar denemeye gerek yok (tuning ile RF'nin ~0.006'lık
   açığı kapanması beklenmiyor).

6. **Kaggle discussion turu** (kullanıcı molada, "acelemiz yok" — kapsamlı tarama yapıldı,
   claude-in-chrome ile JS-render sorunu aşıldı). Öne çıkan bulgular:
   - **broccoli beef'in TE-CV-leakage uyarısı** (chloeprice'ın "stringified TE" thread'inde):
     TE'yi outer model-CV'den BAĞIMSIZ bir 10-fold ile hesaplayıp sabit feature olarak
     kullanmak, outer-fold validation etiketlerinin training-fold TE değerlerine sızmasına
     yol açabilir (aynı mimari BİZİM TÜM script'lerimizde de var — te_skf 10-fold, outer
     skf 5-fold, ikisi bağımsız). Ama bizim submission_log'daki TÜM CV/LB çiftleri
     **CV'nin LB'den ~0.001 DÜŞÜK** çıktığını gösteriyor (leakage olsaydı CV LB'den
     yüksek çıkardı) — bu da bizim ölçümlerimizde bu sızıntının pratikte önemli bir
     bozulma yaratmadığının dolaylı kanıtı. Yine de not düşülmeye değer bir metodolojik
     körnokta.
   - **WOWTIMWOW'un "model capacity was worth 18x my FE" bulgusu**: num_leaves=15→31
     yapınca ratio/diff özelliklerinin CV katkısı (+0.00042) sıfırlanıyor (-0.00046) —
     ağaç yeterli kapasitede zaten kendi kuruyor. Bizim num_leaves=43 (2026-08-12'de
     28-özellik raw+TE seti için Optuna'yla tunelendi) o zamandan beri özellik sayısı
     48'e çıktı ama kapasite hiç yeniden ayarlanmadı. **scripts/capacity_sweep_k1.py**
     başlatıldı: num_leaves∈{31,43,63,90,127} için hem base(28-özellik raw+TE) hem
     K1(48-özellik) CV'si ölçülüyor — K1'in base'e üstünlüğü (gap) kapasite arttıkça
     küçülüyor mu, yoksa kapasitenin kendisi ek kazanç mı veriyor, ikisi birden
     cevaplanacak.
     **SONUÇ (tamamlandı):**
     ```
     num_leaves=  31   base=0.96737   K1=0.96827   gap=+0.00090
     num_leaves=  43   base=0.96738   K1=0.96831   gap=+0.00093  (mevcut production)
     num_leaves=  63   base=0.96734   K1=0.96827   gap=+0.00093
     num_leaves=  90   base=0.96729   K1=0.96825   gap=+0.00096
     num_leaves= 127   base=0.96729   K1=0.96825   gap=+0.00096
     ```
     İki net bulgu: (1) **num_leaves=43 zaten neredeyse optimal** — hem base hem K1
     31'den 127'ye kadar düz/hafif düşüyor, saklı bir kapasite kazancı YOK, kapasite
     yeniden ayarlamaya gerek yok. (2) **K1'in base'e üstünlüğü (gap) kapasite
     arttıkça KÜÇÜLMÜYOR, hatta hafif büyüyor** (+0.00090→+0.00096) — WOWTIMWOW'un
     bulgusunun (kapasite artınca FE kazancı sıfırlanıyor) BİZİM PAKETİMİZE
     UYGULANMADIĞININ kesin kanıtı. Ratio/K1 özelliklerimizin kazancı kapasite
     artefaktı değil, gerçek katkı. Bu konu kapandı — WOWTIMWOW'un uyarısı bizim
     için geçerli değilmiş, ekstra bir işlem gerekmiyor.

7. **Focal loss** (scripts/focal_loss_k1.py, custom objective + sonlu-fark gradyan/
   hessian, K1 48-özellik, seed=42, 5-fold, alpha=0.5 yani sınıf-dengeleme YOK,
   sadece γ ile "zor örnek" vurgusu izole test edildi):
   - baseline (logloss): 0.96831
   - γ=1.0: 0.96831 (Δ=0.00000), γ=2.0: 0.96829 (Δ=-0.00002), γ=3.0: 0.96829 (Δ=-0.00002)
   **SONUÇ: KATKISIZ/HAFİF NEGATİF** — bugünkü class_weight/scale_pos_weight/
   is_unbalance testleriyle (hepsi negatif) aynı akıbet, aynı mekanizma (AUC
   sıralama-bazlı, örnek-yeniden-ağırlıklandırmadan fayda görmüyor). Custom
   objective ~22-24dk/varyant sürdü (native C++ objective'e göre ~6-7x yavaş,
   sonlu-fark 3 kat ekstra fonksiyon değerlendirmesi gerektiriyor) — yavaşlık
   bug değildi, gerçek maliyetti. LB'ye atılmadı. **Hata-fonksiyonu-değiştirme
   ailesi (class weight + focal loss) bu problemde kapandı, tekrar denemeye gerek yok.**

8. **Meta-feature + polinom-etkileşim stacking** (kullanıcının paylaştığı Kaggle
   discussion'ı, Hazmah'ın "single model FE" tekniği — broccoli beef'in dediği gibi
   aslında StackingClassifier(passthrough=True)). İki aşamalı test edildi:
   - **Mekanizma testi** (scripts/meta_stack_test.py, elde hazır 6 OOF: eski-pipeline
     2026-08-12 LGB/XGB/Cat 42-özellik + bugünkü K1 RF/ExtraTrees/HistGB):
     basit 6-model rank-blend=0.96699 (en iyi tekil modelden bile KÖTÜ, zayıf
     RF/ET blend'i aşağı çekiyor) → polinom-etkileşimli nonlinear LGB meta-model
     **0.96819 (+0.0012)** — mekanizma gerçek, zayıf üyeleri otomatik az ağırlıklandırıyor.
   - **Kendi en iyi modellerimizle test** (scripts/blend_4model_stack_k1.py, K1
     48-özellik, LGB+XGB+CatBoost+HistGB, tek seed=42, 5-fold):
     - 3-model (LGB+XGB+Cat) referans: 0.96849
     - 4-model (+HistGB) basit blend: 0.96843 (Δ=-0.00006)
     - 4-model polinom-etkileşimli meta-stack: **0.96843 (aynı, ek kazanç YOK)**
     Korelasyonlar hepsi 0.988-0.997 arası (HistGB dahi LGB'yle 0.996) — modeller
     zaten çok benzer kalitede ve çok korele, meta-model'in "zayıf üyeyi ele" diye
     öğreneceği bir yapı yok, düz blend'e yakınsıyor.
   **SONUÇ: Teknik gerçek (mekanizma testinde kanıtlandı) ama BİZİM MODEL HAVUZUMUZ
   için değersiz** — çünkü tüm modellerimiz (LGB/XGB/Cat/HistGB) aynı 48-özellik
   K1 setinde eğitiliyor, hepsi benzer kalitede ve >0.98 korele. Kazanç ancak
   GERÇEKTEN farklı pipeline'lardan (farklı FE, farklı pratisyen kararları) gelen
   OOF'larla mümkün — ki bunlar elimizde yok, dışarıdan (Kaggle public notebook'ları)
   kullanıcının indirip vermesi gerekiyor. LB'ye atılmadı (CV zaten mevcut en iyinin,
   0.96874 çok-seedli, altında).
   - **Luka Duvanov'un LB-gürültü analizi**: paired bootstrap sigma ~0.00009-0.00011
     (Hanley-McNeil'in tahmininden 6-7x daha küçük çünkü aynı satırlarda ölçülüyor,
     ortak şok iptal oluyor). Pratik kural: "40 sıra oynatıp skor <0.0001 değişiyorsa
     hiçbir şey ölçmedin." **Bizim kalibrasyon kuralımızı (Δ<0.00015 gürültü) bağımsız,
     rigor bir istatistikle doğruluyor.**
   - **Luka Duvanov + L.E.Electron: jeneratör kısıt-onarımı** — orijinal 7500 satırın
     %26'sı `social+gaming > daily` gibi imkansız satırlar içeriyor, jeneratör 691K'da
     bunu %0'a indirmiş (constraint mükemmel uygulanmış). Bizim gün-3 bulgumuzla
     birebir tutarlı, yeni bir şey değil. `ratio_weekend_daily`'yi (zaten elimizde olan
     özellik) LGB'ye tek başına vermek −0.00007 (seed gürültüsü ±0.00004 içinde) —
     ağaçlar zaten yakalıyor, null sonuç.
   - **busyaprime'ın derin analiz notebook'u** ("How the S6E8 generator reshaped the
     rules"): jeneratörün 2 sert kuralı (`daily>8` veya `social>4`) yumuşak olasılığa
     çevirdiğini, "otherwise" bölgesinin (satırların ~1/3'ü) orijinalde tam gürültü
     (AUC 0.501) iken sentetikte öğrenilebilir hale geldiğini (AUC 0.896, tüm
     özelliklerle) gösteriyor. Sonuç: "kuralları hardcode etme, tam modelle jeneratörün
     yumuşak alanına uy" — bizim zaten yaptığımız şey, YENİ bir kaldıraç değil, mevcut
     yaklaşımımızın doğruluğunu teyit ediyor.
   - **Dariush Afshar: "27 takım 0.97086 gösteriyor, hiçbiri eşit değil"** — KRİTİK
     metodolojik bulgu. Kaggle görünen skoru 5 ondalığa yuvarlıyor ama TAM hassasiyetle
     sıralıyor. Daha önemlisi: **public LB skorun, kendi submission geçmişinin
     en-iyisi (best-of-N)** — yani neredeyse-aynı (yüksek korelasyonlu) bir modeli
     tekrar atmak, HİÇBİR gerçek iyileştirme olmasa bile ~%48.6 ihtimalle görünen
     rankını yükseltiyor (çünkü Kaggle en iyi skoru tutuyor, düşüşü göstermiyor).
     **Sonuç/ders: final 2 submission'ı seçerken public LB'ye değil CV'ye güven** —
     bizim yarışma bitiminde (31 Ağustos) hangi 2 submission'ı finalize edeceğimize
     karar verirken bunu uygulamalıyız. Ayrıca resolvability formülü verdi:
     sd(gap) = sd(move)·√(2(1-ρ)) — yüksek korelasyonlu (near-twin) submission
     çiftlerinde küçük deltalar bile istatistiksel olarak daha güvenilir oluyor,
     bu da bizim K0→K1 gibi near-twin karşılaştırmalarımızda küçük deltalara
     (Δ~0.00005) neden makul güvenebildiğimizi açıklıyor.
   - **Dariush Afshar + Georgy Mamarin: "missingness is the whole shift"** —
     adversarial validation (train vs test) ile: ham veride AUC=0.564 (hafif ayrılabilir)
     ama NaN'ler impute edilince AUC=0.503'e (train/test ayırt edilemez) düşüyor.
     **SONUÇ: train/test arasında GERÇEK değer-dağılımı kayması YOK, tek kayma
     missingness oranlarında** (gün-3'te bizim de bulduğumuz şey, burada adversarial
     validation ile kesin kanıtlandı). İYİ HABER: private LB'nin public LB gibi
     davranması bekleniyor, elaborate drift-düzeltme şemalarına gerek yok. Ayrıca
     final (target-task) ablasyonla missingness bayraklarının TÜMÜNÜN (9 numeric + 3
     kategorik) katkısız/hafif-negatif olduğu bir kez daha doğrulandı (no_flags=0.962806,
     +9numeric=0.962804, +3categorical=0.962761) — bu konuyu artık 5 bağımsız kaynak
     (biz, broccoli beef ki-kare, Luka AUC=0.502, Dariush adversarial+target) kapattı.

## 2026-08-13 (3. Gün)

### GÜN SONU ÖZET / YARIN İÇİN BAŞLANGIÇ NOKTASI
- **Gün başı LB: 0.96950 → Gün sonu LB: 0.96955** (+0.00005, tek kaynak: çok-seedli bagging —
  küçük, kalibrasyon kuralına göre gürültü sınırında, bkz. aşağıdaki düzeltme notu)
- Bugün CV'de test edilip LB'ye HENÜZ atılmamış, yarın devam edilecek paket:
  **K1 bundle (48 özellik): AB 42 özellik − `diff_daily_sum` + `diff_daily_sum_clean`
  + Kategori I (max_activity3, range_activity3, gap_social_to_max, gap_gaming_to_max,
  gap_work_to_max, dominant_activity→TE)**. İzole tek-LGB CV: 0.96822→0.96831 (+0.00009).
- **Yarınki ilk adım**: bu K1 paketini `blend_multiseed.py`'nin 42-özellik feature-engineering
  bloğuna göm (scripts/lgbm_fe_bundle_check.py'deki blok birebir kopyalanabilir), 3-model+3-seed
  bagging ile tam blend çalıştır, CV doğrula, LB'ye at. Beklenti düşük tutulmalı: bugünün
  düzeltilmiş transfer oranı (bagging: CV+0.00026 → LB+0.00005, ~0.19x) diğer günlerin
  1.0-1.1x'inden düşük çıktı — +0.00009 CV için LB'de anlamlı bir hareket olmayabilir,
  gürültü ile ayırt edilemeyebilir. Yine de ucuz bir test, denemeye değer.
- Script hazır: scripts/lgbm_fe_bundle_check.py (izole test), scripts/blend_multiseed.py
  (production bagging pipeline — feature bloğu K1 ile güncellenmeli).
- Bugün test edilip KATKISIZ/NEGATİF bulunanlar (tekrar denemeye gerek yok): Kategori E (yeni
  ratio/diff), Kategori F (joint TE notif×opens), SMOOTH=5/10, feature-subset çeşitliliği,
  stress_level sapması (Kategori J), tam-regresyon imputasyonu (H3).
- Denenmemiş, gündemde kalan: decimal-place/ondalık-basamak özellikleri (zayıf kanıtlı, düşük
  öncelik), seed sayısını 3'ten yukarı çıkarmak (kullanıcı ertelemeyi tercih etti).

### Durum (gün başı)
- Devralınan en iyi LB: 0.96950 (v5 AB blend), CV baseline (AB, 42 özellik, tek LGB): 0.96824
- Yarışma bitiş: 31 Ağustos 2026 (18 gün kaldı)
- Bugünkü plan: dünkü "yarın için başlangıç noktası" maddeleri — Kategori E (yeni ratio/diff),
  Kategori F (2-yönlü joint TE), çok-seedli bagging

### Bugün yapılanlar
1. **Kategori E: denenmemiş ratio/diff kombinasyonları** (scripts/lgbm_fe_categories_v2.py,
   izole tek-LGB CV, referans=AB baseline 0.96824):
   `ratio_gaming_sleep, ratio_social_gaming, ratio_notif_opens, ratio_weekend_gaming,
   ratio_weekend_work, diff_weekend_sum, diff_social_gaming` (7 özellik).
   **Sonuç: 0.96824 → 0.96824, TAM SIFIR KATKI.** Feature importance'ta hepsi orta seviyede
   kullanılmış (1169-1543 split) ama model performansına hiç yansımamış — muhtemelen zaten
   var olan A/B oranlarıyla yüksek korelasyonlu / bilgi tekrarı. **Sonuç: A/B ailesindeki
   "kolay" ratio/fark sinyali tükenmiş, bu yönde daha fazla kombinasyon denemenin beklenen
   getirisi düşük.** LB'ye atılmadı (CV'de sıfır kazanç, kalibrasyon kuralına göre gerek yok).

2. **Kategori F: 2-yönlü joint TE** (scripts/lgbm_fe_jointte.py, izole tek-LGB CV):
   `te_joint(notifications_per_day, app_opens_per_day)` — iki lookup-table arketipli sütunun
   ortak anahtarına 10-fold OOF target encoding (smooth=10, ort. ~15 satır/hücre, 37238/38346
   olası kombinasyon dolu). **Sonuç: 0.96824 → 0.96823, Δ=-0.00001, KATKI YOK** (negatif/gürültü
   seviyesinde). Feature importance'ta çok kullanılmış (3679 split, tek başına en yüksek importance)
   ama bu sadece modelin var olan tekil TE(notif) + TE(opens) sütunlarıyla zaten yakalanabilen
   bilgiyi yeniden keşfetmesi — gerçek ek sinyal yok. LB'ye atılmadı.

   **Ara sonuç: bugünün ilk iki kaldıracı (E, F) ikisi de temiz negatif.** Kalan tek kaldıraç:
   çok-seedli bagging.

3. **Çok-seedli bagging** (scripts/blend_multiseed.py): AB pipeline'ı (42 özellik) seed 42/43/44
   ile 3 kez baştan çalıştırıldı (her seed kendi TE fold'ları + model fold'ları + model
   random_state'i ile) — LGB+XGB+CatBoost, toplam 9 (model×seed) OOF/tahmin.
   - Tek-seed (42) referansı doğrulandı: 0.96843 (dünkü v5 AB skoruyla birebir eşleşti).
   - Model-bazlı 3-seed ortalaması → sonra 3-model rank-blend: **0.96869**
   - Tüm 9 (model×seed) doğrudan rank-blend: **0.96869** (iki yöntem birebir aynı sonucu verdi)
   - **Δ=+0.00026, KATKI VAR** — dünkü kalibrasyon kuralına göre (Δ>0.00015 güvenilir bölge,
     Δ>0.0003 çok güvenilir) sınırda ama pozitif yönde net, LB'ye atılmaya değer.
   - Saved: sub/lgbm_xgb_cat_multiseed_2026-08-13.csv — **LB'ye atılacak, henüz atılmadı**
   - Maliyet: 1647s (~27dk) — bugünün en pahalı denemesi ama en net kazancı.

### GÜN İÇİ ÖZET (2026-08-13, devam ediyor)
- Kategori E (yeni ratio/diff): KATKI YOK
- Kategori F (joint TE notif×opens): KATKI YOK (hafif negatif, gürültü)
- Çok-seedli bagging (3 seed): CV 0.96843→0.96869 (+0.00026)
  - **DÜZELTME (kullanıcı, sonradan): LB skoru ilk yanlış kaydedildi (0.96995 yazılmıştı,
    gerçek skor 0.96955).** Doğru okuma: **0.96950→0.96955, Δ=+0.00005.** Bu, dünkü kalibrasyon
    kuralına göre (Δ<0.00015 gürültü) aslında **gürültü sınırında** — CV'deki +0.00026'nın
    LB'ye transfer oranı ~0.19x (düşük, önceki denemelerin 1.0-1.1x'inin altında). Yani
    çok-seedli bagging'in gerçek LB katkısı bugün zannedildiği kadar net değil; küçük ama
    belirsiz bir kazanç olarak görülmeli, "kesin doğrulandı" diye işaretlenmemeli.
  - **GÜN SONU EN İYİ SKOR: 0.96955** (0.96995 değil — bu değer önceki notlarda hatalıydı,
    düzeltildi).
  - Dosya: sub/lgbm_xgb_cat_multiseed_2026-08-13.csv, submission_log.xlsx'e düzeltilmiş
    değerle işlendi.
- LB progresyon (güncel, düzeltilmiş): ...→0.96862→0.96900→0.96950→**0.96955**
- Sonraki olası kaldıraç: seed sayısını 3'ten 5-10'a çıkarmak (bagging işe yaradığına göre
  daha fazla seed muhtemelen ek kazanç verir, azalan getiriyle)

4. **SMOOTH tekrar sorgulandı** (kullanıcı sorusu: SMOOTH=5 ve SMOOTH=10'u da LB'ye atmalı mıyız?):
   scripts/te_smoothing_sweep_v2.py ile GÜNCEL 42-özellik AB pipeline üzerinde (tek LGB, 5-fold CV)
   SMOOTH∈{3,5,10} tekrar test edildi (dünkü sweep eski 28-özellik setindeydi, güncel değildi):
   - SMOOTH=3: 0.96824 (referans)
   - SMOOTH=5: 0.96819 (Δ=-0.00004)
   - SMOOTH=10: 0.96820 (Δ=-0.00004)
   **SONUÇ: SMOOTH=3 hâlâ en iyisi (veya en azından eşiti), 5/10 hafif kötü ama gürültü
   seviyesinde. Dünkü bulgu bugün de doğrulandı — LB'ye atılmadı (kalibrasyon kuralına göre
   Δ<0.00015 olan bir değişikliği submission'a değer görmüyoruz).** SMOOTH=3 kod tabanında kalıyor.

5. **Feature-subset çeşitliliği** (scripts/blend_featuresubset_diversity.py): dünkü nottaki
   "gerçek çeşitlilik için modellere farklı feature subset vermek gerekir" fikri test edildi.
   RAW+TE (28 özellik) çekirdek her modelde sabit tutuldu, ratio aileleri modeller arası
   dağıtıldı: LGB=tam 42 (RAW+TE+R1+R2), XGB=RAW+TE+R1 (35, R2 yok), CatBoost=RAW+TE+R2
   (35, R1 yok). Aynı 5-fold split, seed=42.
   - Tekil modeller: LGB 0.96820, XGB 0.96698 (R2 kaybından düştü), CatBoost 0.96767 (R1 kaybından düştü)
   - **Blend: 0.96836 — referans (aynı-feature-set blend, 0.96843) göre Δ=-0.00007, KATKI YOK.**
   - **OOF korelasyonları: lgb-xgb 0.99096, lgb-cat 0.99534, xgb-cat 0.98828** — aynı-feature-set
     durumuna göre neredeyse hiç düşmemiş (>0.99 civarı zaten "yüksek korelasyon" referansıydı).
   - **SONUÇ: gerçek çeşitlilik sağlanamadı.** Ratio özellikleri raw sütunların türetilmiş
     (deterministik) fonksiyonları olduğu için, bir modelden bir ratio ailesini çıkarmak bilgiyi
     yok etmiyor — ağaç modeli aynı bilgiyi raw sütunlar üzerinden benzer split'lerle yeniden
     kuruyor. Sonuç: OOF korelasyonu neredeyse değişmiyor, ama tekil model kalitesi (özellikle
     doğrudan feature'ı kaybeden model) hafif düşüyor → net kazanç yok/hafif negatif.
   - **Ders:** bu problemde gradyan-boosting ailesi (LGB/XGB/Cat) arasında feature-subset ile
     çeşitlilik yaratmak işe yaramıyor çünkü üçü de aynı bilgi setinden benzer fonksiyon sınıfını
     öğreniyor. Bugünkü tek gerçek çeşitlilik kaynağı seed (random_state) oldu — bu da stacking'in
     neden dün başarısız olduğunu (OOF'lar çok korele) daha da netleştiriyor: sorun sadece "aynı
     feature set" değil, üç modelin de birbirine çok benzer fonksiyonlar öğrenmesiydi. Gerçek
     çeşitlilik için muhtemelen farklı bir MODEL AİLESİ gerekir (örn. lineer/MLP), ağaç
     varyantları arası değil. LB'ye atılmadı (CV negatif).

6. **Kullanıcı önerisi: Kaggle tartışma panosu bulgusu.** Community'de (Georgy Mamarin / Dariush
   Afshar) `other_screen = daily - (social+gaming+work)` residual'ının SADECE 4 sütun tam
   doluyken standalone AUC≈0.765 verdiği paylaşılmış; bizim kısmi-eksik satırlarda 0 ile
   dolduran `diff_daily_sum` özelliğimizin bunu SEYRELTTİĞİ ortaya çıktı. Doğrulama:
   - Bizim mevcut `diff_daily_sum` (kısmi eksikte 0-doldurma): standalone AUC=**0.7130** (n=591937)
   - Temiz residual (sadece 4 sütun da dolu, n=421427, %61.0): standalone AUC=**0.7649**
     (community'nin 0.765/0.7649 sayılarıyla birebir eşleşti)
   - **Kısıt jeneratörde tutuyor**: 421427 tam-satırda 0 ihlal (`daily>=social+gaming+work` her zaman sağlanıyor)
   - Missingness dağılımı: 0 eksik=%61.0, 1 eksik=%23.0 (159020), 2 eksik=%12.4, 3 eksik=%3.4, 4 eksik=%0.3

   **Test G (izole CV, scripts/lgbm_fe_cleanresid.py, referans H1=0.96824):**
   - G_replace (diff_daily_sum yerine temiz versiyon): 0.96828 (Δ=+0.00005)
   - G_add (ikisini birlikte tut): 0.96827 (Δ=+0.00003)
   **Sonuç: küçük ama TUTARLI pozitif (bugünün diğer negatif denemelerinden farklı), eşik
   altı (Δ<0.00015). Standalone sinyal güçlü olsa da (0.713→0.765), zaten var olan raw+TE
   sütunları bu bilginin çoğunu farklı biçimde taşıdığı için modele marjinal katkısı küçük.**
   LB'ye tek başına atmaya değmez ama diğer iyileştirmelerle birlikte pakete eklenebilir.

7. **Jeneratör-farkındalıklı imputasyon araştırması** (kullanıcı isteği: molada kapsamlı analiz).
   Hipotez: "generator-mimicking imputation" ile eksik değerleri jeneratörün bilinen yapısına
   (aritmetik kısıt, monotonik/lookup-table arketipler) uygun şekilde doldurmak, gün-1'in
   başarısız genel-amaçlı imputasyon denemesinden (CV 0.96134, BOZDU) farklı sonuç verebilir mi?

   **Zemin bulguları:**
   - `eda_imputability_r2.py` tekrar çalıştırıldı: daily R²=0.808 (ÇOK İYİ), social R²=0.544,
     gaming R²=0.413, work R²=0.439 (hepsi "kısmen impute edilebilir"); sleep/notif/opens/age
     R²≈0.00-0.02 (impute EDİLEMEZ, lookup-table/bağımsız arketip doğrulandı — bunlar zaten
     gün-1'de imputation'a sokulmamıştı, doğru karardı).
   - **Önemli düzeltme**: ilk tasarladığım "kısıt denklemini sabit medyan residual ile geri çöz"
     yöntemi matematiksel olarak DEJENERE — sabit bir residual varsayımıyla geri çözünce
     kazanılan residual otomatik olarak o sabite eşitleniyor, hiç row-bazlı bilgi katmıyor.
     Düzeltilmiş yöntem: eksik bileşeni (sadece tam-1-eksik satırlarda, n=159020) LGBMRegressor
     ile TÜM diğer sütunlardan (constraint sütunları dahil) tahmin et — bu tam olarak
     eda_imputability_r2.py'nin ölçtüğü R²'leri kullanır, per-row varyans korunur.
   - Gün-1 denemesinden 3 kritik fark: (1) RAW sütunlar değişmiyor, sadece yeni türetilmiş
     residual özelliği için kullanılıyor — LightGBM'in native NaN routing'i korunuyor,
     (2) sadece tam-1-eksik satırlar hedefleniyor (2+ eksik → NaN kalıyor, zincirleme hata yok),
     (3) aynı tuned hiperparametreler ve aynı CV protokolü (gün-1 farklı/tune-edilmemiş
     hiperparametrelerle karışıktı, temiz bir karşılaştırma değildi).
   - **Test tamamlandı (scripts/lgbm_fe_constraintimpute.py).** Kapsam genişletme: %61.0→%71.4
     (159020 tam-1-eksik satırdan 72447'si, diğer tahmin sütunlarının (age/sleep/notif/vb.)
     kendi eksikleri yüzünden geri kalanı regresyona giremedi). Genişletilen kısımda standalone
     AUC=**0.8330** — temiz residual'ın kendisinden (0.7649) bile yüksek (muhtemelen diğer
     tahmin sütunlarının kendi zayıf sinyalini de bulaştırıyor, saf residual değil artık).
     **Ama tam modelde:**
     - H1 (mevcut, seyreltilmiş): 0.96822 (referans)
     - H2 (temiz-only, ekstra regresyon yok): **0.96828 (Δ=+0.00006) — EN İYİSİ**
     - H3 (regresyon-genişletilmiş): 0.96825 (Δ=+0.00003) — H2'den bile kötü
     - H5 (genişletilmiş+mevcut birlikte): 0.96827 (Δ=+0.00005)
     **SONUÇ: standalone AUC'daki büyük kazanç (0.765→0.833) tam modelde YOK OLUYOR — regresyon
     imputasyonunun eklediği bilgi, age/sleep/notif/weekend/kategorik sütunların zaten
     raw+TE olarak modelde bulunmasıyla örtüşüyor (redundant), yeni bilgi katmıyor. En basit
     seçenek (H2, sadece temiz maskeleme, hiç regresyon yok) en iyi sonucu veriyor.**

   **GENEL SONUÇ (jeneratör-farkındalıklı imputasyon araştırması):**
   İki bağımsız deneyde (G-test: 0.96824→0.96828 Δ+0.00005; H-test: 0.96822→0.96828 Δ+0.00006)
   AYNI büyüklükte, AYNI yönde, tekrarlanabilir küçük bir kazanç bulundu — ama sadece "temiz
   residual maskeleme" (diff_daily_sum_clean, regresyon YOK) kısmından. Gerçek "imputasyon"
   (eksik değeri regresyonla tahmin edip doldurma) net katkı sağlamadı, çünkü LightGBM zaten
   diğer sütunlara raw+TE olarak erişiyor — ayrı bir imputasyon adımı bu bilgiyi yeniden
   paketlemekten öteye geçmiyor. **Gün-1'in "model-bazlı imputasyon BOZDU" bulgusu bugün de
   doğrulandı: gerçek/geniş kapsamlı imputasyon bu problemde işe yaramıyor.** Tek pozitif
   çıkarım: `diff_daily_sum`'ı hesaplarken eksik bileşeni 0 SAYMAK yerine (mevcut kod) satırı
   NaN bırakmak (`diff_daily_sum_clean`) küçük ama bedava, tutarlı bir iyileştirme — bu üretim
   pipeline'ına eklenebilir (kod değişikliği: `sum_components.sum(min_count=1)` yerine 4-sütun
   tam-dolu maskesi). Δ çok küçük olduğu için (~0.00005-0.00006) tek başına LB'ye atılmaya
   değmez, bir sonraki toplu iyileştirme paketiyle birlikte doğrulanmalı.

8. **Kullanıcı fikirleri: dominant-activity/rank ve stress_level sapması** (scripts/lgbm_fe_dominant_stressdev.py):
   - **Kategori I (dominant-activity/rank)**: hipotez — sabit ikili oranların aksine (trees birkaç
     split ile yaklaşık yakalıyor), 3 sütun (social/gaming/work) arasında "hangisi en yüksek"
     (argmax/rank) bilgisi trees için axis-aligned split'lerle kurması kombinatorik zor, farklı
     bir eksen olabilir. Özellikler: `max_activity3`, `range_activity3`, `gap_social_to_max`,
     `gap_gaming_to_max`, `gap_work_to_max`, `dominant_activity` (TE'lenmiş kategorik).
     **Sonuç: 0.96824→0.96829, Δ=+0.00005, KATKI VAR (küçük ama pozitif)** — hepsi 860-1199
     split ile kullanılmış, anlamlı ama mütevazı bir sinyal. Bugünkü temiz-residual bulgusuyla
     (+0.00006) aynı mertebede.
   - **Kategori J (stress_level grup sapması)**: hipotez daha zayıftı (trees stress_level'e split
     atıp ardından süreye split atarak zaten dolaylı yakalayabilir). **Sonuç: 0.96824→0.96822,
     Δ=-0.00001, KATKI YOK** (tahmin edildiği gibi, trees zaten yakalıyor).
   - I+J birlikte: 0.96828 (Δ=+0.00004, J'nin hafif negatifi I'yı biraz aşağı çekiyor)
   - **Bugünün küçük-pozitif bulgu havuzu birikti**: temiz residual (+0.00006) + dominant-activity
     (+0.00005) — hiçbiri tek başına LB'ye değecek büyüklükte değil ama ikisi FARKLI bilgi
     kaynaklarından geliyor (biri daily-vs-sum kısıtı, diğeri social-vs-gaming-vs-work iç
     karşılaştırması) — birlikte pakete konulup toplu test edilmeye aday.

   **Dış kaynak değerlendirmesi (Kaggle discussion, kullanıcının paylaştığı):**
   Muhammad Faheem / Georgy Mamarin / Dariush Afshar tartışması bizim kendi bulgularımızla
   noktasal olarak örtüşüyor: is_missing bayrakları katkısız (MCAR, bizim gün-1 bulgumuzla
   aynı), lineer kombinasyonlar (sleep_deficit, total_weekly_screen_time gibi toplamlar)
   katkısız/zararlı (bizim Kategori C/D ve etkileşim-çarpımı negatif sonuçlarımızla tutarlı),
   CV'de görünen küçük kazançların LB'de erimesi (Muhammad'ın 5-seed missing_count denemesi —
   bizim kalibrasyon kuralımızın bağımsız doğrulaması). "CTGAN" jeneratör iddiası teyit
   edilemedi (Kaggle sayfası JS-render, WebFetch içerik çekemedi) — spekülasyon olarak
   işaretlendi, güvenilir kaynak değil. Decimal-place (ondalık basamak) özellik fikri zayıf
   kanıtlı (Dariush'un kendisi "sadece bir sütunda tuttu" diyor), düşük öncelikli, henüz
   test edilmedi.

## 2026-08-12 (2. Gün)

### Durum (gün sonu)
- **En iyi LB: 0.96950** (3-model rank-average blend, raw+TE+14 ek özellik)
- **En iyi CV OOF: 0.96843**
- Dünden bugüne LB ilerlemesi: 0.96847 → 0.96950 (+0.00103)
- Yarışma bitiş: 31 Ağustos 2026 (19 gün kaldı)

### Bugün yapılanlar
1. **Kod kurtarma**: dünkü en iyi 2 sonucun (raw+TE, 3-model blend) kodu ve 6 tanı scripti
   %TEMP%\opencode\ altında kalmıştı, scripts/ altına taşındı (bkz. yukarıdaki not).
2. **LGB retune raw+TE seti üzerinde** (tune_lgbm_raw_te.py, 40 trial, holdout):
   holdout AUC 0.96649. Tam 5-fold CV'de doğrulandı (lgbm_raw_te_v2.py):
   **0.96721** — eski (raw-feature-tuned) params'ın 0.96726'sından **daha kötü** (gürültü
   seviyesinde, ~-0.00005). **Sonuç: LGB retune KATKI YAPMADI, tahmin edilen "en büyük
   kaldıraç" yanlış çıktı.** LGB zaten iyi tune edilmiş durumdaydı. Eski best_params_lgbm.json
   kullanılmaya devam ediliyor.
3. **XGBoost bağımsız Optuna tuning** (tune_xgb.py, GPU, 40 trial, holdout AUC 0.96648):
   min_child_weight=15.2 çıktı (eski çevrilmiş değer 300'den çok uzak, teori doğrulandı).
4. **CatBoost bağımsız Optuna tuning** (tune_cat.py, GPU, 40 trial, holdout AUC 0.96603):
   l2_leaf_reg=6.0 çıktı (eski varsayılan 3.0'dan farklı).
5. **Blend'i yeni XGB+CatBoost params ile güncelleyip çalıştırdık** (blend_lgb_xgb_cat.py,
   LGB params değişmedi çünkü retune katkı yapmamıştı):
   - LightGBM OOF: 0.96726 (değişmedi, beklenen)
   - XGBoost OOF: 0.96665 (5-fold CV, holdout'tan biraz yüksek — normal)
   - CatBoost OOF: 0.96682
   - **Blend (rank-avg) OOF: 0.96751** — eski blend'in 0.96737'sinden **+0.00014 iyi**
   - Saved: sub/lgbm_xgb_cat_rankblend_v2_2026-08-12.csv
   - **SONUÇ: XGB/CatBoost tuning işe yaradı (LGB retune'in aksine).**
   - **LB DOĞRULANDI: 0.96864** (tahmin ~0.9686-0.9687 idi, tuttu). Yeni en iyi skor.
     CV/LB farkı: 0.00113 — önceki denemelerin aralığında (0.0011-0.0021), overfitting yok.

6. **Stacking meta-model** (stack_logreg.py, aynı 5-fold ile nested OOF logistic regresyon):
   **0.96751 — rank-average ile birebir aynı, KATKI YOK.** Öğrenilen katsayılar neredeyse
   eşit çıktı (lgb=5.13, xgb=5.09, cat=5.11) → 3 model aynı feature set üzerinde eğitildiği
   için OOF'lar çok korele, ağırlıklandırmanın eşit-ağırlıktan sapacak alanı yok.

### Bugünkü net sonuç
- LGB retune (raw+TE seti): KATKI YOK (0.96726→0.96721, gürültü seviyesi)
- XGB+CatBoost bağımsız tuning: KATKI VAR (blend CV 0.96737→0.96751, **LB 0.96847→0.96864
  DOĞRULANDI**, yeni en iyi skor)
- Stacking meta-model: KATKI YOK (0.96751→0.96751, öğrenilen ağırlıklar zaten eşit)
- **Ders**: bugünkü kazançların hepsi tek bir yerden geldi — modellerin kendi hiperparametre
  kalitesinden (XGB/Cat tuning). Modelleri kombine etme şekli (rank-avg vs stacking) hiç fark
  etmedi çünkü hepsi aynı feature set'te eğitiliyor, gerçek çeşitlilik yok. Gerçek çeşitlilik
  için modellere FARKLI feature subset'leri / farklı seed'ler vermek gerekir (bkz. kaldıraç
  #4 çok-seedli bagging — bu en azından farklı random_state ile gerçek varyans yaratır).
7. **TE smoothing taraması** (te_smoothing_model_sweep.py, model-seviyesi 5-fold CV,
   SMOOTH ∈ {0.5,1,2,3,5,10,20,50}): 50→20→10→5 arası net monotonik artış (0.96718→0.96733),
   0.5-5 aralığı gürültü seviyesinde birbirine yakın (0.96733-0.96738). SMOOTH=20 kafadan
   seçilmiş, yanlışmış — **SMOOTH=3 seçildi**, tüm raw+TE script'lerinde güncellendi.
8. **Blend'i SMOOTH=3 ile tekrar çalıştırdık**: LightGBM 0.96726→0.96738, XGBoost
   0.96665→0.96670, CatBoost 0.96682→0.96699, **Blend 0.96751→0.96762** (+0.00011,
   üç modelde de tutarlı iyileşme). Saved: sub/lgbm_xgb_cat_rankblend_v3_smooth3_2026-08-12.csv
   - **LB DOĞRULANMADI: 0.96864 → 0.96862.** CV'deki +0.00011 kazanç LB'ye taşınmadı,
     hatta LB gürültü seviyesinde (-0.00002) düştü. **SONUÇ: TE smoothing 20→3 değişikliği
     gerçek bir kazanç değilmiş, CV fold gürültüsünü avlamışız.** 0.5-5 aralığının "gürültü
     seviyesinde" olduğu şüphesi (sweep sırasında not edilmişti) LB ile doğrulandı.
     Ders: dar aralıktaki (Δ<0.0001) CV farklarına güvenmeden önce LB ile teyit etmeden
     production'a alma — bu sefer ucuza (~10dk+1 submission) öğrenildi.
   - Şu an için EN İYİ SKOR hâlâ v2 (LB 0.96864, SMOOTH=20). SMOOTH=3 script'lerde kalabilir
     (istatistiksel olarak eşdeğer, maliyeti aynı) ama "iyileştirme" olarak sayılmamalı.

### Bugünkü net sonuç (güncellendi)
- LGB retune (raw+TE seti): KATKI YOK
- XGB+CatBoost bağımsız tuning: KATKI VAR (LB 0.96847→0.96864 DOĞRULANDI) — **bugünün tek
  net, LB'de doğrulanmış kazancı**
- Stacking meta-model: KATKI YOK
- TE smoothing (20→3): CV'de katkı görünüyordu, **LB'de doğrulanmadı** (gürültü)
- Kalan kaldıraçlar (öncelik güncellendi): çok-seedli bagging, 2-yönlü TE — ikisi de küçük
  kaldıraç olarak işaretlensin, bugünkü desene göre (küçük CV farkları LB'de kaybolabiliyor)
  önce büyük CV farkı (>0.0005) vermeyen deneyleri LB'ye taşımadan önce şüpheyle karşıla.

9. **Ratio özellikleri** (kullanıcı önerisi: ekran/uyku oranı + 6 benzer oran):
   `ratio_screen_sleep, ratio_work_daily, ratio_social_daily, ratio_opens_daily,
   ratio_social_sleep, ratio_weekend_sleep, ratio_gaming_daily`. Gerekçe: ağaç modelleri
   bölme işlemini split'lerle yaklaşık bile öğrenemez, bu yüzden ham oranı ayrı feature
   olarak vermek TE'den farklı, yeni bilgi katabilir. İzole LGB testinde: **0.96738→0.96771
   (+0.00033)**, bugünün en büyük tekil CV kazancı. Aynı anda test edilen "native categorical
   düzeltmesi" (gender/stress_level/academic_work_impact raw sütunlarının -999 yerine gerçek
   category dtype alması) KATKI VERMEDİ (0.96738→0.96736) — TE zaten bu zayıf kategorik
   sinyali tüketmiş.
   Tam blende taşındı: LightGBM 0.96738→0.96771, XGBoost 0.96670→0.96701, CatBoost
   0.96699→0.96736 — **Blend 0.96762→0.96793 (+0.00031), üç modelde de tutarlı.**
   Saved: sub/lgbm_xgb_cat_rankblend_v4_ratios_2026-08-12.csv
   - **LB DOĞRULANDI: 0.96862 → 0.96900 (+0.00038).** Bugünün en büyük tek kazancı,
     tahminin üst sınırına denk geldi. **YENİ EN İYİ SKOR: 0.96900.**
   - Native-cat izole test de LB'ye atıldı (doğru referans: tek-LGB baseline 0.96845,
     blend'le KARIŞTIRMA): sonuç **0.96838**, yani eski tek-LGB'den bile hafif kötü.
     CV'nin "katkı yok" verdiği sinyal LB ile birebir doğrulandı — bu sefer CV/LB
     çelişkisi yok, temiz negatif sonuç.

### Bugünkü net sonuç (3. güncelleme — GÜN SONU)
- XGB+CatBoost bağımsız tuning: KATKI VAR, LB DOĞRULANDI (0.96847→0.96864)
- TE smoothing (20→3): CV'de vardı, LB'de DOĞRULANMADI (gürültü, 0.96864→0.96862)
- **Ratio özellikleri (7 adet): KATKI VAR, LB DOĞRULANDI (0.96862→0.96900, +0.00038)**
  — bugünün en büyük ve en net kazancı
- Native categorical düzeltmesi: KATKI YOK, hem CV hem LB ile doğrulandı (temiz negatif)
- **Gün sonu en iyi skor: 0.96900** (sub/lgbm_xgb_cat_rankblend_v4_ratios_2026-08-12.csv)
- LB progresyon (tam): 0.96511→0.96513→0.96340→0.96531→0.96703→0.96845→0.96847→0.96864→0.96862→**0.96900**

10. **4 yeni FE kategorisi izole test edildi** (raw+TE+7ratio referans=0.96771 üzerine, her biri ayrı):
    - **A (ek oranlar: notif/daily, notif/sleep, opens/sleep, work/sleep, sum/daily): 0.96826 (+0.00055)**
      — bugünün en büyük CV sinyali
    - **B (farklar: daily-sum, weekend-daily): 0.96811 (+0.00040)** — güçlü
    - C (uyanık-gün bütçesi: daily/waking, other_life_hours): 0.96774 (+0.00003) — gürültü
    - D (ters-yön oran: session_len_opens/notif): 0.96772 (+0.00001) — gürültü
    - 5 dosya (referans + A/B/C/D) LB'de ayrı ayrı test edildi (izole submission metodolojisi).
      **SONUÇLAR (referans LB=0.96879):**
      - **A: LB 0.96934, Δ=+0.00055 — CV ile BİREBİR eşleşme (1.0x transfer). DOĞRULANDI.**
      - **B: LB 0.96923, Δ=+0.00044 — CV+0.00040'a yakın (1.1x). DOĞRULANDI.**
      - C: LB 0.96885, Δ=+0.00006 — CV+0.00003 ile tutarlı, gürültü, anlamsız.
      - D: LB 0.96883, Δ=+0.00004 — CV+0.00001 ile tutarlı, gürültü, anlamsız.
      - **Kalibrasyon dersi bugün 3. kez doğrulandı:** CV Δ>0.0003 olan değişiklikler LB'ye
        neredeyse 1:1 taşınıyor, CV Δ<0.00005 olanlar gürültüde kalıyor. Artık güvenilir bir
        karar kuralı: yeni bir FE denemesinde CV farkı bu eşiği geçmiyorsa LB'ye atmaya gerek yok.
      - **Önemli:** tek-model A (LB 0.96934) dünkü 3-model blend'den (LB 0.96900) bile iyi —
        A+B'yi birleştirip tam blende taşımak sıradaki adım, muhtemelen büyük bir sıçrama.

11. **A+B birleşimi tam 3-model blende taşındı** (blend_lgb_xgb_cat_AB.py, 42 özellik:
    28 baseline + 7 ratio + 5 A + 2 B):
    - LightGBM 0.96771→0.96824, XGBoost 0.96701→0.96744, CatBoost 0.96736→0.96792
    - **Blend 0.96793→0.96843 (+0.00050)**, üç modelde tutarlı
    - Saved: sub/lgbm_xgb_cat_rankblend_v5_AB_2026-08-12.csv
    - **LB DOĞRULANDI: 0.96900→0.96950 (+0.00050) — tahmin (~0.9695) tam tuttu.**
    - **GÜN SONU EN İYİ SKOR: 0.96950**

### GÜN SONU KESİN ÖZET (2026-08-12)
- Gün başı LB: 0.96847 → Gün sonu LB: **0.96950** (+0.00103)
- İşe yarayan değişiklikler (hepsi LB'de doğrulandı): XGB/CatBoost bağımsız tuning (+0.00017),
  ratio özellikleri×2 tur — ilk 7 (+0.00038) + ek 7 (A+B, +0.00050)
- İşe yaramayan/nötr: LGB retune, stacking, TE smoothing (20→3), native categorical düzeltmesi,
  kategori C (uyanık-gün bütçesi), kategori D (ters-yön oran)
- **Kalibrasyon kuralı (bugün 3 kez doğrulandı, yarın için kullan):** CV Δ>0.0003 olan bir
  değişiklik LB'ye ~1.0-1.1x oranında taşınıyor, güvenilir. CV Δ<0.00015 olanlar gürültü,
  LB'ye atmadan önce şüpheyle karşıla (gerekirse CV'de daha büyük örneklem/tekrar ile teyit et).
- **Üretim reçetesi (yarın buradan devam):** raw+TE(SMOOTH=3, 14 sütun) + 7 orijinal ratio
  + 5 kategori-A ratio + 2 kategori-B fark = 42 özellik, LGB(best_params_lgbm.json, değişmedi)
  + XGB(best_params_xgb.json) + CatBoost(best_params_cat.json), rank-average blend.
  Script: scripts/blend_lgb_xgb_cat_AB.py — **yarın buradan çoğaltarak devam et.**

### YARIN İÇİN BAŞLANGIÇ NOKTASI
1. En iyi pipeline: `scripts/blend_lgb_xgb_cat_AB.py` (LB 0.96950) — yeni denemeler buna göre kıyaslanmalı
2. Henüz denenmemiş, gündemde kalan kaldıraçlar:
   - Çok-seedli bagging (2-3 seed, tahminleri ortalama) — küçük kaldıraç bekleniyor
   - 2-yönlü joint TE (ör. TE(social_media_hours, notifications_per_day) çifti) — orta öncelik,
     bugün ratio'ların başarısı bunu da gündemde tutuyor ama garanti değil
   - Kategori D'nin reddedilen fikirlerinden farklı, denenmemiş ratio/fark kombinasyonları olabilir
     (bugün A+B'nin işe yaraması, aynı ailede daha fazla adayın da işe yarayabileceğini gösteriyor —
     örn. gaming/sleep, social/gaming gibi denenmemiş çiftler)
3. **Metodoloji hatırlatma:** yeni değişiklikleri İZOLE test et (CV'de, referansa göre), sadece
   gerçek sinyal (Δ>0.0003) verenleri tam blende taşı ve LB ile doğrula — bugün bu yöntem
   4/4 doğru karar verdirdi (A✓ B✓ C✗doğru-red D✗doğru-red)
4. Dosyalar: `sub/2026-08-12/` klasöründe bugünün tüm submission'ları arşivli,
   `sub/submission_log.xlsx`'te tam kayıt var

## 2026-08-11 (1. Gün)

### Durum
- **En iyi LB: 0.96847** (3-model rank-average blend)
- **En iyi CV OOF: 0.96737**
- Yarışma bitiş: 31 Ağustos 2026 (21 gün kaldı)

### Bugün yapılanlar (sırayla)
1. **EDA** — sütunlar, hedef dağılımı (%71 bağımlı), NaN analizi
2. **NaN deneyleri** (4 run):
   - Ham NaN (native): CV 0.96349 → **kazanan**
   - is_missing bayrakları: 0.96347 (katkı yok, MCAR çıktı)
   - Model-bazlı imputasyon: 0.96134 (BOZDU)
   - daily-only semantik: 0.96206 (BOZDU)
   - **Sonuç: LightGBM'e ham NaN ver, hiçbir şey doldurma**
3. **Feature engineering**: `sum_components`, `ratio_weekend_daily` → +0.00018
4. **Optuna tuning** (40 deneme): CV 0.96349 → 0.96571
5. **Target Encoding** (yazar metodolojisi):
   - Native categorical: 0.96030 (başarısız)
   - TE tek başına (10-fold OOF): 0.96073 (başarısız)
   - **raw + TE birleşimi: 0.96726** → LB 0.96845 (büyük kazanç)
   - Etkileşim çarpımları: 0.96698 (katkı yok)
6. **3-model blend** (LGB CPU + XGB GPU + CatBoost GPU + rank-avg): 0.96737 → **LB 0.96847**

### Jeneratör tersine mühendislik bulguları
- **Uniform grid'ler**: sleep 0.01 adım, notifications/app_opens tamsayı, age 18-35
- **İki feature arketipi**:
  - Monotonik (daily, social, weekend): P(bağımlı|değer) düzenli artıyor
  - Lookup-table (notifications, app_opens): her tamsayı = kaynak veriden öğrenilmiş rastgele olasılık
    (42→0.65, 43→0.89, 44→0.61)
- **Hedef etkileşimli üretilmiş**: logistic toplamsal 0.948 vs LightGBM 0.967
- **Kural AUC (daily>8 veya social>4): 0.816** — kural sabitleme YAPMA (tavan)
- Train/test dağılımı aynı (PSI=0) → CV güvenilir, LB'yi tahmin ediyor

### Kalan kaldıraçlar (yarın için, önem sırasıyla)
1. **Params retune (en büyük kazanç ~+0.001-0.002):**
   - Optuna tuning sadece raw özelliklerde yapıldı, ama en iyi model raw+TE (28 özellik)
   - Optuna'yı raw+TE seti üzerinde LGB + XGB + CatBoost için ayrı ayrı çalıştır
   - Başlangıç noktası artık var: scripts/lgbm_raw_te.py
2. **XGBoost + CatBoost tuning (~+0.0005, önceliği yükselt):** blend'deki model params'ları
   LightGBM'den çevrilmişti, bağımsız tune edilmedi. Somut sorun tespit edildi (2026-08-12):
   - XGB `min_child_weight=300` (hessian-toplamı bazlı) LGB'nin `min_child_samples=428`
     (satır-sayısı bazlı) ile aynı birim değil. prior=0.71 olduğundan ort. hessian≈0.206/satır,
     yani 300 ≈ ~1500 satırlık bir minimum → LGB'nin 428'inden ~3.5x daha katı bir kısıt.
     XGB muhtemelen fazla regularize ediliyor.
   - CatBoost `l2_leaf_reg=3.0` hiç çevrilmemiş, kütüphane varsayılanı — sadece depth/lr tahmini.
   - Bağımsız Optuna tuning ile bu ikisi muhtemelen daha büyük katkı verecek, tahmini yükselt.
3. **Stacking meta-model (~+0.0003):** rank-average yerine 3 modelin OOF'larına logistic regresyon
4. **Çok-seedli bagging (~+0.0002):** 2-3 seed'le CV, tahminleri ortalama
5. **Ortak 2-yönlü TE (~+0.0003):** TE(social|notifications) gibi değer çifti kombinasyonları

### Dosya yapısı
```
scripts/  → tüm py scriptleri
  lgbm_raw_te.py            → EN İYİ TEKİL MODEL (raw+TE, 28 özellik, CV 0.96726 / LB 0.96845)
  lgbm_raw_te_inter.py      → raw+TE+etkileşim varyantı (katkı yok, CV 0.96698)
  blend_lgb_xgb_cat.py      → EN İYİ SKOR (3-model rank-blend, CV 0.96737 / LB 0.96847)
  eda_*, te_diag_*          → tanı/EDA scriptleri (jeneratör analizi, PSI, imputability, TE smoothing sweep)
sub/      → submission_log.xlsx (takip), best_params_lgbm.json
sub/2026-08-11/  → bugünün tüm submission'ları (tarihli)
analiz.ipynb  → sadece EDA analizleri
```
NOT (2026-08-12): lgbm_raw_te.py, lgbm_raw_te_inter.py, blend_lgb_xgb_cat.py ve 6 tanı scripti
dün %TEMP%\opencode\ altında scratch olarak çalıştırılmış, projeye hiç kaydedilmemişlerdi.
Bugün scripts/ altına taşındı — kalıcı hale geldi, reprodüksiyon sorunu çözüldü.

### LB progresyon
```
0.96511 → 0.96513 → 0.96340 → 0.96531 → 0.96703 → 0.96845 → 0.96847 → 0.96864
```

---

## 2026-08-30 (gece, son hamle: OOF havuzu genişletme)

### Yapılanlar (sırayla)
1. **Güncel topluluk kütüphaneleri indirildi** (`data/extra_oof/`):
   - `najiama/predicting-smartphone-addiction-oof-submission-csv` → 01-05 (bizde naji01-05 zaten var) + `07..19_blend_oof` (yeni)
   - `dariushafshar/s6e8-golem-oof-library` → oof_a..g (7 model, aynı **frozen 5-fold seed-42** şeması)
   - `raykkretzschmar/s6e8-fm-lattice-blend-members` → 5 FM üyesi (aynı şema; çıktılar ham skor → z-score) + band-local dosyaları
   - `paiky1995/s6e8-oof-library-11-members` → 11 NN üyesi (**10-fold / şema uyumsuz** — dikkatle)
   - `mohankrishnathalla/s6e8-{xgb,lgb-dart,cat-mlp}-oof` → oof_xgb/lgb/cat_v3 (AUC 0.965-0.966)
   - `najiama/predicting-smartphone-addiction-psa` → **403 erişilemedi** (hesap/izinsizlik) → Rayk'ın 0.97100 submission'ı alınamadı
2. **Hizalama doğrulaması** her üye için: satır sayısı + AUC + max-korelasyon (bizim 76'ya karşı → 0.9342-0.9964).
3. **Yeni üyelerin solo değerleri**: najiblen_19 **0.97010** (maxcorr 0.9619 — hem güçlü hem dekorrelasyonlu!), najiblen_14 0.96970, najiblen_18 0.96986, paiky_v14/v19/v23 ~0.9687-0.9689, rayk_fmplr 0.96739 (maxcorr 0.9553).
4. **Stack'ler (dürüst 5-fold meta-CV)**:
   - base76 (eski) LR = **0.96969**
   - **98 üye (76+dariush7+rayk5+najiblen10) LR = 0.96995** (+0.00026, en sağlam kazanç)
   - 109 üye (+paiky11) LR = **0.96999** (+0.00004; paiky'nin partition-mix optimizmi riski — rayk uyardı)
   - 98 üye XGB meta = **0.97003** (lineer LR'yi geçti)
   - 98 üye CatBoost meta = **0.96990**
   - logit-avg(exp_aligned, exp_xgb) = **0.97002**
   - **2. seviye meta-LR [aligned,xgb,cat,all,ax] = 0.97004 (EN İYİ)**
   - paiky'sız temiz muadil: **meta-LR [aligned,xgb,cat] = exp_meta3 = 0.97003** (paiky'nin +0.00001'i gürültü → öneri #1 temiz üyelerden)
5. **Denenen ve ELENEN**:
   - `exp_w66` (w66'yı üye olarak ekle): 0.96994 → katkı yok (LR zaten span ediyor)
   - `exp_a101` (+mkn 3 üye): 0.96995 → katkı yok
   - `exp_l1_fast` (liblinear): 0.96967 → LR'den iyi değil
   - saga-L1/elasticnet 98 üyede: 20 dk'da bitmedi → vazgeçildi (76 üyede zaten 0.96969)
   - **Rayk tarzı band-düzeltme** (3-6h / 6-7.8h band-FM karıştırıcı, n_missing<4 filtresiyle): band içi delta ±0.00001 → **nötr**, kullanılmadı. (Bizim global o bandlarda 0.91898/0.93479 ile zaten Rayk'ın blend'inden güçlüydü.)
6. **Bozuk dosya tespiti**: `extra_ens5`, `extra_topk_avg12/6` submission'ları **sabit 1.0** içeriyor (stack_extra_2026-08-30.py'nin test tahmini bug'ı) → **KULLANMA**, final listesinden çıkarıldı. (OOF skorları sağlıklıydı, test vektörü bozuktu.)
7. Checkpoint `nn_cache/stack_checkpoint.json` → 32 varyant. Final tablo + korrelasyon: `nn_cache/final10_table.csv`, `final10_corr.npy`.

### FİNAL 10 SUBMISSION (öncelik sırasıyla; hepsi OOF>=0.9694, w66 LB kanıtlı)
| # | dosya | OOF | açıklama |
|---|---|---|---|
| 1 | `exp_meta3` | 0.97003 | 2. seviye LR(aligned,xgb,cat) — **TEMİZ**, önerilen |
| 2 | `exp_meta6` | 0.97004 | 2. seviye LR(+all,+ax) — maks OOF (paiky hedge) |
| 3 | `stack_exp_xgb` | 0.97003 | 98 üye XGB meta (non-lineer) |
| 4 | `exp_ax` | 0.97002 | logit-avg(aligned, xgb) |
| 5 | `stack_exp_all` | 0.96999 | +paiky hedge (partition caveat) |
| 6 | `stack_exp_aligned` | 0.96995 | 98 üye LR flagship |
| 7 | `blend_gbdt_origfeat_nn_featfull_w66` | LB **0.97035** | bilinen-iyi (güvenli çapa) |
| 8 | `stack_logit_lr` | 0.96969 | eski 76 üye LR |
| 9 | `stack_nnls` | 0.96943 | en dekorrelasyonlu (corr 0.976) |
| 10 | `stack_extra_lr_raw9` | 0.96968 | raw-features yaklaşımı (farklı aile) |

Test-pred korrelasyonları: meta6/xgb/ax/all/aligned/elasticnet/logit_lr/raw9 birbirine 0.994-1.000 (aynı havuzun dirili üyeleri), **stack_nnls en farklı (0.975-0.978)** — çeşitlilik için #9 şart. Tümü `sub/2026-08-30/` + `sub/2026-08-29/` altında, id sıralı, [1e-6, 1] aralığında, max-norm'lu.

### Beklenti & notlar
- OOF 0.96969 → (önceki transfer ~+0.0010-0.0012) LB ~0.9706-0.9708. Yeni 0.97004 → **beklenen LB ~0.9709-0.9711**. w66 (0.97035) çapa olarak alt sınır güvencesi.
- **CLI submission yine yasak** (kaggle.json dante29 hesabı, gerçek: talhatursun) → CSV'leri web arayüzünden manuel yükleyen kullanıcıya. Önce #1, sonra #2..#7.
- Yarışma bitişi: 31 Ağustos 23:59 UTC. 10 dosya da bugün yüklenmeli.
