# تصميم الابتسامة الرقمي

نظام لتحليل وتصميم الابتسامة لعيادة أسنان، يعمل بالكامل على جوال أو تابلت الطبيب (PWA) بلا خادم. مشروع تخرج في التعلّم العميق.

- **التطبيق:** `https://alnsyryfwzy55-source.github.io/digital-smile-design-ai/`
- **الدفتر:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alnsyryfwzy55-source/digital-smile-design-ai/blob/main/training/smile_teeth_training.ipynb)

## ما يفعله

صور المريض ← قص تلقائي ← كشف الأسنان الست بنموذج U-Net + MobileNetV2 ← حالة كل سن (موجود/مفقود/مكسور) ← معايرة بالملم ← نقاط الوجه (MediaPipe) ← الشفاه في الراحة ← تصميم بالقواعد (الذهبية/RED، خط الوسط، طول الثنية) مع محاكاة ← فحص سلامة واعتماد الطبيب ← Case Sheet للمعمل وعرض قبل/بعد للمريض.

البيانات والصور تبقى على جهاز الطبيب فقط، مع نسخ احتياطي مشفّر (AES-256).

## تجربة كاملة

1. **النموذج:** افتح الدفتر في Colab، واكتب اسم المستودع في `GITHUB_REPO`، ثم Run all. الوضع الافتراضي `evaluate` يقيّم النموذج المنشور على صور الاختبار في نحو 5 دقائق بلا GPU. وضع `train` يعيد التدريب من الصفر.
2. **التطبيق:** افتح رابط التطبيق في Chrome، وأنشئ مريضاً وحالة، وارفع صورة أمامية بمباعد (يمكنك استخدام صورة من `data/smile_teeth_v2.zip`، مجلد `test/images`).
3. **الاختبارات البرمجية:** `cd app && node test/core.test.js` (18 اختبار وحدة)، و`python3 test/e2e.py /tmp/shots` (اختبار شامل في Chrome، يحتاج playwright).

## بنية المستودع

```
docs/        التطبيق المبني (GitHub Pages) + teeth_unet_v2.onnx
app/         كود التطبيق والاختبارات ودليل التشغيل (app/README.md)
training/    دفتر التدريب والتقييم
data/        بيانات التدريب smile_teeth_v2.zip ومصادرها
PROJECT_DOCS.md   التوثيق الكامل للمشروع والقرارات
```

## النتائج

| النموذج | Dice اختبار | IoU اختبار | الحجم |
| --- | --- | --- | --- |
| U-Net + MobileNetV2 (المعتمد) | 0.956 (0.960 مع TTA) | 0.917 | 16.5 MB |

على 20 صورة اختبار من مجموعات مفتوحة. التفاصيل في `training/README.md`.

## البيانات والترخيص

بيانات التدريب من Roboflow Universe بترخيص CC BY 4.0:
[Teeth-detection-FDI-System](https://universe.roboflow.com/rizka-milandga-2k04u/teeth-detection-fdi-system)،
[tooth_frontal](https://universe.roboflow.com/gradproject-g265b/tooth_frontal)،
[teeth](https://universe.roboflow.com/tooth-yqbyw/teeth-lhqjb).
التفاصيل والتعديلات في `data/README.md`.

## تنبيه

أداة مساعدة للطبيب، وليست جهاز تشخيص. كل قياس واقتراح يراجعه الطبيب ويعتمده. دقة الكشف التلقائي على صور العيادة لم تُتحقق بعد؛ انظر «قبل أي استخدام على مرضى» في `app/README.md`.
