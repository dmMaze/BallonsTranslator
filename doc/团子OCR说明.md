简体中文 | [pt_BR](../doc/Manual_TuanziOCR_pt-BR.md)

## 官方提供的请求参数参考：
<p align = "center">
<img src="https://github.com/PiDanShouRouZhouXD/BallonsTranslator/assets/38401147/3c3985e9-f36e-41fb-af94-d6a8088e5ccd" width="85%" height="85%">

</p>

## 团子OCR说明

### 登录
第一次登录时可能会提示密码出错等问题，可以在确认正确输入后勾选并取消勾选`force_refresh_token`选项，以重新登陆。保存后即可正常使用。

### 文本检测
文本检测功能也会提取出文字，而且是整体识别提取。所以当有使用团子的需求时，推荐不要单独使用OCR功能，而是使用团子的文本检测与none_ocr。
团子有自带的拟声词过滤等功能，详细参数设置请参考上方的`官方提供的请求参数参考`