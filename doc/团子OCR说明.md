## 官方提供的请求参数参考：
<p align = "center">
<img src="https://github.com/PiDanShouRouZhouXD/BallonsTranslator/assets/38401147/3c3985e9-f36e-41fb-af94-d6a8088e5ccd" width="85%" height="85%">

</p>

## Token 获取方法

### 方法1：从cookies中获取token

在浏览器中登录并访问[星河云OCR](https://cloud.stariver.org.cn/)，在浏览器的开发者工具中查看`cookie`，其中包含`token`字段，复制其值。
<p align = "center">
<img src="https://github.com/PiDanShouRouZhouXD/BallonsTranslator/assets/38401147/ae2cbcec-b426-4396-a484-62aa09f22cf6" width="50%" height="50%">

</p>

### 方法2：通过API获取token

通过API获取token的方法如下：

```
POST https://capiv1.ap-sh.starivercs.cn/OCR/Admin/Login


Request Body:
{
    "User": "your_username",
    "Password": "your_password"
}

Response Body:
{
    …
    "Token": "your_token"
    …
}
```

其中，`User`和`Password`为登录团子OCR的用户名和密码，`Token`为登录成功后返回的token。
