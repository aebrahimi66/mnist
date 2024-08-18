```python
from fastai.vision.all import *
path = untar_data(URLs.MNIST_SAMPLE)
```
```python
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
```
```python
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
valid_3_tens = torch.stack([tensor(Image.open(o))
for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o))
for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
```
ما قبلاً روی xs کار کرده‌ایم - یعنی متغیرهای مستقل ما، تصاویر 
خود را داریم. ما تمام آنها را به یک تنسور واحد تبدیل می‌کنیم و همچنین آن‌ها را از لیست ماتریس‌ها (یک تنسور رتبه ۳) به لیست بردارها (یک تنسور رتبه ۲) تغییر می‌دهیم. این کار با استفاده از view انجام می‌شود که یک روش در PyTorch است که شکل یک تنسور را بدون تغییر محتوای آن تغییر می‌دهد. -1 یک پارامتر خاص برای view است که به معنی "این محور را به اندازه کافی بزرگ کنید تا همه داده‌ها را جا دهد" است.
```python
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
```
`
(torch.Size([12396, 784]), torch.Size([12396, 1]))
`

در PyTorch، یک مجموعه داده (Dataset) باید هنگامی که بازرسی می‌شود، یک توپل از (x, y) برگرداند. پایتون یک تابع zip را فراهم می‌کند که وقتی با لیست ترکیب شود، راه ساده‌ای برای دستیابی به این قابلیت ارائه می‌دهد:
```python
dset = list(zip(train_x,train_y))
x,y = dset[0]
x.shape,y
```
`
(torch.Size([784]), tensor([1]))`
```pytho
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))
```

حالا  به یک وزن تصادفی (مقداردهی اولیه) برای هر پیکسل نیاز داریم :
```python
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
weights = init_params((28*28,1))
```

تابع weights * pixels انعطاف‌پذیر نخواهد بود - همیشه برابر با 0 خواهد بود زمانی که پیکسل‌ها برابر با 0 هستند (یعنی، عرض از مبداء آن 0 است). شاید از ریاضیات دبیرستانتان خاطره‌ای از فرمول خط باشد که y = w * x + b است؛ ما هنوز هم به b نیاز داریم. ما آن را نیز به یک عدد تصادفی مقداردهی اولیه خواهیم کرد
```python
bias = init_params(1)
```

در شبکه‌های عصبی، w در معادله y = w * x + b به عنوان وزن‌ها شناخته می‌شود و b به عنوان بایاس نامیده می‌شود. وزن‌ها و بایاس‌ها به همراه هم، پارامترهای مدل را تشکیل می‌دهند.

اکنون می‌توانیم برای یک تصویر پیش‌بینی  کنیم:
```python
(train_x[0]*weights.T).sum() + bias
```
`tensor([20.2336], grad_fn=<AddBackward0>)`

در پایتون، ضرب ماتریسی با عملگر @ نمایش داده می‌شود. بیایید امتحان کنیم:
```python
def linear1(xb): return xb@weights + bias
preds = linear1(train_x)
preds
```
```
tensor([[20.2336],
[17.0644],
[15.2384],
...,
[18.3804],
[23.8567],
[28.6816]], grad_fn=<AddBackward0>)
```

عنصر اول همان چیزی است که قبلاً محاسبه کردیم، همانطور که انتظار داشتیم. این معادله، batch @ weights + bias، یکی از دو معادله اساسی هر شبکه عصبی است (معادله دیگر تابع فعال‌سازی است ). بیایید دقتمان را بررسی کنیم. برای تصمیم‌گیری اینکه آیا خروجی نمایانگر 3 یا 7 است، می‌توانیم فقط بررسی کنیم که آیا بیشتر از 0 است، بنابراین دقت ما برای هر آیتم می‌تواند به شرح زیر محاسبه شود (با استفاده از broadcasting، بنابراین هیچ حلقه‌ای نیست!):
```python
corrects = (preds>0.0).float() == train_y
corrects
```
```
tensor([[ True],
[ True],
[ True],
...,
[False],
[False],
[False]])
```
```python
corrects.float().mean().item()
```
` 0.4912068545818329`

