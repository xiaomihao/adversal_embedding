# Adversal_Embedding
词嵌入层的对抗训练组件

## 用途
* 增加模型鲁棒性和泛化性

## 安装
```
pip install adversal_embedding

```
## 使用
```
from keras.models import load_model
from adversal_embedding.core import adversal_embedding

model = load_model(model_dir)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
adversarial_training(model, 'embedding_vocab', 0.5)  # 用在compile之后
model.fit(train_data, test_data, epochs=epochs, batch_size=batch_size)

```

