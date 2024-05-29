#SimCSE无监督训练

对simcse项目进行二次开发，改成基于Trainer的形式训练，支持多卡训练

##数据集格式：
txt数据，一条数据为一行，可根据需要换成自己的领域数据，无需标签

##模型：

在stella基础上做微调，可根据需要更换，可根据需要更换模型

##使用方式：

python train.py

支持deepspeed训练：

deepspeed train.py --deepspeed config.json

##测试

python test.py

可使用sentence_transformer方式或者直接推理方式
