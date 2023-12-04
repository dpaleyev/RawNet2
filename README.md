# RawNet2

## Installation

```shell
pip install -r requirements.txt
```

Веса доступны по [ссылке](https://drive.google.com/file/d/1dR9IMMy3JHfKjo12ZHjvf9gzh7CCEplw/view?usp=sharing)

```shell
gdown https://drive.google.com/uc?id=1dR9IMMy3JHfKjo12ZHjvf9gzh7CCEplw
```

Загрузка данных по [ссылке](https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y)

## Training

``` shell
python train.py -c config.json
```

## Inference

**Для инференса все входные аудио должны быть в формате flac.**

```shell
python test.py --resume checkpoint.pth --config config.json --input /test_data --output /result.txt
```
