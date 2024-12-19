import click
import torch
from pathlib import Path
import pkgutil
from tqdm import tqdm
import json

from imagen import load_imagen_from_checkpoint
from data import Collator
from utils import safeget
from imagen import ImagenTrainer, ElucidatedImagenConfig, ImagenConfig
from datasets import load_dataset, concatenate_datasets


def exists(val):
    """
    检查值是否存在（即不为 None）。

    Args:
        val (Optional[Any]): 要检查的值。

    Returns:
        bool: 如果值存在（即不为 None）则返回 True，否则返回 False。
    """
    return val is not None


def simple_slugify(text: str, max_length = 255):
    """
    将文本转换为简单的 URL 友好的 slug。

    处理步骤：
    1. 将 '-' 替换为 '_'。
    2. 移除 ','。
    3. 将空格 ' ' 替换为 '_'。
    4. 将 '|' 替换为 '--'。
    5. 去除开头和结尾的 '-', '_', '.', '/' 和 '\\'。
    6. 截断到指定的最大长度。

    Args:
        text (str): 要转换的原始文本。
        max_length (int, optional): slug 的最大长度，默认为 255。

    Returns:
        str: 转换后的 slug。
    """
    # 替换特定字符并去除不需要的字符，截断到最大长度
    return text.replace('-', '_').replace(',', '').replace(' ', '_').replace('|', '--').strip('-_./\\')[:max_length]


def main():
    pass


@click.group()
def imagen():
    """
    Image CLI 工具的入口点。
    """
    pass


@imagen.command(help = 'Sample from the Imagen model checkpoint')
# 定义 --model 选项，默认值为 './imagen.pt'
@click.option('--model', default = './imagen.pt', help = 'path to trained Imagen model')
# 定义 --cond_scale 选项，默认值为 5
@click.option('--cond_scale', default = 5, help = 'conditioning scale (classifier free guidance) in decoder')
# 定义 --load_ema 选项，默认值为 True
@click.option('--load_ema', default = True, help = 'load EMA version of unets if available')
# 定义一个位置参数 'text'
@click.argument('text')

def sample(
    model,         # 模型路径
    cond_scale,    # 条件尺度
    load_ema,      # 是否加载 EMA 版本
    text           # 输入文本
):
    # 将模型路径转换为 Path 对象
    model_path = Path(model)
    # 获取模型的绝对路径
    full_model_path = str(model_path.resolve())
    assert model_path.exists(), f'model not found at {full_model_path}'
    # 加载模型文件
    loaded = torch.load(str(model_path))

    # get version

    version = safeget(loaded, 'version')
    print(f'loading Imagen from {full_model_path}, saved at version {version} - current package version is {__version__}')

    # get imagen parameters and type
    # 获取 Imagen 参数和类型

    # 从检查点加载 Imagen 模型
    imagen = load_imagen_from_checkpoint(str(model_path), load_ema_if_available = load_ema)
    # 将模型移动到 GPU
    imagen.cuda()

    # generate image
    # 生成图像

    # 使用输入文本生成图像
    pil_image = imagen.sample([text], cond_scale = cond_scale, return_pil_images = True)

    # 生成图像保存路径
    image_path = f'./{simple_slugify(text)}.png'
    # 保存图像
    pil_image[0].save(image_path)

    print(f'image saved to {str(image_path)}')
    return


@imagen.command(help = 'Generate a config for the Imagen model')
# 定义 --path 选项，默认值为 './imagen_config.json'
@click.option('--path', default = './imagen_config.json', help = 'Path to the Imagen model config')

def config(
    path # 配置文件的路径
):
    # 从包中获取默认配置数据
    data = pkgutil.get_data(__name__, 'default_config.json').decode("utf-8") 
    with open(path, 'w') as f:
        f.write(data) # 将默认配置写入指定的文件路径


@imagen.command(help = 'Train the Imagen model')
# 定义 --config 选项，默认值为 './imagen_config.json'
@click.option('--config', default = './imagen_config.json', help = 'Path to the Imagen model config')
# 定义 --unet 选项，默认值为 1，类型为 1 到 3 之间的整数
@click.option('--unet', default = 1, help = 'Unet to train', type = click.IntRange(1, 3, False, True, True))
# 定义 --epoches 选项，默认值为 50
@click.option('--epoches', default = 50, help = 'Amount of epoches to train for')

def train(
    config,      # 模型配置的路径
    unet,        # 要训练的 Unet 编号
    epoches,     # 训练的 epoch 数量
):
    # check config path
    # 检查配置文件路径
    # 将配置路径转换为 Path 对象
    config_path = Path(config)
    # 获取配置的绝对路径
    full_config_path = str(config_path.resolve())
    assert config_path.exists(), f'config not found at {full_config_path}'
    
    with open(config_path, 'r') as f:
        # 读取并解析配置文件内容
        config_data = json.loads(f.read())

    assert 'checkpoint_path' in config_data, 'checkpoint path not found in config'
    
    # 获取 checkpoint 路径
    model_path = Path(config_data['checkpoint_path'])
    # 获取 checkpoint 的绝对路径
    full_model_path = str(model_path.resolve())
    
    # setup imagen config
    # 设置 Imagen 配置

    # 根据配置类型选择 Imagen 配置类
    imagen_config_klass = ElucidatedImagenConfig if config_data['type'] == 'elucidated' else ImagenConfig
    # 创建 Imagen 模型配置
    imagen = imagen_config_klass(**config_data['imagen']).create()

    trainer = ImagenTrainer(
    imagen = imagen, # 传入 Imagen 模型
        **config_data['trainer'] # 传入训练器配置
    )

    # load pt
    # 加载模型参数
    if model_path.exists():
        loaded = torch.load(str(model_path)) # 加载模型文件
        version = safeget(loaded, 'version')
        print(f'loading Imagen from {full_model_path}, saved at version {version} - current package version is {__version__}')
        trainer.load(model_path) # 加载模型到训练器
        
    if torch.cuda.is_available():
        trainer = trainer.cuda()

    # 获取图像大小
    size = config_data['imagen']['image_sizes'][unet-1]

    # 获取最大批量大小，如果未指定则默认为 1
    max_batch_size = config_data['max_batch_size'] if 'max_batch_size' in config_data else 1

    # 默认通道为 RGB
    channels = 'RGB'
    if 'channels' in config_data['imagen']:
        # 确保通道数是否在有效范围内
        assert config_data['imagen']['channels'] > 0 and config_data['imagen']['channels'] < 5, 'Imagen only support 1 to 4 channels L, LA, RGB, RGBA'
        if config_data['imagen']['channels'] == 4:
            channels = 'RGBA'  # 颜色带透明度
        elif config_data['imagen']['channels'] == 2:
            channels == 'LA' # 亮度（灰度）带透明度
        elif config_data['imagen']['channels'] == 1:
            channels = 'L' # 亮度（灰度）

    # 确保配置文件中是否存在批量大小
    assert 'batch_size' in config_data['dataset'], 'A batch_size is required in the config file'
    
    # load and add train dataset and valid dataset
    # 加载并添加训练数据集和验证数据集
    ds = load_dataset(config_data['dataset_name'])
    
    # 初始化训练数据集
    train_ds = None
    
    # if we have train and valid split we combine them into one dataset to let trainer handle the split
    # 如果有训练和验证分割，则将它们合并为一个数据集，让训练器处理分割
    if 'train' in ds and 'valid' in ds:
        # 合并训练和验证数据集
        train_ds = concatenate_datasets([ds['train'], ds['valid']])
    elif 'train' in ds:
        # 仅使用训练数据集
        train_ds = ds['train']
    elif 'valid' in ds:
        # 仅使用验证数据集
        train_ds = ds['valid']
    else:
        # 使用整个数据集
        train_ds = ds
    
    # 确保训练数据集是否存在
    assert train_ds is not None, 'No train dataset could be fetched from the dataset name provided'
    
    # 添加训练数据集到训练器
    trainer.add_train_dataset(
        ds = train_ds, # 训练数据集
        collate_fn = Collator( # 数据收集器
            image_size = size, # 图像大小
            image_label = config_data['image_label'], # 图像标签
            text_label = config_data['text_label'], # 文本标签
            url_label = config_data['url_label'], # URL 标签
            name = imagen.text_encoder_name, # 文本编码器名称
            channels = channels # 通道数
        ),
        **config_data['dataset']
    )
    
    # 判断是否需要验证、采样和保存模型
    should_validate = trainer.split_valid_from_train and 'validate_at_every' in config_data
    should_sample = 'sample_texts' in config_data and 'sample_at_every' in config_data
    should_save = 'save_at_every' in config_data
    
    # 设置验证、采样和保存的间隔
    valid_at_every = config_data['validate_at_every'] if should_validate else 0
    assert isinstance(valid_at_every, int), 'validate_at_every must be an integer'
    sample_at_every = config_data['sample_at_every'] if should_sample else 0
    assert isinstance(sample_at_every, int), 'sample_at_every must be an integer'
    save_at_every = config_data['save_at_every'] if should_save else 0
    assert isinstance(save_at_every, int), 'save_at_every must be an integer'
    sample_texts = config_data['sample_texts'] if should_sample else []
    assert isinstance(sample_texts, list), 'sample_texts must be a list'
    
    # check if when should_sample is true, sample_texts is not empty
    # 检查当 should_sample 为 True 时，sample_texts 不能为空
    assert not should_sample or len(sample_texts) > 0, 'sample_texts must not be empty when sample_at_every is set'
    
    # 开始训练循环
    for i in range(epoches):
        # 遍历训练数据加载器中的每个批次
        for _ in tqdm(range(len(trainer.train_dl))):
            # 执行训练步骤，返回损失值
            loss = trainer.train_step(unet_number = unet, max_batch_size = max_batch_size)
            # 输出损失值
            print(f'loss: {loss}')

        # 如果当前 epoch 需要进行验证，并且训练器是主进程，则执行验证步骤
        if not (i % valid_at_every) and i > 0 and trainer.is_main and should_validate:
            # 执行验证步骤，返回验证损失值
            valid_loss = trainer.valid_step(unet_number = unet, max_batch_size = max_batch_size)
            # 输出验证损失值
            print(f'valid loss: {valid_loss}')

        # 如果当前 epoch 需要进行采样，并且训练器是主进程，则执行采样并保存图像
        if not (i % save_at_every) and i > 0 and trainer.is_main and should_sample:
            # 执行采样，返回生成的图像
            images = trainer.sample(texts = [sample_texts], batch_size = 1, return_pil_images = True, stop_at_unet_number = unet)
            # 保存生成的图像
            images[0].save(f'./sample-{i // 100}.png')

        # 如果当前 epoch 需要保存模型，并且训练器是主进程，则保存模型
        if not (i % save_at_every) and i > 0 and trainer.is_main and should_save:
            trainer.save(model_path)

    # 训练完成后保存模型
    trainer.save(model_path)
