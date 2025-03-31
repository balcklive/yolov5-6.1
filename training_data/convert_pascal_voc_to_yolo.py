import xml.etree.ElementTree as ET
import os
from pathlib import Path
import logging
from tqdm import tqdm

# 设置常量
CLASS_MAPPING = {
    'fire': 0,
    'smoke': 1
}

# 设置日志
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('conversion.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def convert_voc_to_yolo(xml_file):
    try:
        # 解析XML文件
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        # 存储转换后的标注
        yolo_annotations = []
        
        # 处理每个目标
        for obj in root.findall('object'):
            # 获取类别
            class_name = obj.find('name').text
            if class_name not in CLASS_MAPPING:
                logger.warning(f"未知类别 '{class_name}' 在文件 {xml_file} 中被跳过")
                continue
                
            class_id = CLASS_MAPPING[class_name]
            
            # 获取边界框坐标
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 转换为YOLO格式
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            
            # 确保值在0-1范围内
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            bbox_width = min(max(bbox_width, 0), 1)
            bbox_height = min(max(bbox_height, 0), 1)
            
            # 添加到结果列表
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        
        return yolo_annotations
    except Exception as e:
        logger.error(f"处理文件 {xml_file} 时发生错误: {str(e)}")
        return None

def process_folder(input_folder, output_folder=None):
    """
    处理指定文件夹中的所有XML文件
    
    Args:
        input_folder: 输入文件夹路径，包含XML文件
        output_folder: 输出文件夹路径，如果为None则使用输入文件夹
    """
    # 创建输出文件夹
    if output_folder is None:
        output_folder = input_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 获取所有XML文件
    xml_files = list(Path(input_folder).glob('*.xml'))
    
    if not xml_files:
        logger.warning(f"在 {input_folder} 中没有找到XML文件")
        return
    
    # 统计信息
    total_files = len(xml_files)
    success_count = 0
    error_count = 0
    
    logger.info(f"开始处理 {total_files} 个XML文件...")
    
    # 使用tqdm显示进度条
    for xml_file in tqdm(xml_files, desc="转换进度"):
        try:
            # 生成输出文件路径
            txt_file = Path(output_folder) / f"{xml_file.stem}.txt"
            
            # 转换标注
            yolo_annotations = convert_voc_to_yolo(xml_file)
            
            if yolo_annotations is not None:
                # 写入文件
                with open(txt_file, 'w', encoding='utf-8') as f:
                    for annotation in yolo_annotations:
                        f.write(annotation + '\n')
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
            logger.error(f"处理文件 {xml_file} 时发生错误: {str(e)}")
    
    # 输出统计信息
    logger.info(f"""
转换完成！统计信息：
- 总文件数: {total_files}
- 成功转换: {success_count}
- 转换失败: {error_count}
""")

if __name__ == "__main__":
    # 设置日志
    logger = setup_logger()
    
    # 设置输入输出路径
    input_folder = "/root/yolov5-6.1/training_data/images/fire_smoke_dataset4c4/Annotations"  # 替换为你的输入文件夹路径
    output_folder = "/root/yolov5-6.1/training_data/images/fire_smoke_dataset4c4/Annotations"      # 替换为你的输出文件夹路径
    
    try:
        process_folder(input_folder, output_folder)
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
