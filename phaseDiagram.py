import os
import re
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

def extract_parameters(filename):
    """
    从文件名中提取AroundNumMiniKBT和VisionConeXLen参数值
    """
    # 使用正则表达式匹配参数值
    # around_match = re.search(r'AroundRhoMiniKBT(-?[\d.]+)', filename)
    around_match = re.search(r'particles(-?[\d.]+)', filename)
    vision_match = re.search(r'VisionConeXLen(-?[\d.]+)', filename)
    
    if around_match and vision_match:
        around_value = float(around_match.group(1))
        vision_value = float(vision_match.group(1))
        return around_value, vision_value
    else:
        return None, None

def create_phase_diagram(folder_path, output_filename="phase_diagram.jpg"):
    """
    创建相图
    
    Args:
        folder_path: 包含图片的文件夹路径
        output_filename: 输出文件名
    """
    # 获取所有jpg文件
    pattern = os.path.join(folder_path, "*.jpg")
    image_files = glob.glob(pattern)
    
    if not image_files:
        print("在指定文件夹中未找到jpg文件")
        return
    
    # 存储图片和对应的参数值
    image_data = []
    
    for file_path in image_files:
        filename = os.path.basename(file_path)
        around_value, vision_value = extract_parameters(filename)
        
        if around_value is not None and vision_value is not None:
            image_data.append({
                'file_path': file_path,
                'around_value': around_value,
                'vision_value': vision_value,
                'filename': filename
            })
    
    if not image_data:
        print("未找到包含所需参数的文件")
        return
    
    # 获取唯一的参数值并排序
    around_values = sorted(set(data['around_value'] for data in image_data))
    vision_values = sorted(set(data['vision_value'] for data in image_data))
    
    print(f"找到 {len(image_data)} 个有效文件")
    print(f"AroundNumMiniKBT 值: {around_values}")
    print(f"VisionConeXLen 值: {vision_values}")
    
    # 计算网格大小
    rows = len(vision_values)
    cols = len(around_values)
    
    # 读取第一张图片获取尺寸
    sample_img = Image.open(image_data[0]['file_path'])
    img_width, img_height = sample_img.size
    
    # 创建大画布
    canvas_width = cols * img_width
    canvas_height = rows * img_height
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    # 创建参数值到网格位置的映射
    around_to_col = {val: idx for idx, val in enumerate(around_values)}
    vision_to_row = {val: idx for idx, val in enumerate(vision_values)}
    
    # 将图片粘贴到对应位置
    for data in image_data:
        around_val = data['around_value']
        vision_val = data['vision_value']
        
        col = around_to_col[around_val]
        row = vision_to_row[vision_val]
        
        # VisionConeXLen作为纵坐标，所以需要反转行索引（从上到下）
        row = rows - 1 - row
        
        x = col * img_width
        y = row * img_height
        
        try:
            img = Image.open(data['file_path'])
            canvas.paste(img, (x, y))
            print(f"已添加: {data['filename']} -> 位置({col}, {row})")
        except Exception as e:
            print(f"无法加载图片 {data['filename']}: {e}")
    
    # 保存结果
    canvas.save(output_filename, quality=95)
    print(f"\n相图已保存为: {output_filename}")
    
    # 可选：创建带有坐标轴的版本（使用matplotlib）
    create_annotated_phase_diagram(image_data, around_values, vision_values, 
                                 img_width, img_height, output_filename.replace('.jpg', '_annotated.jpg'))

def create_annotated_phase_diagram(image_data, around_values, vision_values, 
                                 img_width, img_height, output_filename):
    """
    创建带有坐标轴标签的相图版本
    """
    try:
        rows = len(vision_values)
        cols = len(around_values)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        
        # 如果只有一行或一列，确保axes是二维数组
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 创建位置映射
        around_to_col = {val: idx for idx, val in enumerate(around_values)}
        vision_to_row = {val: idx for idx, val in enumerate(vision_values)}
        
        # 初始化所有子图为空白
        for i in range(rows):
            for j in range(cols):
                axes[i, j].axis('off')
        
        # 填充有图片的子图
        for data in image_data:
            around_val = data['around_value']
            vision_val = data['vision_value']
            
            col = around_to_col[around_val]
            row = vision_to_row[vision_val]
            
            # matplotlib的坐标是从上到下的
            try:
                img = plt.imread(data['file_path'])
                axes[row, col].imshow(img)
                axes[row, col].set_title(f'A:{around_val}\nV:{vision_val}', 
                                       fontsize=8, pad=2)
                axes[row, col].axis('on')
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            except Exception as e:
                print(f"无法在标注版本中添加 {data['filename']}: {e}")
        
        # 设置行列标签
        for i, vision_val in enumerate(vision_values):
            axes[i, 0].set_ylabel(f'VisionConeXLen\n{vision_val}', rotation=0, 
                                ha='right', va='center', fontsize=10)
        
        for j, around_val in enumerate(around_values):
            axes[0, j].set_title(f'AroundNumMiniKBT\n{around_val}', fontsize=10, pad=10)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"带标注的相图已保存为: {output_filename}")
        
    except Exception as e:
        print(f"创建带标注版本时出错: {e}")

if __name__ == "__main__":
    # 获取用户输入
    #folder_path = input("请输入包含图片的文件夹路径: ").strip()
    folder_path=sys.argv[1]
    
    # 移除可能的引号
    folder_path = folder_path.strip('"\'')
    
    if not os.path.exists(folder_path):
        print("文件夹不存在，请检查路径")
    else:
        output_name = input("请输入输出文件名（默认为 phase_diagram.jpg）: ").strip()
        if not output_name:
            output_name = "phase_diagram.jpg"
        elif not output_name.lower().endswith('.jpg'):
            output_name += '.jpg'
            
        create_phase_diagram(folder_path, output_name)