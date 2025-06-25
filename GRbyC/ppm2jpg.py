from PIL import Image
import os

def convert_ppm_to_jpg(ppm_file, output_dir):
    # 打开 PPM 文件
    with Image.open(ppm_file) as img:
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 构建输出文件路径
        base_name = os.path.basename(ppm_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{name_without_ext}.jpg")
        
        # 转换并保存为 JPG 格式
        img.convert('RGB').save(output_file, 'JPEG')
        print(f"Converted {ppm_file} to {output_file}")

# 示例用法
ppm_file_path = 'example.ppm'
output_directory = 'output_images'
convert_ppm_to_jpg(ppm_file_path, output_directory)
