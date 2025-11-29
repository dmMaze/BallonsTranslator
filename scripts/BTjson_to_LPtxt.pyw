import json
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from PIL import Image

# 全局变量，记录是否对所有图片使用同一尺寸
GLOBAL_PAGE_SIZE = None

def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        return None

def prompt_for_size(image_name):
    global GLOBAL_PAGE_SIZE
    root = tk.Tk()
    root.withdraw()
    try:
        width = simpledialog.askinteger("输入图片宽度", f"无法读取图片 {image_name} 的尺寸，请输入宽度：", minvalue=1)
        height = simpledialog.askinteger("输入图片高度", f"无法读取图片 {image_name} 的尺寸，请输入高度：", minvalue=1)
        if width and height:
            # 询问是否应用于所有图片
            use_for_all = messagebox.askyesno("应用到所有图片", f"是否将此尺寸({width}x{height})应用于所有图片？")
            if use_for_all:
                GLOBAL_PAGE_SIZE = (width, height)
            return width, height
    except Exception:
        pass
    messagebox.showwarning("警告", f"输入无效，将使用默认尺寸 822*1200")
    return 822, 1200  # 修改为正确的默认尺寸

def get_image_info(json_data, image_name):
    # 优先从JSON的image_info读取
    image_info = json_data.get("image_info", {})
    if image_name in image_info:
        width = image_info[image_name].get("width")
        height = image_info[image_name].get("height")
        if width and height:
            return width, height

    # 如果没有在image_info中找到，则尝试读取同名图像文件
    img_dir = os.path.dirname(json_file_path)
    img_base = os.path.splitext(image_name)[0]
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        candidate = os.path.join(img_dir, img_base + ext)
        if os.path.isfile(candidate):
            size = get_image_size(candidate)
            if size:
                return size

    # 如果没有图像文件，则提示用户输入
    size = prompt_for_size(image_name)
    return size

def extract_text_and_translation(json_file, include_font_info=False):
    global GLOBAL_PAGE_SIZE
    global json_file_path  # 保存json文件路径供其他函数使用
    json_file_path = json_file
    
    # 打开并加载 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取保存 TXT 文件的目录
    output_dir = os.path.dirname(json_file)
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历页面
    output_text = []  # 用于存储输出的内容
    for image_name, items in data.get("pages", {}).items():
        # 获取图像尺寸
        PAGE_WIDTH, PAGE_HEIGHT = get_image_info(data, image_name)

        # 写入页面标题
        output_text.append(f">>>>>>>>[{image_name}]<<<<<<<<\n")
        
        # 遍历每一条文本项，获取坐标和翻译
        for idx, item in enumerate(items):
            if "text" in item and "translation" in item:
                # 获取左上角的坐标（xyxy 中的前两个值）
                coord = item.get("xyxy", [])
                if len(coord) >= 2:
                    coord = coord[:2]  # 只保留前两个坐标
                    # 将坐标转化为比例
                    coord = [coord[0] / PAGE_WIDTH, coord[1] / PAGE_HEIGHT, 1]
                
                text = item["text"]
                translation = item["translation"]
                
                # 检查是否需要添加字体信息
                if include_font_info:
                    font_name = item.get("_detected_font_name", "")
                    font_size = item.get("_detected_font_size", "")*72/96 # 像素转pt
                    if font_name or font_size:
                        font_info = ""
                        if font_name:
                            font_info += f"{{字体：{font_name}}}"
                        if font_size:
                            font_info += f"{{字号：{font_size}}}"
                        translation = font_info + translation
                
                # 添加格式化的文本到输出列表
                output_text.append(f"----------------[{idx+1}]----------------{coord}\n")
                output_text.append(f"{translation}\n")
    
    # 获取输出的 TXT 文件路径
    json_base = os.path.splitext(os.path.basename(json_file))[0]
    output_file = os.path.join(output_dir, f"{json_base}_translations.txt")
    
    # 将内容写入文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("1,0\n-\n框内\n框外\n-\n备注备注备注\n")
        f_out.writelines(output_text)
    
    print(f"翻译文本已保存到: {output_file}")
    # 弹窗提示
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("输出完成", f"翻译文本已成功输出到：\n{output_file}")

def choose_json_file():
    # 使用 tkinter 打开文件选择对话框
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    json_file = filedialog.askopenfilename(
        title="选择 JSON 文件",
        filetypes=[("JSON Files", "*.json")]
    )
    return json_file

def main():
    # 创建简单的GUI界面
    root = tk.Tk()
    root.title("气泡翻译器翻译转lptxt")
    root.geometry("400x200")
    
    # 添加复选框
    font_info_var = tk.BooleanVar()
    font_info_checkbox = tk.Checkbutton(root, text="导出时包含字体字号信息", variable=font_info_var)
    font_info_checkbox.pack(pady=20)
    
    # 添加处理按钮
    def process_json():
        # 用户选择 JSON 文件
        json_file = choose_json_file()
        if not json_file:
            messagebox.showwarning("警告", "未选择文件")
            return
        
        # 提取文本和翻译并保存到 txt 文件
        include_font_info = font_info_var.get()
        try:
            extract_text_and_translation(json_file, include_font_info)
        except Exception as e:
            messagebox.showerror("错误", f"处理文件时出错：{str(e)}")
        # 处理完成后不关闭窗口，继续等待下一次选择
    
    process_button = tk.Button(root, text="选择JSON文件并处理", command=process_json)
    process_button.pack(pady=10)
    
    # 添加退出按钮
    exit_button = tk.Button(root, text="退出", command=root.destroy)
    exit_button.pack(pady=10)
    
    # 运行GUI主循环
    root.mainloop()

if __name__ == "__main__":
    main()
