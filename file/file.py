import os

input_file = 'huge_app.log'
output_file = 'error_report.txt'

# 检查文件是否存在
if not os.path.exists(input_file):
    print(f"错误：文件 {input_file} 不存在！")
else:
    error_lines = []
    # 1. 流式读取源文件
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            if 'ERROR' in line:
                error_lines.append(line)

    # 2. 写入报告文件
    if error_lines:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.writelines(error_lines)
        print(f"分析完成，错误报告已保存至 {output_file}")
    else:
        print("未发现任何错误。")