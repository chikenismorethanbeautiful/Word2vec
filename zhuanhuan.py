import csv

input_file = "男科5-13000.csv"
output_file = "男科5-13000.txt"
total_words = 0

# 常见中文编码列表
encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8-sig', 'big5']

fin = None
for enc in encodings:
    try:
        fin = open(input_file, 'r', encoding=enc)
        # 测试读取一行
        fin.readline()
        fin.seek(0)  # 回到文件开头
        print(f"成功使用编码：{enc}")
        break
    except UnicodeDecodeError:
        continue

if fin is None:
    print("无法确定文件编码，请手动检查。")
    exit(1)

with fin, open(output_file, 'w', encoding='utf-8') as fout:
    reader = csv.reader(fin)
    for row in reader:
        line = ' '.join(row)
        fout.write(line + '\n')
        words = line.split()
        total_words += len(words)

print(f"转换完成！输出文件：{output_file}")
print(f"总单词数：{total_words}")