import re

def extract_boxed_latex(s):
    pattern = r'\\boxed{(.*)}'
    matches = re.findall(pattern, s)
    result = matches
    return result

# 示例字符串
input_string = r"这是一个例子：\boxed{\frac{1}{2}}aaaaa"+'\n'+"}"

# 提取内容
extracted_contents = extract_boxed_latex(input_string)

# 打印结果
print(extracted_contents)
