def calculate(equation_string):
    """
    接收算式字符串（如 '1+2'），返回计算结果
    """
    try:
        # 注意: eval在生产环境中不安全，建议自行实现解析逻辑
        result = eval(equation_string)
        return result
    except Exception as e:
        return f"Error: {str(e)}"
