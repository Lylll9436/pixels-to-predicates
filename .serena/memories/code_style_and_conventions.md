# 代码风格和约定

## 编码规范
- **文件编码**：UTF-8
- **脚本头部**：所有 Python 脚本以 `#!/usr/bin/env python3` 开头
- **文档字符串**：使用中文编写文档字符串（docstrings）

## 命名约定
- **文件名**：使用数字前缀+下划线+描述性名称（如 `01_describe_pic.py`）
- **变量名**：使用 snake_case
- **类名**：使用 PascalCase
- **常量**：使用 UPPER_SNAKE_CASE

## 代码组织
- **类型提示**：使用 `typing` 模块进行类型注解
- **数据结构**：使用 `dataclass` 定义数据类（如 Entity, Relation）
- **导入顺序**：标准库 → 第三方库 → 本地模块

## 注释和文档
- **注释语言**：中文
- **文档字符串**：使用中文，包含 Args 和说明
- **行内注释**：使用中文

## 代码示例风格
```python
#!/usr/bin/env python3
"""
脚本功能描述
"""

from typing import List, Optional
from pathlib import Path

class MyClass:
    """类描述"""
    
    def __init__(self, param: str) -> None:
        """初始化
        
        Args:
            param: 参数描述
        """
        self.param = param
```

## 可视化风格
- 使用 Nature 期刊风格配色方案
- 配色：`['#1f4e5f', '#326273', '#488a99', '#6ba292', '#a3c9a8']`
- 强调色：`'#d1495b'`
- 背景色：`'#f4f4f2'`
- 字体：serif 字体（Times New Roman, Times 等）