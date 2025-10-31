#!/usr/bin/env python3
"""
更新源代码中的 output 路径引用，匹配新的目录结构。
"""
import re
from pathlib import Path
from typing import List, Tuple

# 路径替换映射：旧路径 -> 新路径
REPLACEMENTS: List[Tuple[str, str, str]] = [
    # (文件路径模式, 旧路径, 新路径)
    # 注意：使用原始字符串避免转义问题
    ("*.py", r'"output/01/', r'"output/stage_01_descriptions/'),
    ("*.py", r"'output/01/", r"'output/stage_01_descriptions/"),
    ("*.py", r'output/01/', r'output/stage_01_descriptions/'),
    
    ("*.py", r'"output/sum/', r'"output/stage_02_merged/'),
    ("*.py", r"'output/sum/'", r"'output/stage_02_merged/'"),
    ("*.py", r'output/sum/', r'output/stage_02_merged/'),
    
    ("*.py", r'"output/scene_graphs"', r'"output/stage_03_scene_graphs"'),
    ("*.py", r"'output/scene_graphs'", r"'output/stage_03_scene_graphs'"),
    ("*.py", r'output/scene_graphs', r'output/stage_03_scene_graphs'),
    # 注意：scene_graphs 可能在路径中间，需要更精确的匹配
    
    ("*.py", r'"output/scene_graphs_pytorch"', r'"output/stage_04_pytorch"'),
    ("*.py", r"'output/scene_graphs_pytorch'", r"'output/stage_04_pytorch'"),
    ("*.py", r'output/scene_graphs_pytorch', r'output/stage_04_pytorch'),
    
    ("*.py", r'"output/predict"', r'"output/predictions"'),
    ("*.py", r"'output/predict'", r"'output/predictions'"),
    ("*.py", r'output/predict/', r'output/predictions/'),
    ("*.py", r'Path("output/predict")', r'Path("output/predictions")'),
    ("*.py", r"Path('output/predict')", r"Path('output/predictions')"),
]

def update_file_paths(file_path: Path) -> Tuple[int, List[str]]:
    """更新单个文件中的路径"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        changes = []
        
        for pattern, old_path, new_path in REPLACEMENTS:
            # 检查是否匹配文件模式
            if pattern.endswith('.py'):
                if not file_path.suffix == '.py':
                    continue
            
            # 执行替换
            new_content = content.replace(old_path, new_path)
            if new_content != content:
                count = content.count(old_path)
                changes.append(f"  - {old_path} -> {new_path} ({count} 处)")
                content = new_content
        
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return len(changes), changes
        
        return 0, []
    except Exception as e:
        print(f"  ❌ 处理文件时出错: {e}")
        return 0, []

def update_all_source_files():
    """更新所有源代码文件"""
    print("=" * 60)
    print("更新源代码中的 output 路径引用")
    print("=" * 60)
    
    src_dir = Path("src")
    if not src_dir.exists():
        print("❌ 错误: 未找到 src 目录")
        return False
    
    # 获取所有 Python 文件
    python_files = list(src_dir.glob("*.py"))
    
    print(f"\n📋 找到 {len(python_files)} 个 Python 文件\n")
    
    total_changes = 0
    updated_files = []
    
    for py_file in python_files:
        print(f"处理: {py_file.name}")
        change_count, changes = update_file_paths(py_file)
        
        if change_count > 0:
            total_changes += change_count
            updated_files.append(py_file.name)
            for change in changes:
                print(change)
            print(f"  ✅ 已更新\n")
        else:
            print(f"  ○ 无需更新\n")
    
    print("=" * 60)
    print(f"✅ 更新完成！")
    print(f"  - 处理文件数: {len(python_files)}")
    print(f"  - 更新文件数: {len(updated_files)}")
    print(f"  - 总更改数: {total_changes}")
    print("=" * 60)
    
    if updated_files:
        print("\n更新的文件:")
        for fname in updated_files:
            print(f"  - {fname}")
    
    # 还需要处理一些特殊情况
    print("\n⚠️  请注意以下特殊情况:")
    print("1. 检查 src/04_convert_to_pytorch.py 中的路径列表")
    print("2. 检查 src/09_reasoning.py 中的 GROUNDTRUTH_SCENE_GRAPHS_PYTORCH_DIR")
    print("3. 检查所有脚本中的命令行参数默认值")
    
    return True

if __name__ == "__main__":
    try:
        update_all_source_files()
    except KeyboardInterrupt:
        print("\n\n❌ 操作被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

