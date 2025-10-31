#!/usr/bin/env python3
"""
重组 output 文件夹结构，使其更加清晰和规范。
"""
import shutil
from pathlib import Path
from typing import Dict, Tuple

# 路径映射：旧路径 -> 新路径
PATH_MAPPINGS: Dict[str, str] = {
    "output/01": "output/stage_01_descriptions",
    "output/sum": "output/stage_02_merged",
    "output/scene_graphs": "output/stage_03_scene_graphs",
    "output/scene_graphs_pytorch": "output/stage_04_pytorch",
    "output/predict": "output/predictions",
    # evaluation/ 保持不变
    # groundtruth/ 保持不变（但需要检查内部重复）
}

def move_directory_safe(old_path: Path, new_path: Path) -> bool:
    """安全地移动目录，如果目标已存在则合并"""
    if not old_path.exists():
        print(f"  ⚠️  源路径不存在: {old_path}")
        return False
    
    if old_path.resolve() == new_path.resolve():
        print(f"  ✓ 路径相同，跳过: {old_path}")
        return True
    
    new_path.parent.mkdir(parents=True, exist_ok=True)
    
    if new_path.exists():
        print(f"  ⚠️  目标已存在，尝试合并: {new_path}")
        # 合并目录内容
        for item in old_path.iterdir():
            dest_item = new_path / item.name
            if item.is_dir():
                if dest_item.exists():
                    # 递归合并目录
                    move_directory_safe(item, dest_item)
                else:
                    shutil.move(str(item), str(dest_item))
                    print(f"    ✓ 移动目录: {item.name}")
            else:
                if dest_item.exists():
                    print(f"    ⚠️  文件已存在，跳过: {item.name}")
                else:
                    shutil.move(str(item), str(dest_item))
                    print(f"    ✓ 移动文件: {item.name}")
        
        # 如果旧目录已为空，删除它
        try:
            old_path.rmdir()
            print(f"  ✓ 删除空目录: {old_path}")
        except OSError:
            print(f"  ⚠️  目录非空，保留: {old_path}")
    else:
        shutil.move(str(old_path), str(new_path))
        print(f"  ✓ 移动: {old_path} -> {new_path}")
    
    return True

def check_groundtruth_duplicates():
    """检查 groundtruth 目录中的重复路径"""
    groundtruth_dir = Path("output/groundtruth")
    scene_graphs_pytorch_root = Path("output/scene_graphs_pytorch")
    
    if groundtruth_dir.exists():
        gt_pytorch = groundtruth_dir / "scene_graphs_pytorch"
        if gt_pytorch.exists() and scene_graphs_pytorch_root.exists():
            print(f"\n⚠️  发现重复路径:")
            print(f"  - {gt_pytorch}")
            print(f"  - {scene_graphs_pytorch_root}")
            print(f"  建议: 如果 groundtruth 中的是子集，可以保留；否则需要手动处理")

def reorganize_output():
    """执行重组操作"""
    print("=" * 60)
    print("Output 文件夹重组工具")
    print("=" * 60)
    
    # 检查当前目录
    if not Path("output").exists():
        print("❌ 错误: 未找到 output 目录")
        return False
    
    print("\n📋 路径映射计划:")
    for old, new in PATH_MAPPINGS.items():
        old_path = Path(old)
        new_path = Path(new)
        status = "✓" if old_path.exists() else "○"
        print(f"  {status} {old} -> {new}")
    
    # 确认操作
    print("\n" + "=" * 60)
    response = input("是否继续执行重组？(yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ 操作已取消")
        return False
    
    print("\n🔄 开始重组...\n")
    
    # 执行移动操作
    success_count = 0
    for old_str, new_str in PATH_MAPPINGS.items():
        old_path = Path(old_str)
        new_path = Path(new_str)
        
        print(f"\n处理: {old_str}")
        if move_directory_safe(old_path, new_path):
            success_count += 1
    
    # 检查 groundtruth 中的重复
    print("\n" + "=" * 60)
    check_groundtruth_duplicates()
    
    print("\n" + "=" * 60)
    print(f"✅ 重组完成！成功处理 {success_count}/{len(PATH_MAPPINGS)} 个目录")
    print("=" * 60)
    print("\n⚠️  重要提示:")
    print("1. 请运行以下命令更新代码中的路径引用:")
    print("   python scripts/update_output_paths.py")
    print("2. 建议提交更改前先测试所有脚本")
    
    return True

if __name__ == "__main__":
    try:
        reorganize_output()
    except KeyboardInterrupt:
        print("\n\n❌ 操作被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

