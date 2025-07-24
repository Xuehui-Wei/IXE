import sys
from pathlib import Path

# 强制添加项目根目录到PATH
sys.path.insert(0, str(Path(__file__).parent))

try:
    from IXE.remove_cross import delete_zero_rows_and_columns
    print("✅ 导入成功！")
    print("函数信息：", delete_zero_rows_and_columns.__doc__)
except Exception as e:
    print("❌ 导入失败：")
    print(type(e).__name__, ":", e)
    print("当前sys.path：")
    print("\n".join(sys.path))