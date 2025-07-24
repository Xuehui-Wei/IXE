# -*- coding: utf-8 -*-
# ============= 子模块导入 =============
from IXE import image_processing
from IXE import iad_controls
from IXE import spectrum_controls
from IXE import remove_cross
from IXE import spectrum_utils

# ============= 显式导出核心类 =============
from IXE.spectrum_utils import SpectrumProcessor
from IXE.xes_analyzer import TIFFAnalyzer
from IXE.remove_cross import delete_zero_rows_and_columns


# ============= 定义公开接口 =============
__all__ = [
    'TIFFAnalyzer',        # GUI主类
    'SpectrumProcessor',   # 光谱处理工具类
    'image_processing',    # 图像处理模块
    'iad_controls',        # IAD控制模块
    'spectrum_controls',   # 光谱控制模块
    'remove_cross',        # 十字去除工具
    'spectrum_utils'       # 光谱工具模块
    'delete_zero_rows_and_columns',  # 删除零行列的函数
]
