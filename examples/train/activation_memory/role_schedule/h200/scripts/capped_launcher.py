# 人工分配器上限薄启动器（压力实验专用）
# 用法: python capped_launcher.py <cap_bytes> <profile_dmd2.py 的参数...>
# 在加载模型之前、同一进程内对逻辑设备 0 设上限，再执行原 profiler。
import runpy
import sys

import torch

cap_bytes = int(sys.argv.pop(1))
total = torch.cuda.get_device_properties(0).total_memory
frac = cap_bytes / total
torch.cuda.set_per_process_memory_fraction(frac, 0)
print(json_line := __import__('json').dumps({
    'launcher': 'capped_launcher',
    'cap_bytes_requested': cap_bytes,
    'total_visible_bytes': total,
    'fraction_set': frac,
}), flush=True)

sys.argv[0] = '/inspire/hdd/global_user/s26045/fastvideo/h200_stall_results/code_d7239500/examples/train/activation_memory/profile_dmd2.py'
sys.path.insert(0, str(__import__('pathlib').Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0], run_name='__main__')
