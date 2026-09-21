# 三资产自动赎回票据：RQMC 与布朗桥条件化

本项目研究三资产 worst-of autocallable 的定价、公平票息与 Greeks，比较标准
Monte Carlo、随机化准 Monte Carlo（scrambled Sobol）、布朗桥条件化及其组合。

项目源于 Imperial College Business School 的 MSc Risk Management and
Financial Engineering Applied Project。作者：Evelyn Wang。

## 快速运行

在项目根目录打开终端，使用 Python 3.11 或以上版本：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,notebooks]"
python examples/quickstart.py
python -m pytest -q
python scripts/verify_results.py
```

示例使用合成参数，可直接在 CPU 上运行，不需要 Bloomberg 工作簿或显卡。
结果保存在 `outputs/demo/`。示例预算很小，作用是演示调用方式与检查恒等式，
不能拿它的数值替代论文结果，也不能据此排列四种估计量的优劣。

## 阅读顺序

1. [英文首页](../README.md)：研究问题、方法比较、核心结果。
2. [研究报告](../report/project-report.pdf)：完整论文，已移除学号。
3. [复现说明](reproducibility.md)：环境、数据依赖及实验边界。
4. `notebooks/`：Day 1 至 Day 9 的十份 Notebook。
5. `results/`：保存的逐次实验记录、汇总表和图。

此仓库不包含 Bloomberg 原始工作簿。完整市场参数实验需要自行在 `Data/`
放入有权使用的冻结工作簿；文件名及 SHA-256 见 [数据说明](../Data/README.md)。

核心结论具有条件性：M3 在高精度定价及接近敲入障碍时有优势，但无法普遍消除
离散自动赎回触发带来的 Greek 不稳定性。Gamma 的严格稳定性检查通过比例仅为
12.5%。最终公平票息约为 9.685%，属于特定模型与研究合同的结果。

本次整理已跑通 25 项原有测试和四方法示例，并核对 1,112 次已保存实验；没有
重新执行整套历史实验。Notebook 的输出已清空，原有结果另存在 `results/`。
