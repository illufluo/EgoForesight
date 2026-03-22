# Action Prediction Project — 当前进度同步

> 本文件用于与网页端 Claude 同步项目进度。最后更新：2026-03-21

## 项目目标

基于 VLM（GLM-4.6V）的第一人称视频动作预测系统，使用 Ego4D 数据集。系统分析视频帧，用自然语言预测下一个动作。

## 版本规划

| 版本 | 输入 | 历史 | 微调 | 状态 |
|------|------|------|------|------|
| V1 | 单帧 | 无 | 无 | ✅ 已完成 |
| V2 | 4帧（2s窗口） | 无 | 无 | ✅ 已完成（prompt 已按 output_standard 优化） |
| V3 | 4帧 | 滑动窗口历史（3步） | 无 | ✅ 已完成并测试通过 |
| V4 | 4帧 | 无 | 是 | ❌ 未开始 |
| V5 | 4帧 | 是 | 是 | ❌ 未开始 |

## 当前目录结构

```
sdp/
├── shared/                    # ✅ 公共模块
│   ├── video_frames.py        #   视频帧提取（默认0.2s间隔，run.py调用时传0.5s）
│   ├── glm_client.py          #   VLM API 客户端（zai-sdk, glm-4.6v, 多图, 重试1次）
│   └── utils.py               #   工具函数（JSON保存、narration CSV加载与清洗）
│
├── v1/                        # ✅ V1 单帧预测
│   ├── prompt.py              #   单帧 prompt（要求30-50词预测）
│   └── run.py                 #   入口：提取帧 → 逐帧调用VLM → 保存JSON
│
├── v2/                        # ✅ V2 四帧窗口预测（prompt 已优化）
│   ├── prompt.py              #   四帧 prompt，含 few-shot 示例，不写死秒数
│   └── run.py                 #   入口：提取帧 → 4帧窗口 → 调用VLM → 保存JSON
│
├── v3/                        # ✅ V3 四帧窗口 + 历史上下文
│   ├── history.py             #   HistoryManager：滑动窗口保存最近3步 e/p 全文
│   ├── prompt.py              #   V3 prompt = V2 base prompt + history section
│   └── run.py                 #   入口：同V2流程，额外维护历史并注入prompt
│
├── data/
│   ├── videos/                #   原始视频（已有 test.mp4）
│   ├── frames/                #   提取的帧（按视频名分子目录）
│   ├── narrations/            #   Ego4D narration CSV
│   └── results/               #   输出JSON（已有 test_v1/v2/v3.json）
│
├── output_standard.md         # explanation/prediction 输出质量标准
├── project_overview.md        # 项目总览文档
├── claude_code_task_v1v2.md   # V1/V2 原始任务描述
├── claude_code_tasks.md       # Task 1 (V2 prompt改进) 任务描述
├── claude_code_task2_v3.md    # Task 2 (V3开发) 任务描述
├── glm46v_interface.py        # 旧版 GLM 接口（参考用，已替代）
└── video_frames.py            # 旧版帧提取（参考用，已复制到 shared/）
```

## 各模块详情

### shared/glm_client.py
- 函数签名：`call_vlm(images: List[str], prompt: str) -> str`
- 使用 `zai-sdk`（`ZhipuAiClient`），模型 `glm-4.6v`，启用 thinking
- API key 从环境变量 `ZHIPUAI_API_KEY` 或 `ZAI_API_KEY` 读取（通过 dotenv）
- 图片先排列，prompt 在最后
- 失败自动重试1次，之后抛异常

### shared/utils.py
- `save_results(results, output_path)` — 保存 dict 为 JSON
- `load_narrations(csv_path)` — 加载 narration CSV，自动去除 `#C C` 前缀

### shared/video_frames.py
- `extract_frames(video_path, output_dir, interval=0.2)` — 按时间间隔提取帧
- `get_video_info(video_path)` — 获取视频基本信息

### v2/prompt.py（已优化）
- 不写死秒数/帧数，只说 "consecutive frames at regular intervals"
- 含 few-shot 示例稳定输出风格
- 符合 output_standard.md 要求：动词开头、具体物体、手部细节、时间进展

### v2/run.py, v3/run.py
- `_parse_response()` 使用 regex 解析，支持 Explanation/Prediction 标签顺序不固定
- 解析失败时 fallback 到 raw response 全文

### v3/history.py
- `HistoryManager` 类，`deque(maxlen=3)` 实现滑动窗口
- `add(explanation, prediction)` — 存入全文（不压缩）
- `get_history()` — 格式化为带时间标签的文本块：`[3 steps ago]`, `[2 steps ago]`, `[Previous step]`
- 最近的步骤排在最后（最接近当前上下文）

### v3/prompt.py
- `build_prompt(history=None)` — 有历史时追加 history section，无历史时等同 V2
- History section 明确告知模型：当前帧为主，历史仅为参考，不要复制历史内容，冲突时信任当前帧

## 使用方式

```bash
cd ~/Desktop/sdp
source .venv/bin/activate

# V1: 单帧预测
python -m v1.run --video data/videos/test.mp4 --output data/results/

# V2: 4帧窗口预测
python -m v2.run --video data/videos/test.mp4 --output data/results/

# V3: 4帧窗口 + 历史上下文
python -m v3.run --video data/videos/test.mp4 --output data/results/
```

## 已完成的迭代记录

1. **V1/V2 基础管线搭建** — shared 模块 + V1/V2 入口完成
2. **glm_client.py 重构** — 从 zhipuai SDK 迁移到 zai-sdk，模型升级到 glm-4.6v，启用 thinking
3. **V2 prompt 优化（Task 1）** — 按 output_standard.md 重写 prompt，加 few-shot 示例，修复 parse strip 问题
4. **V3 开发（Task 2）** — 新增 history.py + v3/prompt.py + v3/run.py，滑动窗口历史机制，测试通过

## 待完成事项

- [ ] V4：无历史 + 微调
- [ ] V5：历史 + 微调
- [ ] 数据标注增强（用 VLM 生成更丰富的标注，作为 V4/V5 训练数据和评估 ground truth）
- [ ] Prompt 持续迭代优化
- [ ] 在更多 Ego4D 视频上批量测试
