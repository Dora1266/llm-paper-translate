# DocTranslate: 多格式文档自动翻译系统

### 核心功能
- **多格式文档支持**: 一站式翻译PDF、Word(DOC/DOCX)和Excel(XLSX)文档
- **保留原始格式**: 精确保留原文档的字体、样式、表格结构和排版
- **智能语言检测**: 自动识别源文档语言，支持中、英、日、法、德、西等多语言互译
- **高质量翻译**: 集成大语言模型，确保翻译的专业性和准确性

### 技术架构
- **双服务器架构**:
  - 文档处理服务: 负责文档解析、翻译管理和格式重建
  - LLM推理服务: 提供高性能翻译能力
- **高并发处理**:
  - 线程池并行翻译多页文档
  - 队列管理和任务调度机制
  - 批处理优化，减少API调用次数
- **性能优化**:
  - 翻译结果缓存机制，避免重复翻译
  - 分块处理大型文档，优化内存使用
  - 自适应批量处理，平衡吞吐量和响应时间

### API设计
- **RESTful API**: 标准化接口设计，便于集成
- **异步处理模式**: 上传文件后立即返回，后台处理完成后通知
- **实时进度跟踪**: 提供任务状态查询接口，实时监控翻译进度
- **OpenAI兼容接口**: LLM服务支持OpenAI风格的请求格式

## 部署注意事项

### 系统要求
- Python 3.10+
- 至少8GB内存(推荐16GB+，特别是处理大型PDF)
- 6GB磁盘空间用于应用和临时文件存储
- NVIDIA GPU(可选，加速LLM推理)

### 依赖安装
```bash
# 安装主要依赖
pip install flask requests PyMuPDF python-docx openpyxl mammoth

# 安装LLM服务依赖
pip install lmdeploy waitress
```

### 文件目录设置
```bash
mkdir -p uploads translated
chmod 755 uploads translated
```

### 环境配置
1. **文档处理服务配置**
   - 设置API_URL指向LLM服务地址
   - 配置上传文件大小限制
   - 配置语言映射关系

2. **LLM服务配置**
   - 设置模型路径
   - 配置工作线程数和批处理大小
   - 调整内存使用和会话长度

## 部署教程

### 步骤1: 准备环境
```bash
# 创建并激活虚拟环境
python -m venv doctranslate-env
source doctranslate-env/bin/activate  # Linux/Mac
# 或
doctranslate-env\Scripts\activate  # Windows

# 安装依赖
pip install flask requests PyMuPDF python-docx openpyxl mammoth lmdeploy waitress
```

### 步骤2: 部署LLM推理服务
```bash
# 编辑LLM服务配置
nano paste-2.txt  # 修改model_path指向你的模型目录

# 运行LLM服务
python paste-2.txt
```

成功启动后，您将看到:
```
正在加载模型: D:\tran\models\DeepSeek-R1-Distill-Qwen-1.5B...
模型加载完成！服务器准备就绪。
启动工作线程 1/10
...
启动Flask服务器在端口 8000...
```

### 步骤3: 部署文档处理服务
```bash
# 创建必要的目录
mkdir -p uploads translated

# 编辑文档处理服务配置
nano paste.txt  # 确认API_URL设置为http://localhost:8000/v1/chat/completions

# 运行文档处理服务
python paste.txt
```

成功启动后，您将看到:
```
🔄 测试翻译API连接...
✅ 翻译API连接成功
 * Running on http://127.0.0.1:5000
```

### 步骤4: 测试系统
1. **API调用测试**
```bash
curl -X POST http://localhost:5000/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "file_url": "https://example.com/sample.pdf",
    "callback_url": "http://localhost:8080/callback",
    "target_language": "zh",
    "request_id": "test-001"
  }'
```

2. **查询翻译状态**
```bash
curl http://localhost:5000/api/status?request_id=test-001
```

## 使用指南

### 通过API集成
文档处理服务提供以下API端点:

1. **开始翻译**
   - POST `/api/translate`
   - 参数:
     - `file_url`: 文档URL地址
     - `target_language`: 目标语言(en/zh/ja/fr/de/es)
     - `callback_url`: 处理完成后的回调地址
     - `request_id`(可选): 自定义请求ID

2. **查询状态**
   - GET `/api/status?request_id={request_id}`
   - 返回当前翻译进度和状态

3. **取消任务**
   - POST `/api/cancel`
   - 参数: `request_id`

### 通过函数调用
可以直接在Python代码中调用翻译功能:

```python
from paste import translate

# 参数1: 本地文件路径或URL
# 参数2: 目标语言代码
result_path = translate("path/to/document.pdf", "zh")
# 或
result_path = translate("https://example.com/document.docx", "en")

if result_path:
    print(f"翻译成功，结果保存在: {result_path}")
else:
    print("翻译失败")
```

函数返回值:
- 翻译成功: 返回翻译后文件的保存路径
- 翻译失败: 返回None

### 高级配置

#### 文档处理服务
```python
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 增加最大文件大小至32MB
app.config['API_KEY'] = "your_api_key"  # 添加API密钥
```

#### LLM服务
```python
CONFIG = {
    "max_workers": 20,  # 增加工作线程
    "max_batch_size": 20,  # 增加批处理大小
    "session_len": 4096,  # 增加会话长度
    "temperature": 0.3,  # 调低温度提高翻译准确性
}
```

## 故障排除

### 常见问题
1. **翻译服务无响应**
   - 检查LLM服务是否正常运行
   - 确认API_URL配置正确

2. **内存不足错误**
   - 减小批处理大小和并行线程数
   - 处理大型PDF时考虑分段处理

3. **翻译质量问题**
   - 调整temperature为较低值(0.1-0.3)
   - 为特定领域添加专业提示词

4. **文件格式保留问题**
   - 复杂表格和图表可能需要手动调整
   - PDF中的复杂排版可能不能完全保留

## 性能优化建议
- 使用SSD存储临时文件，加快IO操作
- 增加系统内存，特别是处理大型Excel或PDF文件时
- 启用结果缓存以提高重复翻译效率
- 对于高流量场景，考虑使用Redis进行分布式缓存

---

系统架构由两个独立服务组成，可根据需求单独扩展。文档处理服务负责格式处理的CPU密集型任务，而LLM服务可利用GPU加速翻译推理。
