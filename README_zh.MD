# Local Knowledge Graph

![Example](example.png)

这个应用程序使用本地 Llama 模型来回答查询、构建嵌入和创建知识图谱，用于探索相关问题和答案。

## 描述

Local Knowledge Graph 是一个基于 Flask 的 Web 应用程序，它利用本地 Llama 语言模型来处理用户查询、生成逐步推理过程，并将思考过程可视化为交互式知识图谱。它还可以根据语义相似性找到并显示相关的问题和答案。

## 特性

- 用于提交查询的交互式 Web 界面
- 实时显示的逐步推理过程
- 推理步骤的动态知识图谱可视化
- 计算并显示最强推理路径
- 基于语义相似性的相关问题和答案
- 使用 Llama 语言模型进行本地处理

## 使用方法

1. 确保已安装所有必需的依赖项。
2. 通过运行 `app.py` 启动 Flask 应用程序。
3. 打开 Web 浏览器并导航到 `http://127.0.0.1:5100`（如果修改了端口，请使用适当的端口）。
4. 在输入字段中输入您的查询，然后点击"Submit"。
5. 观察应用程序如何生成逐步推理过程，实时更新知识图谱。
6. 查看最终答案和最强推理路径。
7. 探索显示在主要响应下方的相关问题和答案。

## 要求

- Python 3.7+
- Flask
- NumPy
- Chroma
- scikit-learn
- NetworkX
- 在 `http://localhost:11434` 上运行的本地 Llama 语言模型（例如 llama3.1:8b）

## 安装

1. 克隆此仓库。
2. 使用 requirements.txt 文件安装所需的 Python 包：
   ```
   pip install -r requirements.txt
   ```
3. 确保您有一个正在运行且可访问的本地 Llama 模型。
4. 运行 Flask 应用程序：
   ```
   python main.py
   ```

## 注意

此应用程序需要一个正在运行且可访问的本地 Llama 语言模型。在使用此应用程序之前，请确保设置了Ollama， 具体请参考[Ollama 安装指南](https://ollama.com/docs/guide/install)。

