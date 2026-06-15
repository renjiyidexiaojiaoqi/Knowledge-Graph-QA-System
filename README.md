本项目已部署到云端点击链接进入https://knowledge-graph-app-system-e4htkoeckcojwxnlnru6vy.streamlit.app/

WikiMultiHop 知识图谱分析平台
一个基于 Neo4j 和 MongoDB 的多跳推理知识图谱可视化分析平台，支持问题检索、实体关系探索、知识图谱可视化和问题聚类分析。
功能特性
-  **问题检索与查询** - 支持关键词搜索问题，查看完整上下文和答案
-  **实体关系探索** - 查看实体关联网络，发现实体之间的关系强度
-  **知识图谱可视化** - 交互式网络图展示实体关系
-  **问题聚类分析** - 基于 TF-IDF 的问题自动聚类
-  **数据统计仪表板** - 问题类型分布、热门实体统计
-  **批量数据导出** - 支持 JSON 格式导出问题数据

##  安装与部署

### 本地运行

1. 克隆仓库
```bash
git clone https://github.com/your-username/knowledge-graph-qa-system.git
cd knowledge-graph-qa-system
安装依赖

bash
pip install -r requirements.txt

运行应用

bash
streamlit run app.py


项目结构
text
knowledge-graph-qa-system/
├── app.py                  # 主程序
├── requirements.txt        # Python 依赖
├── .streamlit/
│   ├── config.toml        # Streamlit 配置
│   └── secrets.toml       # 数据库密钥（本地使用）
└── README.md              # 项目文档
数据模型
Neo4j 图结构
Question - 问题节点

Entity - 实体节点

MENTIONS - 问题与实体的关系

MongoDB 文档结构
json
{
  "_id": "问题ID",
  "question": "问题文本",
  "answer": "答案",
  "type": "问题类型",
  "context": "上下文文档",
  "supporting_facts": "支持事实"
}
🚀 使用指南
1. 问题检索
输入关键词（如 "Who", "What"）

选择搜索结果中的问题

查看问题详情、答案和关联实体

2. 实体关系探索
查看热门实体推荐

搜索特定实体

查看实体的关联网络和相关问题

3. 知识图谱可视化
按问题ID查看关联实体

按实体名称查看关系网络

4. 问题聚类分析
选择聚类数量

自动对问题进行聚类

查看每个聚类的示例问题

📊 系统要求
Python 3.9+

Neo4j 数据库（云端或本地）

MongoDB 数据库（云端或本地）

