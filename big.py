import streamlit as st
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re
from collections import Counter
import time
import json
from datetime import datetime

# ========================
# 页面配置
# ========================
st.set_page_config(
    page_title="WikiMultiHop 知识图谱分析平台",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# 数据库连接
# ========================
@st.cache_resource
def init_connections():
    """初始化数据库连接"""
    try:
        neo4j_driver = GraphDatabase.driver(
            "neo4j+s://398bb8cd.databases.neo4j.io",
            auth=("398bb8cd", "WzUky0y7CXAo-nmxahMwsn5eoP1u-HUm1_zimdtMl8A")
        )
        
        mongo_client = MongoClient(
            "mongodb+srv://zhh:123456Qwe@cluster0.7sonwry.mongodb.net/?appName=Cluster0"
        )
        mongo_db = mongo_client["hotpotqa"]
        mongo_col = mongo_db["questions"]
        
        mongo_col.count_documents({})
        st.sidebar.success("✅ 数据库连接成功")
        
        return neo4j_driver, mongo_col
    except Exception as e:
        st.sidebar.error(f"❌ 数据库连接失败: {e}")
        return None, None

# ========================
# Neo4j查询函数
# ========================
def run_neo4j_query(driver, query, params=None):
    """执行Neo4j查询"""
    try:
        with driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    except Exception as e:
        st.error(f"Neo4j查询错误: {e}")
        return []

def get_question_from_neo4j(driver, qid):
    """从Neo4j获取问题"""
    query = """
    MATCH (q:Question {id: $qid})
    OPTIONAL MATCH (q)-[:MENTIONS]->(e:Entity)
    RETURN q.id as id, q.text as question, collect(distinct e.name) as entities
    """
    result = run_neo4j_query(driver, query, {"qid": qid})
    return result[0] if result else None

def get_all_questions_from_neo4j(driver, limit=100):
    """从Neo4j获取所有问题"""
    query = """
    MATCH (q:Question)
    RETURN q.id as id, q.text as question
    LIMIT $limit
    """
    return run_neo4j_query(driver, query, {"limit": limit})

def get_all_entities(driver, limit=100):
    """获取所有实体"""
    query = """
    MATCH (e:Entity)
    RETURN e.name as name
    ORDER BY e.name
    LIMIT $limit
    """
    return run_neo4j_query(driver, query, {"limit": limit})

def get_questions_by_entity(driver, mongo_col, entity_name, limit=20):
    """通过实体查找相关的问题"""
    # 从Neo4j获取问题ID
    query = """
    MATCH (q:Question)-[:MENTIONS]->(e:Entity {name: $entity})
    RETURN q.id as qid, q.text as question
    LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(query, {"entity": entity_name, "limit": limit})
        questions = [record.data() for record in result]
    
    # 从MongoDB补充详细信息
    for q in questions:
        mongo_data = mongo_col.find_one({"_id": q['qid']})
        if mongo_data:
            q['answer'] = mongo_data.get('answer', 'N/A')
            q['type'] = mongo_data.get('type', 'N/A')
    
    return questions

def get_related_entities(driver, entity_name, limit=20):
    """获取相关的实体（通过共同问题）"""
    query = """
    MATCH (e1:Entity {name: $entity})-[:MENTIONS]-(q:Question)-[:MENTIONS]-(e2:Entity)
    WHERE e2.name <> $entity
    RETURN e2.name as related_entity, count(DISTINCT q) as cooccurrence
    ORDER BY cooccurrence DESC
    LIMIT $limit
    """
    return run_neo4j_query(driver, query, {"entity": entity_name, "limit": limit})

def search_entities(driver, keyword, limit=20):
    """搜索实体"""
    query = """
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower($keyword)
    RETURN e.name as name
    ORDER BY e.name
    LIMIT $limit
    """
    return run_neo4j_query(driver, query, {"keyword": keyword, "limit": limit})

def get_hot_entities(driver, limit=10):
    """获取热门实体"""
    query = """
    MATCH (e:Entity)<-[:MENTIONS]-(q:Question)
    RETURN e.name as entity, count(DISTINCT q) as question_count
    ORDER BY question_count DESC
    LIMIT $limit
    """
    return run_neo4j_query(driver, query, {"limit": limit})

# ========================
# MongoDB查询函数
# ========================
def search_mongodb(mongo_col, query_text, limit=50):
    """在MongoDB中搜索问题"""
    if mongo_col is None:
        return []
    
    try:
        results = list(mongo_col.find(
            {
                "$or": [
                    {"question": {"$regex": query_text, "$options": "i"}},
                    {"_id": {"$regex": query_text, "$options": "i"}}
                ]
            },
            {"_id": 1, "question": 1, "answer": 1, "type": 1}
        ).limit(limit))
        
        return results
    except Exception as e:
        st.error(f"MongoDB查询错误: {e}")
        return []

def get_all_mongodb_questions(mongo_col, limit=100):
    """获取MongoDB中的所有问题"""
    try:
        results = list(mongo_col.find(
            {},
            {"_id": 1, "question": 1, "answer": 1, "type": 1}
        ).limit(limit))
        return results
    except Exception as e:
        st.error(f"MongoDB查询错误: {e}")
        return []

def get_full_data_by_id(mongo_col, qid):
    """获取问题的完整数据"""
    try:
        return mongo_col.find_one({"_id": qid})
    except Exception as e:
        st.error(f"获取详情失败: {e}")
        return None

# ========================
# 聚类函数
# ========================
def cluster_questions(df, n_clusters=5):
    """对问题进行聚类"""
    if len(df) < n_clusters:
        n_clusters = max(2, len(df))
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['question'].fillna(''))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf_matrix.toarray())
    
    return clusters, coords, vectorizer.get_feature_names_out()

# ========================
# 可视化函数
# ========================
def create_entity_network(entity_name, related_entities):
    """创建实体关系网络图"""
    if not related_entities:
        return None
    
    import math
    labels = [entity_name] + [r['related_entity'] for r in related_entities[:8]]
    sizes = [30] + [r['cooccurrence'] * 5 + 15 for r in related_entities[:8]]
    
    n = len(labels)
    angles = [i * 2 * math.pi / n for i in range(n)]
    x = [math.cos(angle) * 2 for angle in angles]
    y = [math.sin(angle) * 2 for angle in angles]
    
    fig = go.Figure()
    
    # 添加连接线
    for i in range(1, n):
        fig.add_trace(go.Scatter(
            x=[x[0], x[i]],
            y=[y[0], y[i]],
            mode='lines',
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # 添加节点
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text',
        marker=dict(size=sizes, color='lightblue', line=dict(width=2, color='darkblue')),
        text=labels,
        textposition="top center",
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"实体关系网络 - {entity_name}",
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3])
    )
    
    return fig

# ========================
# 主界面
# ========================
def main():
    st.title("🔍 WikiMultiHop 多跳推理知识图谱平台")
    st.markdown("---")
    
    # 初始化session state
    if 'selected_qid' not in st.session_state:
        st.session_state.selected_qid = None
    
    # 初始化连接
    neo4j_driver, mongo_col = init_connections()
    
    if neo4j_driver is None:
        st.error("无法连接到数据库，请检查网络")
        return
    
    # 侧边栏导航
    st.sidebar.title("📊 功能导航")
    page = st.sidebar.radio(
        "选择功能",
        ["📝 问题检索与查询", 
         "🔗 实体关系探索", 
         "📊 知识图谱可视化",
         "🎯 问题聚类分析",
         "📈 数据统计与仪表板",
         "💾 批量数据导出"]
    )
    
    # ========================
    # 1. 问题检索与查询（保持原样）
    # ========================
    if page == "📝 问题检索与查询":
        st.header("📝 问题检索与查询")
        
        # 搜索功能
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 输入关键词搜索问题", 
                                       placeholder="例如: Who, What, When, Where...",
                                       help="支持中英文搜索")
        with col2:
            limit = st.number_input("显示数量", min_value=1, max_value=100, value=20)
        
        # 显示所有问题的按钮
        if st.button("📋 显示所有问题"):
            search_term = ""
            st.rerun()
        
        if search_term:
            with st.spinner(f"正在搜索 '{search_term}'..."):
                results = search_mongodb(mongo_col, search_term, limit)
                
                if results:
                    st.success(f"✅ 找到 {len(results)} 个相关问题")
                    
                    # 创建选择框
                    question_dict = {}
                    for r in results:
                        display_text = f"{r.get('question', 'N/A')[:100]}..."
                        if len(results) > 1:
                            display_text = f"[{r.get('type', 'N/A')}] {display_text}"
                        question_dict[display_text] = r['_id']
                    
                    selected_label = st.selectbox("选择要查看的问题", list(question_dict.keys()))
                    selected_id = question_dict[selected_label]
                    
                    # 获取详细信息
                    full_data = get_full_data_by_id(mongo_col, selected_id)
                    neo4j_data = get_question_from_neo4j(neo4j_driver, selected_id)
                    
                    if full_data:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📖 问题详情")
                            st.markdown(f"**ID:** `{full_data.get('_id', 'N/A')}`")
                            st.markdown(f"**问题:** {full_data.get('question', 'N/A')}")
                            st.markdown(f"**答案:** {full_data.get('answer', 'N/A')}")
                            st.markdown(f"**类型:** {full_data.get('type', 'N/A')}")
                            
                            # 显示上下文
                            if full_data.get('context'):
                                with st.expander("📄 查看上下文 (前5句)"):
                                    for ctx in full_data['context'][:3]:
                                        if isinstance(ctx, dict):
                                            st.markdown(f"**文档: {ctx.get('title', 'Unknown')}**")
                                            sentences = ctx.get('sentences', [])
                                            for i, sent in enumerate(sentences[:5]):
                                                st.markdown(f"- {sent}")
                                        elif isinstance(ctx, list) and len(ctx) >= 2:
                                            st.markdown(f"**文档: {ctx[0]}**")
                                            for i, sent in enumerate(ctx[1][:5]):
                                                st.markdown(f"- {sent}")
                        
                        with col2:
                            st.subheader("🎯 知识图谱关联")
                            if neo4j_data:
                                entities = neo4j_data.get('entities', [])
                                if entities:
                                    st.markdown(f"**关联实体:**")
                                    for e in entities[:15]:
                                        st.markdown(f"- {e}")
                                else:
                                    st.info("暂无关联实体")
                            else:
                                st.info("Neo4j中暂无该问题的图谱数据")
                else:
                    st.warning(f"未找到包含 '{search_term}' 的问题")
        
        else:
            # 显示示例问题
            st.subheader("📋 示例问题")
            sample_questions = get_all_mongodb_questions(mongo_col, 20)
            if sample_questions:
                for q in sample_questions[:10]:
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{q.get('question', 'N/A')[:150]}**")
                        with col2:
                            if st.button("查看", key=f"btn_{q['_id']}"):
                                st.session_state.selected_qid = q['_id']
                                st.rerun()
                        st.markdown("---")
        
        # 显示选中的问题详情
        if st.session_state.selected_qid:
            st.markdown("---")
            st.header("📖 问题详情")
            
            full_data = get_full_data_by_id(mongo_col, st.session_state.selected_qid)
            if full_data:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**问题:** {full_data.get('question', 'N/A')}")
                    st.markdown(f"**答案:** {full_data.get('answer', 'N/A')}")
                    st.markdown(f"**类型:** {full_data.get('type', 'N/A')}")
                with col2:
                    if st.button("关闭"):
                        st.session_state.selected_qid = None
                        st.rerun()
    
    # ========================
    # 2. 实体关系探索（新功能）
    # ========================
    elif page == "🔗 实体关系探索":
        st.header("🔗 实体关系探索")
        
        # 热门实体推荐
        st.subheader("🔥 热门实体")
        hot_entities = get_hot_entities(neo4j_driver, 10)
        if hot_entities:
            cols = st.columns(5)
            for i, entity in enumerate(hot_entities[:10]):
                with cols[i % 5]:
                    st.button(entity['entity'], key=f"hot_{entity['entity']}")
                    if st.session_state.get(f"hot_{entity['entity']}"):
                        st.session_state.selected_entity = entity['entity']
        
        st.markdown("---")
        
        # 实体搜索
        entity_search = st.text_input("🔎 搜索实体", placeholder="输入实体名称...")
        
        if entity_search:
            entities = search_entities(neo4j_driver, entity_search)
            entity_names = [e['name'] for e in entities]
        else:
            entities = get_all_entities(neo4j_driver, 50)
            entity_names = [e['name'] for e in entities]
        
        if entity_names:
            selected_entity = st.selectbox("选择实体查看详情", entity_names)
            
            if selected_entity:
                st.header(f"📌 实体: {selected_entity}")
                
                with st.spinner("加载数据..."):
                    related = get_related_entities(neo4j_driver, selected_entity)
                    questions = get_questions_by_entity(neo4j_driver, mongo_col, selected_entity)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"🔗 关联实体 ({len(related)})")
                    if related:
                        for r in related[:10]:
                            st.markdown(f"**• {r['related_entity']}** (共现: {r['cooccurrence']}次)")
                        
                        fig = create_entity_network(selected_entity, related)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("暂无关联实体")
                
                with col2:
                    st.subheader(f"📝 相关问题 ({len(questions)})")
                    if questions:
                        for q in questions[:10]:
                            with st.expander(f"问题: {q['question'][:80]}..."):
                                st.markdown(f"**完整问题:** {q['question']}")
                                st.markdown(f"**答案:** {q.get('answer', 'N/A')}")
                                st.markdown(f"**类型:** {q.get('type', 'N/A')}")
                                st.markdown(f"**ID:** `{q['qid']}`")
                    else:
                        st.info("该实体暂无相关问题")
    
    # ========================
    # 3. 知识图谱可视化
    # ========================
    elif page == "📊 知识图谱可视化":
        st.header("📊 知识图谱可视化")
        
        option = st.radio("选择模式", ["按问题ID查看", "按实体名称查看"])
        
        if option == "按问题ID查看":
            qid = st.text_input("输入问题ID", placeholder="例如: 5a7b9c8d_001")
            if qid:
                with st.spinner("加载中..."):
                    neo4j_data = get_question_from_neo4j(neo4j_driver, qid)
                    if neo4j_data:
                        entities = neo4j_data.get('entities', [])
                        if entities:
                            st.subheader("关联实体")
                            for e in entities:
                                st.markdown(f"- {e}")
                        else:
                            st.info("该问题暂无关联实体")
                    else:
                        st.warning("未找到该问题")
        
        else:
            entity_name = st.text_input("输入实体名称", placeholder="例如: Einstein")
            if entity_name:
                with st.spinner("加载中..."):
                    related = get_related_entities(neo4j_driver, entity_name)
                    if related:
                        st.subheader(f"与 '{entity_name}' 相关的实体")
                        for r in related[:15]:
                            st.markdown(f"- {r['related_entity']} (共现: {r['cooccurrence']}次)")
                    else:
                        st.info("未找到相关实体")
    
    # ========================
    # 4. 问题聚类分析（保持原样）
    # ========================
    elif page == "🎯 问题聚类分析":
        st.header("🎯 问题聚类分析")
        
        n_clusters = st.slider("聚类数量", 2, 8, 4)
        
        if st.button("开始聚类分析", type="primary"):
            with st.spinner("正在分析问题聚类..."):
                questions_data = get_all_mongodb_questions(mongo_col, 200)
                
                if len(questions_data) >= n_clusters:
                    df = pd.DataFrame(questions_data)
                    
                    clusters, coords, features = cluster_questions(df, n_clusters)
                    df['cluster'] = clusters
                    df['x'] = coords[:, 0]
                    df['y'] = coords[:, 1]
                    
                    fig = px.scatter(
                        df, x='x', y='y', color='cluster',
                        hover_data=['question', 'type'],
                        title=f"问题聚类可视化 ({n_clusters}个簇)",
                        color_continuous_scale='Viridis'
                    )
                    fig.update_traces(marker=dict(size=10))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    for i in range(n_clusters):
                        cluster_df = df[df['cluster'] == i]
                        with st.expander(f"聚类 {i} - {len(cluster_df)} 个问题"):
                            for q in cluster_df['question'].head(5):
                                st.markdown(f"- {q[:100]}...")
                else:
                    st.warning(f"数据不足，需要至少 {n_clusters} 个问题")
    
    # ========================
    # 5. 数据统计与仪表板
    # ========================
    elif page == "📈 数据统计与仪表板":
        st.header("📈 数据统计与仪表板")
        
        with st.spinner("加载统计数据..."):
            # 问题类型分布
            try:
                pipeline = [
                    {"$group": {"_id": "$type", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}}
                ]
                type_dist = list(mongo_col.aggregate(pipeline))
                
                if type_dist:
                    st.subheader("问题类型分布")
                    df_types = pd.DataFrame(type_dist)
                    fig = px.pie(df_types, values='count', names='_id', title="问题类型占比")
                    st.plotly_chart(fig, use_container_width=True)
            except:
                pass
            
            # 热门实体
            hot_entities = get_hot_entities(neo4j_driver, 20)
            if hot_entities:
                st.subheader("热门实体 Top 20")
                df_hot = pd.DataFrame(hot_entities)
                fig = px.bar(df_hot, x='entity', y='question_count', title="实体相关问题数量")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    # ========================
    # 6. 批量数据导出
    # ========================
    elif page == "💾 批量数据导出":
        st.header("💾 批量数据导出")
        
        export_limit = st.number_input("导出数量", min_value=10, max_value=1000, value=100, step=50)
        
        if st.button("开始导出", type="primary"):
            with st.spinner(f"正在导出 {export_limit} 条数据..."):
                questions = list(mongo_col.find({}, {"_id": 1, "question": 1, "answer": 1, "type": 1}).limit(export_limit))
                
                export_data = []
                for q in questions:
                    export_data.append({
                        "id": str(q['_id']),
                        "question": q.get('question', ''),
                        "answer": q.get('answer', ''),
                        "type": q.get('type', '')
                    })
                
                json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
                
                st.download_button(
                    label="📥 下载JSON文件",
                    data=json_str,
                    file_name=f"questions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                st.success(f"✅ 成功导出 {len(questions)} 条数据")

if __name__ == "__main__":
    main()
