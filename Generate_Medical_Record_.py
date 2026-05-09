import streamlit as st
import openai
import json
import os
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

# LangChain 相关
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
# ... existing code ...

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="智能病历生成系统 · AI Doctor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 数据持久化路径 ====================
FEEDBACK_FILE = "feedback.json"
CHROMA_DIR = "./chroma_db"
KNOWLEDGE_BASE_DIR = "./knowledge_base"

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)


# ==================== 会话状态初始化 ====================
def init_session_state():
    """初始化所有会话状态"""
    state_defaults = {
        'generated_history': [],
        'feedback_data': [],
        'knowledge_base_ready': False,
        'vectorstore': None,
        'embeddings_loaded': False
    }

    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            if key == 'feedback_data' and os.path.exists(FEEDBACK_FILE):
                try:
                    with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                        st.session_state[key] = json.load(f)
                except (json.JSONDecodeError, IOError):
                    st.session_state[key] = []
            else:
                st.session_state[key] = default_value


init_session_state()


# ==================== 助手函数 ====================

@st.cache_resource
def init_embeddings():
    """初始化中文嵌入模型（带缓存，避免重复加载）"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        st.session_state.embeddings_loaded = True
        return embeddings
    except Exception as e:
        st.error(f"嵌入模型加载失败: {e}")
        st.session_state.embeddings_loaded = False
        return None


def init_deepseek_client(api_key: str) -> openai.OpenAI:
    """初始化 DeepSeek 客户端"""
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )


def load_document_to_vectorstore(file_path: str, embeddings) -> bool:
    """加载文档到向量数据库"""
    try:
        # 文件类型判断
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(('.txt', '.md')):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            st.error(f"不支持的文件类型: {file_path}")
            return False

        documents = loader.load()

        if not documents:
            st.warning("文档内容为空，请检查文件。")
            return False

        # 文本分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        docs = text_splitter.split_documents(documents)

        # 添加到向量数据库
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = Chroma.from_documents(
                docs,
                embeddings,
                persist_directory=CHROMA_DIR
            )
        else:
            st.session_state.vectorstore.add_documents(docs)

        st.session_state.vectorstore.persist()
        return True

    except Exception as e:
        st.error(f"加载文档失败: {e}")
        return False


def retrieve_relevant_context(query: str, k: int = 3) -> str:
    """从知识库检索相关内容"""
    if st.session_state.vectorstore is None:
        return ""

    try:
        docs = st.session_state.vectorstore.similarity_search(query, k=k)
        if docs:
            # 去重并合并
            seen = set()
            unique_docs = []
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)

            context = "\n\n---\n\n".join([doc.page_content for doc in unique_docs])
            return context
        return ""
    except Exception as e:
        st.warning(f"检索失败: {e}")
        return ""


def build_prompt(patient_info: Dict[str, Any], context: str = "") -> str:
    """构建病历生成提示词"""

    # 科室专用提示
    disease_specific_prompts = {
        "心内科": "重点关注血压、心率、既往心血管病史（如高血压、冠心病、心衰）、用药史（如阿司匹林、他汀类）。",
        "呼吸内科": "重点关注吸烟史、过敏史、肺部基础疾病（如哮喘、COPD）、最近是否接触过流感或新冠患者。",
        "儿科": "重点关注生长发育情况、疫苗接种史、出生时情况、喂养史、家族遗传病史。",
        "妇科": "重点关注月经史（初潮、周期、经期、末次月经）、孕产史、妇科手术史、激素使用情况。",
        "男科": "重点关注泌尿系统症状、性功能情况、前列腺相关检查史。"
    }

    disease_specific_prompt = disease_specific_prompts.get(
        patient_info['department'],
        "重点关注既往病史、用药史、过敏史、家族遗传病史。"
    )

    prompt = f"""你是一位经验丰富的临床医生。请根据以下患者信息，撰写一份专业、详细的电子病历。

### 病历书写要求
**输出格式**：请使用 Markdown 格式输出，包含以下章节标题（使用 ## 级别）：
- ## 主诉
- ## 现病史
- ## 既往史
- ## 体格检查
- ## 辅助检查
- ## 初步诊断
- ## 治疗建议

### 患者信息
- **性别**: {patient_info['gender']}
- **年龄**: {patient_info['age']} 岁
- **科室**: {patient_info['department']}
- **病种**: {patient_info['disease']}

### 症状描述
患者出现以下症状：{', '.join(patient_info['symptoms'])}。

### 病程持续时间
{patient_info['duration']}

### 专科重点关注
{disease_specific_prompt}

### 参考信息（来自过往病历或知识库）
{context if context else "无相关参考信息。"}

### 注意事项
1. 病历语言应专业、简洁、规范，使用医学术语。
2. 请根据专科重点关注要求，体现不同科室的病历特点。
3. 直接输出病历内容，不要添加额外解释。"""

    return prompt


def generate_medical_record(client, patient_info: Dict[str, Any], context: str = "") -> str:
    """调用 DeepSeek 生成病历"""

    prompt = build_prompt(patient_info, context)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业严谨的临床医生，负责撰写病历。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content

    except openai.APIError as e:
        st.error(f"API调用失败 (状态码: {e.status_code}): {e.message}")
        return generate_mock_record(patient_info)
    except openai.APIConnectionError:
        st.error("无法连接到DeepSeek API，请检查网络连接。")
        return generate_mock_record(patient_info)
    except openai.RateLimitError:
        st.error("API调用频率超限，请稍后重试。")
        return generate_mock_record(patient_info)
    except Exception as e:
        st.error(f"未知错误: {e}")
        return generate_mock_record(patient_info)


def generate_mock_record(patient_info: Dict[str, Any]) -> str:
    """生成模拟病历（API不可用时的降级方案）"""

    gender = patient_info['gender']
    age = patient_info['age']
    symptoms = '、'.join(patient_info['symptoms'])
    duration = patient_info['duration']
    disease = patient_info['disease']

    return f"""## 主诉
{gender}性，{age}岁，"{symptoms}" {duration}。

## 现病史
患者于{duration}前无明显诱因出现{symptoms}，呈进行性加重，无发热、寒战等全身症状。发病以来精神、食欲、睡眠尚可，大小便正常。

## 既往史
平素体健，否认高血压、糖尿病等慢性病史。否认药物及食物过敏史。否认手术外伤史。

## 体格检查
T 36.5℃， P 72次/分， R 18次/分， BP 120/80mmHg。
神志清楚，查体合作。双肺呼吸音清，未闻及干湿啰音。心率72次/分，律齐，各瓣膜听诊区未闻及病理性杂音。

## 辅助检查
暂缺（建议完善血常规、CRP、胸部X线等检查）。

## 初步诊断
{disease}待查

## 治疗建议
1. 完善上述辅助检查以明确诊断。
2. 对症支持治疗。
3. 随诊。

*(此为系统自动生成的模拟病历，仅供参考。请配置有效的API Key以获取高质量病历。)*"""


def record_feedback(original_text: str, revised_text: str, patient_info: Dict[str, Any]) -> bool:
    """记录用户反馈"""
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "original": original_text,
        "revised": revised_text,
        "patient_info": {
            k: v for k, v in patient_info.items() if k != 'symptoms'
        },
        "symptoms": patient_info.get('symptoms', [])
    }

    st.session_state.feedback_data.append(feedback_entry)

    try:
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.feedback_data, f, ensure_ascii=False, indent=2)
        return True
    except IOError as e:
        st.error(f"反馈保存失败: {e}")
        return False


def list_knowledge_files() -> List[str]:
    """列出知识库中的文件"""
    try:
        files = os.listdir(KNOWLEDGE_BASE_DIR)
        return [f for f in files if os.path.isfile(os.path.join(KNOWLEDGE_BASE_DIR, f))]
    except FileNotFoundError:
        return []


# ==================== 模拟数据 ====================

DISEASE_MAP = {
    "呼吸内科": ["上呼吸道感染", "肺炎", "支气管哮喘", "慢性阻塞性肺疾病", "肺结节"],
    "消化内科": ["胃炎", "消化性溃疡", "肠炎", "肝炎", "胰腺炎"],
    "心内科": ["高血压", "冠心病", "心律失常", "心力衰竭", "心肌炎"],
    "神经内科": ["偏头痛", "脑卒中", "帕金森病", "癫痫", "周围神经病"],
    "儿科": ["儿童发热", "婴幼儿腹泻", "小儿肺炎", "儿童哮喘", "手足口病"],
    "妇科": ["月经不调", "阴道炎", "盆腔炎", "子宫肌瘤", "更年期综合征"],
    "男科": ["前列腺炎", "性功能障碍", "前列腺增生", "男性不育", "睾丸炎"],
    "皮肤科": ["湿疹", "荨麻疹", "银屑病", "痤疮", "带状疱疹"],
    "骨科": ["颈椎病", "腰椎间盘突出症", "关节炎", "骨折", "骨质疏松"],
    "眼科": ["结膜炎", "白内障", "青光眼", "近视", "干眼症"],
    "耳鼻喉科": ["中耳炎", "鼻炎", "咽炎", "扁桃体炎", "耳鸣"],
    "口腔科": ["龋齿", "牙髓炎", "牙周炎", "口腔溃疡", "智齿冠周炎"]
}

# 按器官系统分类的症状
SYMPTOM_CATEGORIES = {
    "全身症状": ["发热", "寒战", "乏力", "消瘦", "盗汗", "食欲不振"],
    "呼吸系统": ["咳嗽", "咳痰", "流涕", "鼻塞", "咽痛", "胸痛", "胸闷", "气短", "呼吸困难"],
    "消化系统": ["腹痛", "腹胀", "腹泻", "便秘", "恶心", "呕吐", "便血"],
    "神经系统": ["头痛", "头晕", "失眠", "多梦", "健忘", "焦虑", "眩晕"],
    "心血管系统": ["心悸", "胸痛", "胸闷", "水肿", "紫绀"],
    "泌尿生殖系统": ["尿频", "尿急", "尿痛", "血尿", "月经不调", "痛经", "白带异常"],
    "运动系统": ["关节痛", "腰痛", "肌肉酸痛", "肢体麻木"],
    "皮肤": ["皮疹", "瘙痒", "干燥", "脱屑"]
}

# 展平症状列表（保留分类，用于搜索）
ALL_SYMPTOMS = []
for category, symptoms in SYMPTOM_CATEGORIES.items():
    for symptom in symptoms:
        ALL_SYMPTOMS.append({"symptom": symptom, "category": category})

DURATION_OPTIONS = {
    "1-3天": "3天",
    "4-7天": "1周",
    "1-2周": "2周",
    "3-4周": "1个月",
    "1-3个月": "3个月",
    "3-6个月": "半年",
    "6个月以上": "半年余",
    "数年": "数年"
}

# ==================== Streamlit UI ====================

# 标题
st.title("🏥 智能病历生成系统 · AI Doctor")
st.markdown("基于 **DeepSeek 大模型** 和 **RAG 检索增强**，一键生成专业、个性化的电子病历。")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 配置")

    # API Key配置
    api_key = st.text_input(
        "DeepSeek API Key:",
        type="password",
        placeholder="请输入您的 API Key",
        help="在 https://platform.deepseek.com 获取 API Key。不配置则使用模拟数据。"
    )

    if not api_key:
        st.warning("⚠️ 未配置 API Key，系统将使用模拟数据运行。")
        st.caption("模拟数据仅用于演示功能，请配置 API Key 以体验完整功能。")

    # 模型参数
    with st.expander("🔧 高级参数", expanded=False):
        temperature = st.slider("温度 (Temperature)", 0.0, 1.0, 0.7,
                                help="控制生成文本的创造性：0.0 为最保守，1.0 为最创造")
        max_tokens = st.slider("最大 Token 数", 512, 4096, 2048, step=256,
                               help="控制生成文本的最大长度")

    st.divider()

    # 知识库管理
    st.header("📚 知识库管理")

    uploaded_file = st.file_uploader(
        "导入过往病历或医学文档",
        type=['txt', 'md', 'pdf', 'docx'],
        help="支持 .txt, .md, .pdf, .docx 格式"
    )

    if uploaded_file is not None:
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"正在处理文件: {uploaded_file.name}..."):
            embeddings = init_embeddings()
            if embeddings:
                success = load_document_to_vectorstore(file_path, embeddings)
                if success:
                    st.success(f"✅ '{uploaded_file.name}' 已导入并索引！")
                    st.session_state.knowledge_base_ready = True
                else:
                    st.error(f"'{uploaded_file.name}' 处理失败。")

    # 显示已导入文件
    knowledge_files = list_knowledge_files()
    if knowledge_files:
        st.subheader("已导入文件:")
        for f in knowledge_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(f"📄 {f}")
            col2.write(f"")
            if col3.button("🗑️", key=f"del_{f}", help="删除此文件"):
                try:
                    os.remove(os.path.join(KNOWLEDGE_BASE_DIR, f))
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")
    else:
        st.info("知识库为空，上传文件以启用 RAG 增强。")

    # 底部提示
    st.divider()
    st.caption("👨‍⚕️ **智能病历生成系统 v1.0**")
    st.caption("⚠️ 仅供演示使用，不构成医疗建议。")

# ==================== 主界面 ====================
tab1, tab2 = st.tabs(["📝 病历生成", "ℹ️ 关于系统"])

with tab1:
    col1, col2 = st.columns([1.5, 2.5])

    with col1:
        st.header("患者信息")

        # 基础信息
        gender = st.selectbox("性别:", ["男", "女", "其他"])
        age = st.slider("年龄:", 0, 120, 35, help="患者年龄，0表示婴儿")

        # 科室与病种
        department = st.selectbox("就诊科室:", options=list(DISEASE_MAP.keys()))

        disease_options = DISEASE_MAP.get(department, ["其他"])
        disease = st.selectbox("初步诊断/病种:", options=disease_options)

        # 症状选择（带分类和搜索）
        st.markdown("**症状选择**（可多选）:")

        # 搜索过滤
        symptom_search = st.text_input(
            "🔍 搜索症状:",
            placeholder="输入关键词过滤（如：发热、咳嗽...）"
        )

        # 根据搜索词过滤
        if symptom_search:
            filtered_symptoms = [
                s for s in ALL_SYMPTOMS
                if symptom_search.lower() in s["symptom"].lower()
            ]
        else:
            filtered_symptoms = ALL_SYMPTOMS

        # 按分类组织显示
        symptom_options = [s["symptom"] for s in filtered_symptoms]
        symptom_labels = {s["symptom"]: f"{s['symptom']} ( {s['category']} )"
                          for s in filtered_symptoms}

        selected_symptoms = st.multiselect(
            "从列表中选择症状:",
            options=symptom_options,
            format_func=lambda x: symptom_labels.get(x, x),
            default=[],
            placeholder="输入或选择症状..."
        )

        # 显示已选症状
        if selected_symptoms:
            st.success(f"✅ 已选症状: {', '.join(selected_symptoms)}")
        else:
            st.warning("请至少选择一个症状。")

        # 持续时间
        duration = st.selectbox(
            "症状持续时间:",
            options=list(DURATION_OPTIONS.keys()),
            help="选择症状持续的大致时间范围"
        )

        # 生成按钮
        generate_button = st.button(
            "🚀 生成病历",
            type="primary",
            use_container_width=True,
            disabled=not selected_symptoms
        )

    with col2:
        st.header("生成的病历")

        # 生成逻辑
        if generate_button:
            if not selected_symptoms:
                st.error("请至少选择一个症状！")
            else:
                # 准备患者信息
                patient_info = {
                    "gender": gender,
                    "age": age,
                    "department": department,
                    "disease": disease,
                    "duration": DURATION_OPTIONS[duration],
                    "symptoms": selected_symptoms
                }

                # 显示处理状态
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 步骤1：检索知识库
                status_text.info("🔍 正在检索知识库...")
                progress_bar.progress(25)

                context = ""
                if st.session_state.knowledge_base_ready or st.session_state.vectorstore is not None:
                    query = f"{gender} {age}岁 {department} {disease} {' '.join(selected_symptoms)}"
                    context = retrieve_relevant_context(query)
                    if context:
                        st.success("✅ 已检索到相关病历参考信息。")
                    else:
                        st.info("📖 没有找到相关的参考信息。")

                # 步骤2：生成病历
                status_text.info("🤖 AI 正在生成病历...")
                progress_bar.progress(50)

                if api_key:
                    client = init_deepseek_client(api_key)
                    result = generate_medical_record(client, patient_info, context)
                else:
                    status_text.info("📝 使用模拟数据生成（未配置 API Key）")
                    time.sleep(1)  # 模拟延迟
                    result = generate_mock_record(patient_info)

                progress_bar.progress(100)
                status_text.success("✅ 病历生成完成！")

                # 展示结果
                st.divider()

                # 渲染 Markdown
                st.markdown(result)

                # 编辑区域（用于反馈）
                with st.expander("✏️ 编辑病历内容（用于提交指正）", expanded=False):
                    edited_text = st.text_area(
                        "编辑病历内容:",
                        value=result,
                        height=300,
                        key="generated_text_area"
                    )

                    # 提交指正按钮
                    if st.button("✅ 提交指正", type="secondary", use_container_width=True):
                        if edited_text != result:
                            success = record_feedback(result, edited_text, patient_info)
                            if success:
                                st.success("🎉 感谢您的指正！反馈已记录，将用于优化模型。")
                            else:
                                st.error("反馈保存失败，请重试。")
                        else:
                            st.info("您没有修改病历内容，无需提交指正。")

                # 操作按钮
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    if st.button("📋 复制内容", use_container_width=True):
                        st.info("请手动复制上方病历内容 (Ctrl+C / Cmd+C)")
                with col_btn2:
                    if st.button("✏️ 润色病历", use_container_width=True):
                        st.info("润色功能将在后续版本实现。")
                with col_btn3:
                    if st.button("🔄 重新生成", use_container_width=True):
                        st.rerun()

                # 记录历史
                st.session_state.generated_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "patient": patient_info,
                    "content": result
                })

        # 显示历史记录
        if st.session_state.generated_history:
            with st.expander("📜 生成历史", expanded=False):
                for i, record in enumerate(reversed(st.session_state.generated_history)):
                    with st.container():
                        st.markdown(f"**{len(st.session_state.generated_history) - i}. {record['timestamp']}**")
                        st.caption(
                            f"{record['patient']['gender']} / {record['patient']['age']}岁 / {record['patient']['department']}")
                        preview = record['content'][:200] + "..." if len(record['content']) > 200 else record['content']
                        st.text(preview)
                        if i < len(st.session_state.generated_history) - 1:
                            st.divider()

with tab2:
    st.header("关于系统")

    st.markdown("""
    ### 系统架构

    本系统采用 **多层架构** 设计：

    1. **前端层**：Streamlit Web 应用
       - 提供友好的用户交互界面
       - 支持实时状态反馈

    2. **AI 层**：DeepSeek 大模型
       - 负责病历生成和润色
       - 支持个性化内容生成

    3. **知识层**：RAG 检索增强
       - 基于 ChromaDB 向量数据库
       - 支持文档导入和语义检索

    ### 技术栈

    | 组件 | 技术选型 | 作用 |
    |------|----------|------|
    | 前端框架 | Streamlit | Web 应用界面 |
    | 大模型 | DeepSeek API | 病历生成核心 |
    | RAG 框架 | LangChain | 检索增强 |
    | 向量数据库 | ChromaDB | 知识存储 |
    | 嵌入模型 | BAAI/bge-base-zh-v1.5 | 中文语义理解 |

    ### 使用流程

    1. **配置**：在侧边栏输入 DeepSeek API Key
    2. **输入**：选择患者信息、症状、持续时间
    3. **检索**：系统自动从知识库检索相关病历
    4. **生成**：AI 综合信息生成专业病历
    5. **反馈**：编辑优化后提交指正，持续改进

    ### 注意事项

    - **数据安全**：所有数据存储在本地，建议定期备份
    - **隐私保护**：请勿上传真实患者隐私信息
    - **免责声明**：系统生成内容仅供参考，不构成医疗建议
    """)

    # 系统状态
    st.divider()
    st.subheader("系统状态")

    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
    with col_status1:
        st.metric("知识库文件数", len(list_knowledge_files()))
    with col_status2:
        st.metric("已记录反馈数", len(st.session_state.feedback_data))
    with col_status3:
        st.metric("本次会话生成数", len(st.session_state.generated_history))
    with col_status4:
        embed_status = "✅ 已加载" if st.session_state.embeddings_loaded else "❌ 未加载"
        st.metric("嵌入模型", embed_status)

# ==================== 底部 ====================
st.sidebar.divider()
st.sidebar.caption(
    "👨‍⚕️ **AI Doctor v1.0**\n\n"
    "基于 DeepSeek + RAG\n\n"
)