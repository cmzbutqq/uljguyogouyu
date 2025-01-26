import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

'''
在与deepseek缠斗了几回合后
这段代码终于能运行了
'''


# 1. 生成模拟数据（修复国家ID和概率计算）
def generate_simulated_data(n_countries=200, n_olympics=5):
    np.random.seed(42)
    data = []
    for olympic in range(n_olympics):
        for country in range(n_countries):
            country_id = country  # 固定国家ID（跨届保持相同）
            gdp = np.exp(np.random.normal(8, 1))  # 调整GDP范围
            population = np.exp(np.random.normal(12, 0.5))  # 合理人口基数
            athletes = int(
                max(1, population * 0.001 + np.random.poisson(5))
            )  # 运动员基数
            participations = olympic + 1  # 参赛次数递增
            is_host = 1 if np.random.rand() < 0.05 else 0  # 5%概率为主办国
            # 调整logit参数确保合理概率分布
            logit = (
                -4  # 截距项（控制基准概率）
                + 0.5 * (np.log(gdp) - 8)  # 标准化GDP影响
                + 0.3 * athletes / 10  # 运动员数量影响
                + 1.2 * is_host  # 主办国优势
            )
            prob = 1 / (1 + np.exp(-logit))
            has_won = 1 if np.random.rand() < prob else 0
            data.append(
                [
                    country_id,
                    gdp,
                    population,
                    athletes,
                    participations,
                    is_host,
                    has_won,
                ]
            )
    df = pd.DataFrame(
        data,
        columns=[
            "country_id",
            "gdp",
            "population",
            "athletes",
            "participations",
            "is_host",
            "has_won",
        ],
    )
    return df


# 生成数据并验证正样本比例
historical_data = generate_simulated_data()
print("正样本比例:", historical_data["has_won"].mean())

# 2. 数据预处理（正确过滤历史获奖国家）
historical_data = historical_data.sort_values(["country_id", "participations"])
historical_data_filtered = historical_data.groupby("country_id").filter(
    lambda group: group["has_won"].cumsum().shift(1, fill_value=0).eq(0).all()
)

# 强制保留至少10%正样本
if historical_data_filtered["has_won"].mean() < 0.05:
    raise ValueError("正样本比例不足5%，请调整数据生成参数")

# 3. 划分数据集（强制分层抽样）
X = historical_data_filtered[
    ["gdp", "population", "athletes", "participations", "is_host"]
]
y = historical_data_filtered["has_won"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 训练模型（使用弹性网正则化）
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        C=0.5,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=2000,
    ),
)
model.fit(X_train, y_train)

# 5. 模型评估
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("\n模型评估：")
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.2f}")
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))


# 6. 预测下一届奥运会
def predict_next_olympics(model):
    # 生成候选国家（仅包含未获奖国家）
    candidate_data = generate_simulated_data(n_olympics=1, n_countries=100)
    candidate_data = candidate_data.groupby("country_id").filter(
        lambda group: group["has_won"].cumsum().eq(0).all()
    )
    X_candidate = candidate_data[
        ["gdp", "population", "athletes", "participations", "is_host"]
    ]
    candidate_data["win_prob"] = model.predict_proba(X_candidate)[:, 1]
    return candidate_data["win_prob"].sum()


expected_wins = predict_next_olympics(model)
print(f"\n预测结果: 下一届奥运会预计有 {expected_wins:.1f} 个国家首次获奖")
