# UNTESTED CODE FROM DEEPSEEK
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# 1. 生成模拟数据（修正概率计算）
def generate_simulated_data(n_countries=200, n_olympics=5):
    """
    生成模拟数据，包含以下特征：
    - country_id: 国家ID
    - gdp_per_capita: 人均GDP（对数正态分布）
    - population: 人口基数（对数正态分布）
    - athletes_count: 参赛运动员数量（与人口正相关）
    - prev_participations: 往届参赛次数
    - is_host: 是否为主办国（二分类）
    - has_won_medal: 是否首次获奖（标签）
    """
    np.random.seed(42)
    data = []
    for olympic in range(n_olympics):
        for country in range(n_countries):
            gdp = np.exp(np.random.normal(8, 1))  # 调整参数范围
            population = np.exp(np.random.normal(12, 0.5))  # 降低人口基数
            athletes = int(max(1, np.random.poisson(population * 0.0001)))
            participations = np.random.randint(1, 10)
            is_host = 1 if np.random.rand() < 0.05 else 0
            # 使用逻辑回归的线性组合 + Sigmoid函数计算概率
            logit = (
                -4  # 截距项（控制基准概率）
                + 0.5 * np.log(gdp) / 20  # GDP影响（标准化后）
                + 0.3 * athletes / 50  # 运动员数量影响
                + 1.5 * is_host  # 主办国优势
            )
            prob = 1 / (1 + np.exp(-logit))  # Sigmoid函数约束概率在[0,1]
            has_won = 1 if np.random.rand() < prob else 0
            data.append(
                [country, gdp, population, athletes, participations, is_host, has_won]
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


# 生成模拟历史数据
historical_data = generate_simulated_data()
print("模拟数据示例：\n", historical_data.head())

# 2. 数据预处理（增加空数据检查）
# 移除已获奖国家
historical_data_filtered = historical_data.groupby("country_id").filter(
    lambda x: x["has_won"].cumsum().eq(0).all()
)

if historical_data_filtered.empty:
    raise ValueError("过滤后数据为空！请检查数据生成逻辑或调整参数。")

# 特征与标签
X = historical_data_filtered[
    ["gdp", "population", "athletes", "participations", "is_host"]
]
y = historical_data_filtered["has_won"]

# 3. 处理类别不平衡
model = make_pipeline(
    StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
)

# 划分训练集与测试集（确保最小样本数）
if len(X) < 2:
    raise ValueError("样本量不足以划分训练集和测试集。")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 模型评估
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("\n模型评估：")
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.2f}")
print("混淆矩阵：\n", confusion_matrix(y_test, y_pred))


# 6. 预测下一届奥运会
def predict_next_olympics(model, n_candidate_countries=50):
    # 生成候选国家特征（从未获奖的国家）
    candidate_data = generate_simulated_data(
        n_countries=n_candidate_countries, n_olympics=1
    )
    X_candidate = candidate_data[
        ["gdp", "population", "athletes", "participations", "is_host"]
    ]
    # 计算每个国家的获奖概率
    candidate_data["win_prob"] = model.predict_proba(X_candidate)[:, 1]
    # 计算预期首次获奖国家数
    expected_wins = candidate_data["win_prob"].sum()
    return expected_wins


# 预测结果
expected_wins = predict_next_olympics(model)
print(f"\n预测结果：下一届奥运会预计有 {expected_wins:.1f} 个国家首次获奖。")

# 置信区间（泊松分布）
alpha = 0.95
lower = np.percentile(np.random.poisson(expected_wins, 10000), (1 - alpha) * 100)
upper = np.percentile(np.random.poisson(expected_wins, 10000), alpha * 100)
print(f"95% 置信区间：[{lower:.1f}, {upper:.1f}]")
